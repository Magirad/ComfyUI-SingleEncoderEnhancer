from contextlib import ExitStack, contextmanager, nullcontext
import json
import logging
import math
import re
import time

import torch

import comfy.model_management as model_management
import comfy.sd
import comfy.utils
import node_helpers
import nodes
from comfy_api.latest import io

from .caches import (REWRITE_STORE, CachedVAE, blank_latent_shape, hash_memo, rewrite_key, text_encode_cache,
                     vision_feature_cache)


def unload_clip(patcher):
    # all_devices exists only in newer ComfyUI; older versions get the same call without it
    import inspect
    kwargs = {"unload_additional_models": False}
    if "all_devices" in inspect.signature(model_management.unload_model_and_clones).parameters:
        kwargs["all_devices"] = True
    model_management.unload_model_and_clones(patcher, **kwargs)


@contextmanager
def encoder_lifetime(clip, keep_model_loaded):
    try:
        yield
    finally:
        if not keep_model_loaded:
            unload_clip(clip.patcher)
            logging.info("Single Encoder Enhancer: encoder unloaded after node execution.")


EDIT_SYSTEM_PROMPT = """Clarify this image-editing request into an actionable directive.
Start with the requested operation. Ground changes in the supplied images. Detail the changed
attributes; preserve everything else with a brief clause instead of redescribing faces or scenery.
When the request constructs a new scene from references, develop the requested subject arrangement,
interaction, setting and visual treatment into a coherent scene rather than merely correcting grammar.
Retain identity and medium unless targeted. Add no unrequested repairs, objects or camera changes.
Refer to the inputs as <image1>, <image2>, etc. in input order, identifying each one's role and what
transfers; with a single input a tag is optional. Preserve requested visible text literally.
Write the directive in English; keep any rendered text exactly in its requested language and script.
Return only the rewritten directive as plain text, without reasoning, JSON, headings or Markdown."""

T2I_SYSTEM_PROMPT = """Expand this image request into coherent English prose describing the
finished picture. Preserve specified subjects, counts, colors, positions and quoted lettering.
Develop unspecified visual details consistently. Identify the medium and style, then describe
the setting, composition, subject placement, surfaces and lighting. For people, describe visible
appearance and clothing. Keep readable text exact in its original script. Write flowing prose,
using present-tense observations rather than commands to a renderer. Do not add invented signage
or empty quality slogans. Return only the description as plain text, without reasoning, JSON,
headings, aspect-ratio metadata or Markdown."""

JSON_EDIT_PROMPT = """Describe the requested image edit as structured fields. Ground details in the supplied
images. Do not invent a new pose, identity, crop or background unless requested. For a new composition
built from references, describe the new scene in target_description and leave preserve empty."""

JSON_T2I_PROMPT = """Describe the requested picture as structured fields. Keep requested subjects, counts,
colors and exact lettering faithful; develop unspecified visual details consistently."""

REQUEST_FIDELITY = """The ORIGINAL_REQUEST defines the task; reference pictures and TASK_CONTEXT
only supply supporting information. Preserve its subjects, counts, actions, relationships and requested
scene. Distinguish composing people/objects together from replacing one with another. If both reference
subjects are requested together, retain both; do not turn that into a face swap, person replacement or
identity merge. Never invent an operation just because two references are supplied. Describe an identity
replacement only when the original request actually asks for it. Preserve requested appearance without
inventing incompatible identities. Exclusions are constraints to respect, not features to add. Lose no
requested subject or action and introduce no unrequested one."""

ENHANCE_OUTPUT_CONTRACT = """Return only the final natural-language rewritten prompt. Match the level of
visual detail asked for by the guidance; do not merely correct the input's grammar.
For a new scene/composition, write descriptive prose that develops subject placement, relationships,
the requested setting, medium, composition and lighting into one coherent picture. Ground details in
the references and request; fill unspecified scene details only when the content guidance permits it.
For a local edit such as changing clothing, describe the targeted change precisely and preserve the
untargeted scene with a brief clause; do not invent a new background, crop or camera setup.
Keep all requested subjects, actions and constraints faithful. Do not return JSON, headings, Markdown
or reasoning."""

LEGACY_LENGTHS = {"same": 1.0, "longer": 2.0, "longest": 4.0}


def length_scale(value):
    # The dial is a multiple of the instruction's word count; older callers pass same/longer/longest.
    scale = float(LEGACY_LENGTHS.get(value, value) if isinstance(value, str) else value)
    if not 1.0 <= scale <= 10.0:
        raise ValueError("Enhance length must be 1-10 (a multiple of the instruction's word count).")
    return scale


def length_guidance(scale):
    if scale <= 1.25:
        return "Keep approximately the original request's length. Improve clarity and visual precision without copying it verbatim or padding it."
    text = (f"Aim for roughly {scale:g} times the original request's length. Develop relevant visual details into "
            "fuller prose while keeping every requested constraint.")
    if scale >= 3:
        text += (" Use comprehensive descriptive prose, in multiple paragraphs when useful. Add only relevant details "
                 "consistent with the request and references, without repetition or invented identities/actions.")
    return text

JSON_EDIT_CONTRACT = """Return exactly one JSON object and nothing else: no Markdown fences, no prose
before or after it. Use this shape (values are examples):
{"edit_instruction": "Replace the jacket of the woman in <image1> with the red coat from <image2>.",
 "reference_roles": {"image1": "the woman and scene to edit", "image2": "source of the red coat"},
 "target_description": "A knee-length red wool coat, buttoned, over her existing outfit.",
 "preserve": ["her face and hair", "pose", "background and lighting"]}
edit_instruction and target_description are required strings; reference_roles maps each supplied image;
preserve is a list of strings and may be empty."""

JSON_T2I_CONTRACT = """Return exactly one JSON object and nothing else: no Markdown fences, no prose
before or after it. Use this shape (values are examples):
{"description": "A quiet harbor at dawn with fishing boats moored along a stone pier.",
 "subjects": ["three wooden fishing boats", "a gull on a mooring post"],
 "background": "low hills under a pale sky", "composition": "wide shot from the end of the pier",
 "lighting": "soft golden light from the left", "style": "natural color photograph",
 "rendered_text": ""}
description is a required string; the other fields are strings or lists of strings and may be empty."""

JSON_REQUIRED = {True: ("edit_instruction", "target_description"), False: ("description",)}

IMAGE_TOKEN = "<image{}>"


@contextmanager
def matched_attention_dtypes():
    # Workaround for ComfyUI's text generation on GPUs without fast bf16 (seen on an RTX 2060 with dynamic VRAM):
    # attention gets a bf16 query beside fp32 keys/values and stops with "Expected query, key, and value to have the
    # same dtype". Only while the rewrite generates, the smaller side is cast to the other's dtype when they differ
    # (per generated token that is the one-token query, not the whole key/value cache) and the result is returned
    # in the query's dtype; matching calls are passed through untouched.
    import comfy.ops
    original = comfy.ops.scaled_dot_product_attention
    warned = []

    def matched(q, k, v, *args, **kwargs):
        if k.dtype == q.dtype and v.dtype == q.dtype:
            return original(q, k, v, *args, **kwargs)
        if not warned:
            logging.warning("Single Encoder Enhancer: attention dtypes differed during generation (query %s, key %s); "
                            "the smaller side is cast to match.", q.dtype, k.dtype)
            warned.append(True)
        target = k.dtype if q.numel() < k.numel() else q.dtype
        mask = kwargs.get("attn_mask")
        if torch.is_tensor(mask) and mask.is_floating_point() and mask.dtype != target:
            kwargs["attn_mask"] = mask.to(target)
        return original(q.to(target), k.to(target), v.to(target), *args, **kwargs).to(q.dtype)

    comfy.ops.scaled_dot_product_attention = matched
    try:
        yield
    finally:
        comfy.ops.scaled_dot_product_attention = original


def json_object(raw, required=()):
    # The first JSON object in the reply that has the required keys; reasoning and Markdown fences are skipped.
    text = raw.partition("</think>")[-1] if "</think>" in raw else raw
    if text.lstrip().startswith("<think>"):
        raise ValueError("unfinished reasoning")
    text = re.sub(r"```(?:json)?", "", text)
    decoder, found, error = json.JSONDecoder(), None, "no JSON object"
    for match in re.finditer(r"\{", text):
        try:
            value = decoder.raw_decode(text[match.start():])[0]
        except json.JSONDecodeError as problem:
            error = f"invalid JSON ({problem.msg} at character {problem.pos})"
            continue
        if not isinstance(value, dict) or not value:
            continue
        if missing := [key for key in required if not isinstance(value.get(key), str) or not value[key].strip()]:
            found, error = found or value, "missing required field(s): " + ", ".join(missing)
            continue
        return value
    raise ValueError(error)


def rewrite_tokens(clip, text, images, thinking):
    # The CLIP's own chat template with its own system turn (the one its encode uses too). Pictures get <imageN>
    # labels only when the template does not already put them in front of each picture (Qwen-Image-2.1 does).
    def rows_of(tokens):
        return next((value for value in tokens.values() if isinstance(value, list)), []) if isinstance(tokens, dict) else tokens

    if images:
        # probe the template alone (the request itself may contain <image1>): does it label the pictures?
        probe = rows_of(clip.tokenize(" ", images=images, skip_template=False, min_length=1, thinking=thinking))
        if not any(isinstance(token[0], dict) for row in probe for token in row if isinstance(token, tuple)):
            logging.warning("Single Encoder Enhancer: this text encoder cannot see images; the rewrite uses the instruction only.")
        ids = [token[0] for row in probe for token in row if isinstance(token, tuple) and isinstance(token[0], int)]
        if IMAGE_TOKEN.format(1) not in clip.decode(ids):
            text = "".join(f"{IMAGE_TOKEN.format(i)}: picture {i}\n" for i in range(1, len(images) + 1)) + text
    tokens = clip.tokenize(text, images=images, skip_template=False, min_length=1, thinking=thinking)
    return tokens, rows_of(tokens)


def generate_text(clip, guidance, user_prompt, images, options, *, original_instruction=None,
                  negative_prompt="", output_contract="", chat=None):
    # One user turn: guidance, request, exclusions and the output format, each once. The model's own system turn stays.
    # With an output contract the guidance (built-in or the user's system prompt) steers content and style only;
    # the format is set once, last, by the contract.
    label = "[GUIDANCE: content and style rules; ignore any output format it asks for]" if output_contract else "[GUIDANCE]"
    parts = [label + "\n" + guidance.strip() + "\n[/GUIDANCE]"]
    if original_instruction is not None:
        parts.append(REQUEST_FIDELITY)
        parts.append("[ORIGINAL_REQUEST]\n" + original_instruction + "\n[/ORIGINAL_REQUEST]")
        if user_prompt != original_instruction:
            parts.append("[TASK_CONTEXT]\n" + user_prompt + "\n[/TASK_CONTEXT]")
        if negative_prompt.strip():
            parts.append("[EXCLUSIONS: avoid these; do not add them]\n" + negative_prompt)
    else:
        parts.append(user_prompt)
    if output_contract:
        parts.append("[REQUIRED_OUTPUT: this format overrides any format the guidance asks for]\n" + output_contract)
    thinking = bool(chat and chat.get("thinking"))
    tokens, rows = rewrite_tokens(clip, "\n\n".join(parts), images, thinking)
    with matched_attention_dtypes():
        text = clip.decode(clip.generate(tokens, **options))
    if thinking and "</think>" not in text and not text.lstrip().startswith("<think>"):
        # a template that opens the reasoning block itself: no close means Max tokens cut the reasoning off
        ids = [token[0] for token in (rows[-1] if rows else []) if isinstance(token, tuple) and isinstance(token[0], int)]
        if ids and clip.decode(ids[-4:]).rstrip().endswith("<think>"):
            text = "<think>" + text
    return text


def json_to_prose(plan):
    # Prose from a structured answer: the rewritten_prompt field, the JSON-mode edit/scene fields, or its text values.
    def words(value):
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return ", ".join(item for item in (words(v) for v in value) if item)
        if isinstance(value, dict):
            return "; ".join(f"{key}: {item}" for key, item in ((k, words(v)) for k, v in value.items()) if item)
        return ""

    def sentence(value):
        value = words(value)
        return value if not value or value[-1] in ".!?" else value + "."

    if words(plan.get("rewritten_prompt")):
        return words(plan["rewritten_prompt"])
    if words(plan.get("edit_instruction")) or words(plan.get("target_description")):
        parts = [sentence(plan.get("edit_instruction")), sentence(plan.get("target_description"))]
        if keep := words(plan.get("preserve")):
            parts.append(sentence("Keep " + keep + " unchanged"))
        return " ".join(part for part in parts if part)
    if words(plan.get("description")):
        parts = [sentence(plan["description"])]
        for key in ("subjects", "background", "composition", "lighting", "style"):
            if value := words(plan.get(key)):
                parts.append(sentence(f"{key.capitalize()}: {value}"))
        if lettering := words(plan.get("rendered_text")):
            parts.append(sentence(f'Visible text reads "{lettering}"'))
        return " ".join(parts)
    return " ".join(sentence(value) for value in plan.values() if words(value))


def rewrite_prompt(raw, image_count):
    text = (raw.partition("</think>")[-1] if "</think>" in raw else raw).strip()
    if text.startswith("<think>") or not text:
        raise ValueError("empty or unfinished rewrite")
    if text.startswith("```"):
        if not text.endswith("```"):
            raise ValueError("unfinished fenced rewrite")
        text = text.partition("\n")[2].rsplit("```", 1)[0].strip()
    # A model that answers with a JSON object (its own transport format or JSON-mode fields) is turned into prose:
    # Enhance never conditions on JSON itself.
    if text.startswith("{"):
        plan = json_object(text)
        text = json_to_prose(plan) if isinstance(plan, dict) else None
    if not isinstance(text, str) or not text.strip() or text.lstrip().startswith(("{", "[", "```", "<think>")):
        raise ValueError("missing rewritten prose")
    text = normalize_tags(text.strip())
    slots = {int(i) for i in re.findall(r"<image_?(\d+)>", text)}
    if unavailable := slots - set(range(1, image_count + 1)):
        tags = ", ".join(IMAGE_TOKEN.format(index) for index in sorted(unavailable))
        raise ValueError(f"references unavailable image tags {tags}; {image_count} reference image(s) supplied")
    return text


def normalize_tags(text):
    return re.sub(r"<image_?(\d+)>", lambda match: IMAGE_TOKEN.format(int(match[1])), text)


# Text-to-image prompts must describe a whole scene, edits only the change: the same dial means twice the words without images.
T2I_LENGTH_FACTOR = 2.0


def effective_scale(scale, has_images):
    return scale if has_images else scale * T2I_LENGTH_FACTOR


TOKENS_PER_WORD = 1.6   # covers punctuation and image tags in typical LLM tokenizers


def length_budget(instruction, scale, max_tokens=None):
    # Word target from the original request; the hard limit leaves 25% slack over it. A smaller token cap
    # (max_length) wins: the budget shrinks so the model is told the length it will really get.
    target = max(20, round(len(instruction.split()) * scale))
    limit = math.ceil(target * 1.25)
    if max_tokens:
        fit = max(1, math.floor((max_tokens - 8) / TOKENS_PER_WORD))
        if limit > fit:
            limit, target = fit, max(1, math.floor(fit / 1.25))
    return target, limit


# sentence-final mark, optionally followed by closing quotes/brackets (straight and curly)
SENTENCE_END = "[.!?…][\"'”’)\\]]*"


def trim_to_words(text, limit):
    # Keep at most `limit` words and end on a complete sentence, also when generation stopped at the token cap.
    words = text.split()
    cut = " ".join(words[:limit]) if len(words) > limit else text.strip()
    if re.search(SENTENCE_END + "$", cut):
        return cut
    ends = [match.end() for match in re.finditer(SENTENCE_END + r"(?=\s)", cut)]
    if ends and ends[-1] > len(cut) // 2:
        return cut[:ends[-1]]
    return cut.rstrip(",;:-— ") + "."


def generate_enhancement(clip, system_prompt, default_system, user_prompt, images, options,
                         *, original_instruction, negative_prompt, enhance_length=1.0, chat=None):
    scale = effective_scale(length_scale(enhance_length), bool(images))
    thinking = bool(chat and chat.get("thinking"))
    target, limit = length_budget(original_instruction, scale, None if thinking else options.get("max_length"))
    output_contract = (ENHANCE_OUTPUT_CONTRACT + "\n\n[ENHANCE_LENGTH]\n" + length_guidance(scale)
                       + f" Write about {target} words and never more than {limit} words; "
                       "put every requested change and preserved attribute inside that budget.")
    # generation stops at the word limit's token equivalent at the latest
    cap = math.ceil(limit * TOKENS_PER_WORD) + 8
    if not thinking:   # reasoning comes first and needs room; the answer is still trimmed to the word limit
        options = {**options, "max_length": min(options.get("max_length", cap), cap)}
    tags = ", ".join(IMAGE_TOKEN.format(index) for index in range(1, len(images) + 1))
    output_contract += (f"\n\n[IMAGE_REFERENCES]\nThere are {len(images)} reference images. "
                        + (f"Available tags, in input order: {tags}. " if tags else "Do not use image tags. ")
                        + "With several images, identify each source by its tag; with one image a tag is optional. "
                        "Never reference an image outside this list. Preserve all subjects and relationships requested by the user.")
    guidance = system_prompt or default_system
    raw = generate_text(clip, guidance, user_prompt, images, options,
                        original_instruction=original_instruction, negative_prompt=negative_prompt,
                        output_contract=output_contract, chat=chat)
    for attempt in range(2):
        try:
            prompt = rewrite_prompt(raw, len(images))
            for source in (original_instruction, user_prompt):
                if " ".join(prompt.split()) == " ".join(normalize_tags(source).split()):
                    raise ValueError("the rewrite only echoed the input or its preservation context")
            # trim after the echo check: an added final period must not disguise an echoed input
            return raw, trim_to_words(prompt, limit)
        except ValueError as error:
            if attempt:
                raise ValueError(f"Enhance could not produce rewritten prose after one retry: {error}. "
                                 "Check the system prompt, image inputs and max_length. No original-prompt fallback was encoded.") from error
            logging.warning("Single Encoder Enhancer: %s. Retrying once on the same CLIP with prose guidance.", error)
            repair_request = user_prompt + "\n\nRewrite the original request as prose; do not copy it verbatim."
            raw = generate_text(clip, guidance, repair_request, images, options,
                                original_instruction=original_instruction, negative_prompt=negative_prompt,
                                output_contract=output_contract, chat=chat)


@contextmanager
def bf16_encode(clip, enabled):
    # ComfyUI runs many LLM conditioning passes with fp32 activations (sd1_clip passes dtype=float32); bf16 is faster.
    if not enabled:
        yield
        return
    if not model_management.should_use_bf16(clip.patcher.load_device):
        logging.warning("Single Encoder Enhancer: this device has no fast bf16; Fast encode is ignored.")
        yield
        return
    name = getattr(clip.cond_stage_model, "clip", None)   # single-model CLIPs only (not dual/multi-encoder CLIPs)
    transformer = getattr(getattr(clip.cond_stage_model, name, None), "transformer", None) if isinstance(name, str) else None
    if transformer is None:
        logging.warning("Single Encoder Enhancer: Fast encode supports single-model text encoders only; ignored for this CLIP.")
        yield
        return
    previous = transformer.__dict__.get("forward")
    forward = transformer.forward

    def bf16_forward(*args, **kwargs):
        if kwargs.get("dtype") == torch.float32:
            kwargs["dtype"] = torch.bfloat16
            if torch.is_tensor(kwargs.get("embeds")):
                kwargs["embeds"] = kwargs["embeds"].to(torch.bfloat16)
        return forward(*args, **kwargs)
    transformer.forward = bf16_forward
    try:
        yield
    finally:
        if previous is None:
            del transformer.forward
        else:
            transformer.forward = previous


def reference_images(references, resolution):
    # Each picture resized to about resolution^2 pixels (aspect kept, multiples of 32). The vision tower gets it with
    # alpha composited over white; the VAE gets all channels (it keeps what it can take, e.g. RGBA VAEs keep alpha).
    vision, full, latent_size = [], [], None
    for image in references.values():
        samples = image[:1].movedim(-1, 1)
        if resolution > 0:
            ratio = samples.shape[3] / samples.shape[2]
            width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
            height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
        else:
            width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
        width, height = max(32, width), max(32, height)
        if (width, height) == (samples.shape[3], samples.shape[2]):
            s = image[:1]
        else:
            s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
        latent_size = latent_size or (width, height)
        rgb = s[:, :, :, :3]
        if s.shape[-1] > 3:
            rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])
        vision.append(rgb)
        full.append(s)
    return vision, full, latent_size


def ordered_references(images):
    # one picture per entry, in socket order; batches are split
    references = {}
    for name in sorted(images or {}, key=lambda n: int(n.rsplit("_", 1)[-1])):
        image = images[name]
        if image is not None:
            for j in range(image.shape[0]):
                references[f"image_{len(references) + 1}"] = image[j:j + 1]
    return references


def cannot_rewrite(clip):
    # None when this CLIP can write text; otherwise the reason it cannot.
    model = clip.cond_stage_model
    name = getattr(model, "clip", None)
    if not isinstance(name, str):   # no single inner encoder: trust the model's own generate
        return None if callable(getattr(model, "generate", None)) else "it has no text generation"
    transformer = getattr(getattr(model, name, None), "transformer", None)
    if not callable(getattr(transformer, "generate", None)):
        return "it has no text generation"
    config = getattr(getattr(transformer, "model", None), "config", None)
    if getattr(config, "final_norm", True) is False:
        return "it is a truncated encoder (no final layers / output head)"
    return None


def pick_writer(clip, refiner_clip, choice="auto"):
    # auto/main: the main CLIP writes when it can (no second model load), otherwise the refiner; refiner: the refiner
    # writes when it can, otherwise the main CLIP. None = nobody can, encode as written.
    reasons = []
    order = [("main CLIP", clip), ("refiner CLIP", refiner_clip)]
    for label, candidate in (reversed(order) if choice == "refiner" else order):
        if candidate is None:
            continue
        reason = cannot_rewrite(candidate)
        if reason is None:
            return candidate, reasons
        reasons.append(f"{label}: {reason}")
    return None, reasons


def plan(clip, instruction, vision, mode, system_prompt, negative_prompt, options, enhance_length, chat):
    # -> raw output, prepared prompt, whether the result may be stored for reuse
    request = normalize_tags(instruction)
    if mode == "enhance":
        return (*generate_enhancement(clip, system_prompt.strip(), EDIT_SYSTEM_PROMPT if vision else T2I_SYSTEM_PROMPT,
                                      request, vision, options, original_instruction=request, negative_prompt=negative_prompt,
                                      enhance_length=enhance_length, chat=chat), True)
    edit = bool(vision)
    # A connected system prompt is kept as the content guide, next to the built-in field guidance; the JSON shape
    # still comes from the contract and is checked below, so a prose-asking prompt cannot change the output format.
    built_in = JSON_EDIT_PROMPT if edit else JSON_T2I_PROMPT
    guidance = (system_prompt.strip() + "\n\n" + built_in) if system_prompt.strip() else built_in
    contract = JSON_EDIT_CONTRACT if edit else JSON_T2I_CONTRACT
    task = request
    for attempt in range(2):
        raw = generate_text(clip, guidance, task, vision, options, original_instruction=request,
                            negative_prompt=negative_prompt, output_contract=contract, chat=chat)
        try:
            caption = json_object(raw, JSON_REQUIRED[edit])
            prompt = json.dumps(caption, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            return raw, normalize_tags(prompt), True
        except ValueError as error:
            if attempt:
                raise ValueError(f"JSON mode could not produce a valid JSON object after one retry: {error}. "
                                 "See raw_output; raise Max tokens if the object was cut off. Nothing was encoded.") from error
            logging.warning("Single Encoder Enhancer JSON: %s. Retrying once with the error.", error)
            task = (request + f"\n\nYour previous answer was not usable ({error}). Answer again with exactly one "
                    "complete JSON object in the required shape and nothing else.")


def has_pictures(tokens):
    rows = next((v for v in tokens.values() if isinstance(v, list)), []) if isinstance(tokens, dict) else tokens
    return any(isinstance(t, tuple) and isinstance(t[0], dict) and t[0].get("type") == "image" for row in rows for t in row)


def tokenize_for_encode(clip, text, vision, keep_vision):
    # Edit encoders take the pictures in their own token sequence (as their native encode node does); every other CLIP
    # ignores them, and then the plain text tokens are used, as CLIPTextEncode would.
    if vision:
        tokens = clip.tokenize(text, images=vision, keep_vision=keep_vision, prevent_empty_text=True)
        if has_pictures(tokens):
            return tokens, True
    return clip.tokenize(text), False


def encode_plain(clip, prompt, vision, latent_size, vae, resolution, negative_prompt, encode_negative, vae_pixels=None):
    # Any CLIP: text encode (pictures inside it when the encoder takes them) + reference latents for edit models.
    ref_latents = []
    if vision and vae is not None:
        # all channels, as the native edit encode does; ComfyUI's VAE drops channels it does not take
        ref_latents = [vae.encode(pixels) for pixels in (vae_pixels or vision)]
    # with reference latents the vision tokens are replaced by them in the image model (keep_vision off), as natively
    tokens, image_aware = tokenize_for_encode(clip, prompt, vision, keep_vision=not ref_latents)
    positive = clip.encode_from_tokens_scheduled(tokens)
    if vision and not image_aware:
        logging.info("Single Encoder Enhancer: this text encoder does not take pictures; the encode is text only.")
    # an encoder that ignored the pictures for the positive is not asked about them again
    negative = (clip.encode_from_tokens_scheduled(
                    tokenize_for_encode(clip, negative_prompt, vision if image_aware else None, not ref_latents)[0])
                if encode_negative else nodes.ConditioningZeroOut().zero_out(positive)[0])
    if ref_latents:
        # on both conds like the native edit nodes, so CFG does not treat the reference as a positive-only push
        positive, negative = (node_helpers.conditioning_set_values(c, {"reference_latents": ref_latents}, append=True)
                              for c in (positive, negative))
    elif vision:
        logging.warning("Single Encoder Enhancer: no VAE connected; images only guide the rewrite, no reference latents are added.")
    width, height = latent_size or (resolution or 1024, resolution or 1024)
    if vae is None:
        return positive, negative, nodes.EmptyLatentImage().generate(width, height, 1)[0]
    # empty latent in the connected VAE's own layout (channels, compression, dims) at image 1's size
    shape = ref_latents[0].shape if ref_latents else blank_latent_shape(vae, width, height)
    return positive, negative, {"samples": torch.zeros(shape, device=model_management.intermediate_device())}


class SingleEncoderEnhancer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SingleEncoderEnhancer",
            display_name="Single Encoder Enhancer",
            search_aliases=["single encoder", "prompt enhance", "prompt rewrite", "enhance and encode", "TextGenerate encode", "LLM text encode"],
            category="conditioning/single_encoder",
            description="One CLIP rewrites the instruction and then encodes the result: no second model. Normal encodes the instruction as written; Enhance rewrites it as prose; JSON writes a structured JSON prompt and encodes that JSON. The rewrite uses the CLIP's own chat template, like Generate Text; the encode passes the images into the encoder's own tokens when it takes pictures (as edit models' native encode nodes do), otherwise it is a plain text encode; with a VAE the images are also added as reference latents. Enhance/JSON need a CLIP that works with Generate Text; if the main CLIP cannot write (truncated or non-LLM encoder) connect a refiner CLIP to write for it, otherwise the instruction is encoded as written. Normal works with any CLIP.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("instruction", display_name="Instruction", multiline=True),
                io.Combo.Input("mode", display_name="Mode", options=["enhance", "normal", "json"], default="enhance",
                               tooltip="Normal: encode the instruction as written (no text generation). Enhance: the CLIP rewrites the instruction as prose, which is encoded. JSON: the CLIP writes a structured JSON prompt (text-to-image: description, subjects, background, composition, lighting, style, rendered_text; edits: edit_instruction, reference_roles, target_description, preserve) and the JSON text itself is encoded."),
                io.Combo.Input("writer", display_name="Writer", options=["auto", "main", "refiner"], default="auto",
                               tooltip="Active while a refiner CLIP is connected (greyed out otherwise): which model writes the Enhance/JSON rewrite. Auto: the main CLIP when it can write (no second model load), the refiner only when it cannot (truncated or non-LLM encoders). Main: same as auto. Refiner: the refiner writes, e.g. a different or larger writer. If the chosen model cannot write, the other one does; the main CLIP always encodes."),
                io.Int.Input("resolution", display_name="Resolution", default=1024, min=0, max=4096, step=32,
                             tooltip="Output size: images are resized to about resolution x resolution pixels (aspect kept) and the latent follows image 1; without images the latent is resolution x resolution. 0 = images at their own size. The rewrite sees the same resized images the encode uses."),
                io.Autogrow.Input("images", template=io.Autogrow.TemplateNames(
                    io.Image.Input("image"), names=[f"image_{i}" for i in range(1, 17)], min=0),
                    tooltip="Optional reference images, one per socket (image 1, image 2, ...); a new socket appears when you connect the last one. Use <image1>, <image2> in the instruction to refer to them. Batches are split into single images. With images connected the node treats the job as an edit, otherwise as text-to-image. Edit models need the VAE for reference latents."),
                io.Vae.Input("vae", optional=True, tooltip="Needed for edits: encodes the images into reference latents. Without it the images only guide the rewrite."),
                io.Clip.Input("refiner_clip", optional=True,
                              tooltip="Optional second text model that writes the Enhance/JSON rewrite; the main CLIP then encodes it. Use it when the main CLIP cannot write text (truncated or non-LLM encoders) or when you want a different writer. With Writer = auto it writes only when the main CLIP cannot; set Writer = refiner to make it write. Not connected: the main CLIP writes and encodes. If no connected CLIP can write, the instruction is encoded as written. Costs a second model load; it is unloaded before the encode unless Keep model loaded is on."),
                io.String.Input("negative_prompt", display_name="Negative prompt", multiline=True, default="", advanced=True,
                                tooltip="Things to avoid. In Enhance/JSON it is passed to the rewrite as exclusions, and it is encoded as the negative conditioning (unless Encode negative is off). Has no effect on the image at CFG 1."),
                io.Float.Input("enhance_length", display_name="Prompt length x", default=1.0, min=1.0, max=10.0, step=0.5,
                               display_mode=io.NumberDisplay.slider,
                               tooltip="Enhance only. Target length of the rewrite as a multiple of your instruction's word count (at least 20 words). With images: 1 = about the same length, up to 10x. Without images (text-to-image) the multiple is doubled: 1 = 2x, up to 20x. Hard limit: +25% over the target, cut back to the last full sentence. A low Max tokens shrinks it."),
                io.Int.Input("max_length", display_name="Max tokens", default=4096, min=1, max=32768,
                             tooltip="Maximum tokens the CLIP may generate in Enhance and JSON. In Enhance it also shrinks the word budget when it is lower than the length dial needs (about 1.6 tokens per word). With thinking on it must cover reasoning + answer. JSON output cut off by this limit is retried once, then the node stops with an error (nothing is encoded). Not used in Normal."),
                io.Boolean.Input("do_sample", display_name="Sampling", default=False,
                                 tooltip="Off: the rewrite is deterministic (always the most likely token), so the same inputs give the same prompt. On: random sampling with the seed, temperature and the top k / top p / min p / penalty settings below, which are active only while Sampling is on (greyed out otherwise). Only affects the text rewrite, not the image sampler."),
                io.Int.Input("seed", display_name="Seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True,
                             tooltip="Seed for the rewrite when Sampling is on. Separate from the image sampler's seed."),
                io.Float.Input("temperature", display_name="Temperature", default=0.2, min=0.01, max=2.0, step=0.05,
                               tooltip="Randomness of the rewrite when Sampling is on: lower = closer to the most likely wording."),
                io.Int.Input("top_k", display_name="Top k", default=64, min=0, max=1000,
                             tooltip="Sampling on: pick only from the k most likely next tokens (0 = no limit). Same as Generate Text."),
                io.Float.Input("top_p", display_name="Top p", default=0.95, min=0.0, max=1.0, step=0.01,
                               tooltip="Sampling on: pick only from the most likely tokens that together reach this probability. Same as Generate Text."),
                io.Float.Input("min_p", display_name="Min p", default=0.05, min=0.0, max=1.0, step=0.01,
                               tooltip="Sampling on: drop tokens less likely than this fraction of the top token's probability. Same as Generate Text."),
                io.Float.Input("repetition_penalty", display_name="Repetition penalty", default=1.05, min=0.0, max=5.0, step=0.01,
                               tooltip="Applies with Sampling on or off: values above 1 make already-used tokens less likely, reducing repeated words. 1 = off. Same as Generate Text."),
                io.Float.Input("presence_penalty", display_name="Presence penalty", default=0.0, min=0.0, max=5.0, step=0.01,
                               tooltip="Applies with Sampling on or off: subtracts this from every already-used token, pushing toward new words. 0 = off. Same as Generate Text."),
                io.Boolean.Input("thinking", display_name="Thinking", default=False,
                                 tooltip="Ask the model to reason before it writes, like Generate Text's thinking option. Works only on models whose chat template supports it; others ignore it. Much slower. The reasoning appears in raw_output and is not encoded; set Max tokens high enough for reasoning + answer."),
                io.String.Input("system_prompt", display_name="System prompt", optional=True, force_input=True,
                                tooltip="Optional: replaces the built-in rewrite guidance (Enhance/JSON) with your own. It is given to the model as guidance in the request; the model's own system prompt is kept. The node still adds its own output-format rules for the selected mode, so guidance asking for another format cannot break the encode. Empty = built-in guidance."),
                io.Boolean.Input("force_rewrite", display_name="Force rewrite", default=False,
                                 tooltip="Off: an unchanged request (same writer, texts, pictures and settings) reuses its stored rewrite, so nothing is generated again. On: do all the work again on every run, also when nothing changed: a fresh rewrite, a fresh encode and fresh image features (nothing is reused from earlier runs). Use it for timing tests. With Sampling off the fresh rewrite is the same text."),
                io.Combo.Input("encode_negative", display_name="Encode negative", options=["on", "auto", "off"], default="auto",
                               tooltip="On: encode the negative prompt normally. Off: skip that encode pass and output a zeroed negative. Auto: Off while the negative prompt is empty, On when it has text. A zeroed negative is only the same image at CFG 1, where the sampler never reads the negative; above CFG 1 an encoded empty negative and a zeroed one give different images."),
                io.Boolean.Input("fast_encode", display_name="Fast encode (bf16)", default=False,
                                 tooltip="Single-model text encoders that ComfyUI runs with fp32 activations: encode in bf16 instead. Faster; the conditioning differs slightly from the default. Off = exactly ComfyUI's default encode."),
                io.Boolean.Input("keep_model_loaded", display_name="Keep model loaded", default=False,
                                 tooltip="On: leave the CLIP loaded for the next run (ComfyUI may still offload it when VRAM is needed). Off: unload it from the GPU after this node, also on errors, to free VRAM for the diffusion model."),
            ],
            outputs=[
                io.String.Output(display_name="prepared_prompt"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.String.Output(display_name="raw_output"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def validate_inputs(cls, encode_negative="auto"):
        # workflows saved while this was an on/off switch still send true/false
        return True if encode_negative in (True, False, "on", "auto", "off") else "Choose Encode negative = on, auto or off."

    @classmethod
    def fingerprint_inputs(cls, mode="enhance", force_rewrite=False, **_inputs):
        # Force rewrite: run on every queue, also when ComfyUI sees no changed input.
        return time.time() if force_rewrite and mode != "normal" else ""

    @classmethod
    def execute(cls, clip, instruction, images=None, vae=None, refiner_clip=None, writer="auto", max_length=4096, do_sample=False, seed=0,
                negative_prompt="", system_prompt="", mode="enhance", temperature=0.2, resolution=1024,
                enhance_length=1.0, keep_model_loaded=False, encode_negative="auto", fast_encode=False, thinking=False,
                top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0, force_rewrite=False,
                **_retired):
        if mode not in ("enhance", "json", "normal"):
            raise ValueError("Choose mode=normal, enhance or json.")
        if isinstance(encode_negative, str):
            if encode_negative not in ("on", "auto", "off"):
                raise ValueError("Choose Encode negative = on, auto or off.")
            encode_negative = encode_negative == "on" or (encode_negative == "auto" and bool(negative_prompt.strip()))
        references = ordered_references(images)
        reuse = not force_rewrite   # Force rewrite: nothing from earlier runs is used
        if reuse and isinstance(vae, comfy.sd.VAE):   # reference latents of an unchanged picture are reused across runs
            vae = CachedVAE(vae)
        if writer not in ("auto", "main", "refiner"):
            raise ValueError("Choose writer=auto, main or refiner.")
        author, notice = None, ""
        if mode != "normal":
            author, reasons = pick_writer(clip, refiner_clip, writer)
            for reason in reasons:
                logging.warning("Single Encoder Enhancer: cannot rewrite with the %s.", reason)
            if author is None:
                notice = (f"[Single Encoder Enhancer] {mode.capitalize()} was not applied: no connected CLIP can write text "
                          f"({'; '.join(reasons) or 'no writer'}). The instruction was encoded as written. Connect a "
                          "refiner CLIP that can generate text (e.g. Qwen3-VL) or use Normal.")
                logging.warning("Single Encoder Enhancer: no connected CLIP can rewrite; the instruction is encoded as written (Normal).")
                mode = "normal"
        # resized once: the rewrite, the reference latents and the latent size all use these pictures
        vision, vae_pixels, latent_size = reference_images(references, resolution)

        with ExitStack() as stack:
            stack.enter_context(hash_memo())   # each picture is hashed once per run for all the caches
            stack.enter_context(encoder_lifetime(clip, keep_model_loaded))
            main_writes = author is clip
            if vision:   # the rewrite, the positive and the negative encode share one vision-tower pass per picture
                clip = stack.enter_context(vision_feature_cache(clip, across_runs=reuse))
            raw, prompt, keep = notice, normalize_tags(instruction), True
            if mode != "normal":
                # like Generate Text: with sampling off ComfyUI takes the most likely token and ignores these
                options = dict(do_sample=do_sample, max_length=max_length, temperature=temperature, top_k=top_k,
                               top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty,
                               presence_penalty=presence_penalty, seed=seed)
                if mode == "json":   # JSON repeats quotes, colons and braces by design: penalties would break its syntax
                    options.update(repetition_penalty=1.0, presence_penalty=0.0)
                # an unchanged request is not written again: same writer, texts, pictures and settings -> stored rewrite
                key = rewrite_key(author, mode, instruction, system_prompt, negative_prompt,
                                  enhance_length if mode == "enhance" else 0.0, thinking, options, vision)
                owner = author.cond_stage_model
                found = REWRITE_STORE.lookup(key, owner) if reuse else None
                if found is not None:
                    raw, prompt = found
                    logging.info("Single Encoder Enhancer: rewrite reused from an earlier run (nothing it depends on changed).")
                # each picture passes the writer's vision tower once: a retry and later runs reuse the features
                elif main_writes:
                    raw, prompt, keep = plan(clip, instruction, vision, mode, system_prompt, negative_prompt, options,
                                             enhance_length, {"thinking": thinking})
                else:   # a separate refiner leaves the GPU before the main CLIP encodes (unless models are kept loaded)
                    with encoder_lifetime(author, keep_model_loaded), \
                            (vision_feature_cache(author, across_runs=reuse) if vision else nullcontext(author)) as cached_author:
                        raw, prompt, keep = plan(cached_author, instruction, vision, mode, system_prompt, negative_prompt,
                                                 options, enhance_length, {"thinking": thinking})
                if found is None and keep:
                    REWRITE_STORE.store(key, owner, (raw, prompt))
            if reuse:
                stack.enter_context(text_encode_cache(clip, fast_encode))
            with bf16_encode(clip, fast_encode):
                positive, negative, latent = encode_plain(clip, prompt, vision, latent_size, vae, resolution,
                                                          negative_prompt, encode_negative, vae_pixels)
            return io.NodeOutput(prompt, positive, negative, raw, latent)
