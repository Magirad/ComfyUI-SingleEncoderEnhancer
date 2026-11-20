"""Mock check of the three modes on CPU. Run with ComfyUI's Python."""

import importlib
from pathlib import Path
import sys

import torch

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package.parents[1]))   # ComfyUI root
sys.path.insert(0, str(package.parent))       # custom_nodes
node_module = importlib.import_module(package.name + ".nodes")
Node = node_module.SingleEncoderEnhancer


class Patcher:
    patches_uuid = "p"
    forced_hooks = None
    load_device = torch.device("cpu")


class Inner:
    class transformer:
        @staticmethod
        def generate(*a, **k):
            return None


class Model:
    clip = "llm"
    llm = Inner()


class Truncated:
    class transformer(Inner.transformer):
        class model:
            class config:
                final_norm = False


class TruncatedModel:
    clip = "llm"
    llm = Truncated()


class Clip:
    def __init__(self, reply, model=Model, sees_images=True):
        self.reply, self.calls, self.encoded, self.sees_images = reply, [], [], sees_images
        self.patcher, self.cond_stage_model = Patcher(), model()
        self.use_clip_schedule, self.layer_idx = False, None

    def tokenize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        pictures = [({"type": "image", "data": image}, 1.0) for image in kwargs.get("images", [])] if self.sees_images else []
        return {"t": [pictures + [(text, 1.0)]]}

    def generate(self, tokens, **options):
        self.generated = getattr(self, "generated", 0) + 1
        return tokens

    def decode(self, tokens):
        return self.reply

    def encode_from_tokens_scheduled(self, tokens, **kwargs):
        self.encoded.append(tokens["t"][0][-1][0])
        return [[torch.ones(1, 2, 4), {}]]


common = dict(keep_model_loaded=True, resolution=64, encode_negative="on")
clip = Clip("unused")
prompt, pos, neg, raw, latent = Node.execute(clip, "a red cube", mode="normal", **common).args
assert prompt == "a red cube" and raw == "" and clip.encoded == ["a red cube", ""] and latent["samples"].shape == (1, 4, 8, 8)

clip = Clip("A glossy red cube rests on a pale wooden table under soft morning light from a window, casting a gentle shadow.")
prompt, *_ = Node.execute(clip, "a red cube", mode="enhance", **common).args
assert prompt.startswith("A glossy red cube") and clip.encoded[0] == prompt
kwargs = clip.calls[0][1]
assert "[GUIDANCE" in clip.calls[0][0] and "system_prompt" not in kwargs and kwargs["skip_template"] is False and kwargs["images"] == [] and kwargs["thinking"] is False

clip = Clip('{"description": "red cube", "style": "photo"}')
prompt, *_ = Node.execute(clip, "a red cube", mode="json", **common).args
assert prompt == '{"description":"red cube","style":"photo"}'

image = torch.rand(1, 48, 32, 3)
clip = Clip("Make <image_1> blue while keeping the table, lighting and framing of the original picture unchanged.")
prompt, *_ = Node.execute(clip, "make <image1> blue", mode="enhance", images={"image_1": image}, **common).args
assert "<image1>" in prompt and len(clip.calls[0][1]["images"]) == 1
# the encode gives the pictures to an encoder that takes them (positive and negative), like the native edit encode
encodes = [kwargs for text, kwargs in clip.calls if "keep_vision" in kwargs]
assert len(encodes) == 2 and all(len(k["images"]) == 1 and k["keep_vision"] is True for k in encodes)
# Normal mode with a picture: same image-aware encode, no generation
clip = Clip("unused")
Node.execute(clip, "make <image1> blue", mode="normal", images={"image_1": image}, **common)
assert not getattr(clip, "generated", 0) and all(len(k.get("images", [])) == 1 for _, k in clip.calls)
# an encoder that ignores pictures falls back to a plain text encode
clip = Clip("unused", sees_images=False)
Node.execute(clip, "make <image1> blue", mode="normal", images={"image_1": image}, **common)
assert [k for _, k in clip.calls if "images" not in k] and clip.encoded[0] == "make <image1> blue"

# two pictures of different shapes go to the writer as a list, each at its own aspect
wide = torch.rand(1, 32, 64, 3)
clip = Clip("Place the subject of <image1> into the scene of <image2>, keeping its lighting and the original framing intact.")
Node.execute(clip, "put <image1> in <image2>", mode="enhance", images={"image_1": image, "image_2": wide}, **common)
shapes = [tuple(i.shape[1:3]) for i in clip.calls[0][1]["images"]]
assert len(shapes) == 2 and shapes[0] != shapes[1]

# truncated main CLIP, no refiner: Enhance falls back to Normal, nothing is generated
reply = "A glossy red cube rests on a pale wooden table under soft morning light from a window, casting a gentle shadow."
main = Clip(reply, model=TruncatedModel)
prompt, pos, neg, raw, latent = Node.execute(main, "a red cube", mode="enhance", **common).args
assert prompt == "a red cube" and raw.startswith("[Single Encoder Enhancer] Enhance was not applied") and len(main.calls) == 2 and main.encoded[0] == "a red cube"

# truncated main CLIP + refiner: the refiner writes, the main CLIP encodes the rewrite
main, refiner = Clip("unused", model=TruncatedModel), Clip(reply)
prompt, *_ = Node.execute(main, "a red cube", mode="enhance", refiner_clip=refiner, **common).args
assert prompt == reply and main.encoded[0] == reply and not refiner.encoded and "[GUIDANCE" in refiner.calls[0][0]

# able main CLIP + refiner, Writer auto: the main CLIP writes and the refiner is never loaded
main, refiner = Clip(reply), Clip("unused")
prompt, *_ = Node.execute(main, "a red cube", mode="enhance", refiner_clip=refiner, **common).args
assert prompt == reply and not refiner.calls and main.encoded[0] == reply

# writer choice: main keeps the refiner idle; refiner/auto use it; a choice that cannot write falls to the other
main, refiner = Clip(reply), Clip("unused")
prompt, *_ = Node.execute(main, "a red cube", mode="enhance", refiner_clip=refiner, writer="main", **common).args
assert prompt == reply and not refiner.calls and "[GUIDANCE" in main.calls[0][0]
main, refiner = Clip("unused"), Clip(reply)
prompt, *_ = Node.execute(main, "a red cube", mode="enhance", refiner_clip=refiner, writer="refiner", **common).args
assert prompt == reply and len(main.calls) == 2
main, refiner = Clip("unused", model=TruncatedModel), Clip(reply)
prompt, *_ = Node.execute(main, "a red cube", mode="enhance", refiner_clip=refiner, writer="main", **common).args
assert prompt == reply and main.encoded[0] == reply

# rewrite cache: an unchanged request is not written again; with sampling off a new seed still reuses
main = Clip(reply)
first = Node.execute(main, "a red cube", mode="enhance", images={"image_1": image}, seed=1, **common).args
again = Node.execute(main, "a red cube", mode="enhance", images={"image_1": image.clone()}, seed=2, **common).args
assert main.generated == 1 and again[0] == first[0] and again[3] == first[3]
for change in (dict(instruction="a blue cube"), dict(images={"image_1": image * 0.5}), dict(enhance_length=2.0),
               dict(negative_prompt="blur"), dict(max_length=512), dict(do_sample=True, seed=1), dict(do_sample=True, seed=2)):
    count = main.generated
    Node.execute(main, **{"instruction": "a red cube", "mode": "enhance", "images": {"image_1": image}, "seed": 1, **common, **change})
    assert main.generated == count + 1, change
# Force rewrite: everything is done again (rewrite and encode), and ComfyUI is told the node changed
main = Clip(reply)
for _ in range(2):
    Node.execute(main, "a red cube", mode="enhance", force_rewrite=True, **common)
assert main.generated == 2 and len(main.encoded) == 4       # positive + negative, both runs
Node.execute(main, "a red cube", mode="enhance", **common)
assert main.generated == 2                                   # the forced rewrite was still stored for later reuse
assert Node.fingerprint_inputs(mode="enhance", force_rewrite=False) == Node.fingerprint_inputs(mode="enhance", force_rewrite=False)
assert isinstance(Node.fingerprint_inputs(mode="enhance", force_rewrite=True), float)
assert Node.fingerprint_inputs(mode="normal", force_rewrite=True) == ""

# a refiner that already wrote this request is not loaded again
main, refiner = Clip("unused", model=TruncatedModel), Clip(reply)
for _ in range(2):
    prompt, *_ = Node.execute(main, "a red cube", mode="enhance", refiner_clip=refiner, **common).args
assert refiner.generated == 1 and prompt == reply

# vision features: a real ComfyUI CLIP shell around a counting vision tower
import comfy.model_patcher
import comfy.sd
caches = importlib.import_module(package.name + ".caches")


class Tower(torch.nn.Module):
    calls = 0

    def preprocess_embed(self, embed, device):
        Tower.calls += 1
        return embed["data"].mean(dim=(1, 2)), {"grid": torch.tensor([[1, 2, 2]])}


def vision_clip():
    shell = comfy.sd.CLIP(no_init=True)
    shell.cond_stage_model = torch.nn.Module()
    shell.cond_stage_model.clip = "llm"
    inner = torch.nn.Module()
    inner.transformer = Tower()
    shell.cond_stage_model.add_module("llm", inner)
    shell.patcher = comfy.model_patcher.ModelPatcher(shell.cond_stage_model, torch.device("cpu"), torch.device("cpu"))
    shell.tokenizer, shell.layer_idx, shell.tokenizer_options = None, None, {}
    shell.use_clip_schedule, shell.apply_hooks_to_conds = False, None
    return shell


def look(shell, picture, **options):
    with caches.vision_feature_cache(shell) as cached:
        assert cached is not shell and cached.cond_stage_model is shell.cond_stage_model
        see = cached.patcher.get_model_object("llm.transformer.preprocess_embed")
        first = see({"type": "image", "data": picture, **options}, torch.device("cpu"))
        again = see({"type": "image", "data": picture.clone(), **options}, torch.device("cpu"))   # e.g. the Enhance retry
        assert torch.equal(first[0], again[0])
        return first


shell, picture = vision_clip(), torch.rand(1, 8, 8, 3)
expected = picture.mean(dim=(1, 2))
assert torch.equal(look(shell, picture)[0], expected) and Tower.calls == 1      # twice in one run: one tower pass
assert torch.equal(look(shell, picture)[0], expected) and Tower.calls == 1      # next run, same picture: none
look(shell, picture, max_soft_tokens=70)
assert Tower.calls == 2                                                         # other per-picture options: recomputed
look(shell, picture + 0.1)
assert Tower.calls == 3                                                         # other picture: recomputed
look(vision_clip(), picture)
assert Tower.calls == 4                                                         # other weights: recomputed

# reasoning cut off by Max tokens on a template that opens the think block itself is not encoded as the prompt
class OpensThink(Clip):
    def tokenize(self, text, **kwargs):
        out = super().tokenize(text, **kwargs)
        if kwargs.get("thinking"):
            out["t"][0] += [(1, 1.0), (2, 1.0)]
        return out

    def decode(self, tokens):
        return "<think>\n" if isinstance(tokens, list) else self.reply

clip = OpensThink("Let me consider what the user wants here, the cube should be")
try:
    Node.execute(clip, "a red cube", mode="enhance", thinking=True, **common)
    raise SystemExit("FAIL: cut-off reasoning was encoded")
except ValueError:
    pass

# JSON mode: fenced object with chatter is parsed; wrong keys -> one retry -> clear error; penalties off for JSON
class Replies(Clip):
    def __init__(self, replies):
        super().__init__(replies[0])
        self.replies, self.options = list(replies), []

    def generate(self, tokens, **options):
        self.options.append(options)
        return super().generate(tokens, **options)

    def decode(self, tokens):
        return self.replies[min(getattr(self, "generated", 0), len(self.replies)) - 1] if getattr(self, "generated", 0) else ""


clip = Replies(['Sure! ```json\n{"description": "a red cube", "style": "photo"}\n``` hope it helps'])
prompt, *_ = Node.execute(clip, "a red cube", mode="json", repetition_penalty=1.3, presence_penalty=0.5, **common).args
assert prompt == '{"description":"a red cube","style":"photo"}' and clip.generated == 1
assert clip.options[0]["repetition_penalty"] == 1.0 and clip.options[0]["presence_penalty"] == 0.0
clip = Replies(['{"rewritten_prompt": "x"}', '{"description": "a blue cube"}'])
prompt, *_ = Node.execute(clip, "a blue cube", mode="json", **common).args
assert prompt == '{"description":"a blue cube"}' and clip.generated == 2 and "not usable" in clip.calls[-3][0]
clip = Replies(['no json here', '{"subjects": ["cube"]}'])
try:
    Node.execute(clip, "a green cube", mode="json", **common)
    raise SystemExit("FAIL: invalid JSON was encoded")
except ValueError as error:
    assert "after one retry" in str(error) and not clip.encoded
clip = Replies(['{"edit_instruction": "make it blue", "target_description": "a blue cube", "preserve": []}'])
prompt, *_ = Node.execute(clip, "make <image1> blue", mode="json", images={"image_1": image}, **common).args
assert '"edit_instruction":"make it blue"' in prompt
# the template's own <imageN> labels are not repeated; a template without them gets the labels once
clip = Replies(['{"edit_instruction": "make it red", "target_description": "a red cube"}'])
clip.decode = lambda tokens: "<image1> x" if isinstance(tokens, list) else '{"edit_instruction": "make it red", "target_description": "a red cube"}'
Node.execute(clip, "make <image1> red", mode="json", images={"image_1": image}, **common)
assert "picture 1" not in clip.calls[0][0] and sum("[GUIDANCE" in text for text, _ in clip.calls) == 1
# Encode negative defaults to auto: an empty negative is not encoded
clip = Clip("unused")
Node.execute(clip, "a red cube", mode="normal", keep_model_loaded=True, resolution=64, force_rewrite=True)
assert clip.encoded == ["a red cube"]

# JSON mode with a connected system prompt that asks for prose: it stays a content guide, the format stays JSON
style = "You are a poet. Answer in flowing prose paragraphs only. Favor warm colors."
clip = Replies(['A warm red cube glows softly.', '{"description": "a warm red cube", "style": "warm palette"}'])
prompt, *_ = Node.execute(clip, "a red cube", mode="json", system_prompt=style, **common).args
asked = clip.calls[0][0]
assert style in asked and node_module.JSON_T2I_PROMPT in asked and "ignore any output format it asks for" in asked
assert asked.rindex("[REQUIRED_OUTPUT") > asked.index(style)              # the format comes last and wins
assert prompt == '{"description":"a warm red cube","style":"warm palette"}' and clip.generated == 2
assert any(style in text and "not usable" in text for text, _ in clip.calls)   # the retry keeps the guide

# Enhance given a JSON answer builds prose from its fields instead of failing; JSON is never encoded in Enhance
clip = Replies(['{"edit_instruction": "Put the red coat from <image2> on the woman in <image1>", '
                '"target_description": "a knee-length red wool coat", "preserve": ["her face", "the beach"]}'])
prompt, *_ = Node.execute(clip, "coat of <image2> on <image1>", mode="enhance", images={"image_1": image, "image_2": wide}, **common).args
assert prompt.startswith("Put the red coat from <image2> on the woman in <image1>.") and "Keep her face, the beach unchanged." in prompt
assert not prompt.lstrip().startswith("{") and clip.generated == 1
clip = Replies(['{"description": "A harbor at dawn", "lighting": "soft gold light"}'])
prompt, *_ = Node.execute(clip, "a harbor", mode="enhance", **common).args
assert prompt == "A harbor at dawn. Lighting: soft gold light."


# the VAE gets all four channels of an RGBA picture (as natively); the vision tower gets alpha over white
class CountingVAE:
    def __init__(self):
        self.first_stage_model, self.seen = torch.nn.Identity(), []

    def encode(self, pixels):
        self.seen.append(pixels.shape[-1])
        return torch.zeros(1, 16, pixels.shape[1] // 8, pixels.shape[2] // 8)


rgba = torch.rand(1, 64, 64, 4)
vae, clip = CountingVAE(), Clip("unused")
Node.execute(clip, "make <image1> blue", mode="normal", images={"image_1": rgba}, vae=vae, **common)
assert vae.seen == [4] and all(k["images"][0].shape[-1] == 3 for _, k in clip.calls if k.get("images"))

# Qwen-Image-2.1's real template: its own system turn and <imageN> labels, no duplicated labels
import comfy.text_encoders.qwen_image21 as q21


class TemplateClip:
    def __init__(self):
        self.tokenizer = q21.QwenImage21Tokenizer()

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize_with_weights(text, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


tokens, rows = node_module.rewrite_tokens(TemplateClip(), "[GUIDANCE]\nrules\n[/GUIDANCE]\n\nput <image1> in <image2>",
                                          [image, wide], False)
seen = TemplateClip().decode([t[0] for row in rows for t in row if isinstance(t, tuple) and isinstance(t[0], int)])
assert seen.startswith("<|im_start|>system\nComprehend and analyze the provided prompt.") and "picture 1" not in seen
assert sum(isinstance(t[0], dict) for row in rows for t in row if isinstance(t, tuple)) == 2

print("PASS: normal/enhance/json; Generate Text keywords; image list; truncated -> Normal; refiner writes, main encodes; writer choice; rewrite reused when nothing changed; vision tower once per picture in a run and across runs; cut-off reasoning rejected")
