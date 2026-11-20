# Single Encoder Enhancer for ComfyUI

One node that rewrites your instruction with the text encoder's own language model and then encodes the result with
that same model. You need no second LLM, the model loads once, and each reference picture goes through the vision
tower once, for both the rewrite and the encode.

The node was built for Qwen-Image-2.1 (Qwen3-VL encoder). It works with any CLIP that ComfyUI's **Generate Text** node
can run. With other CLIPs, use **Normal** mode or connect a **refiner CLIP** to do the writing.

## Requirements
- ComfyUI **0.45** or newer. The node uses the V3 node API and `CLIP.generate`; it was tested on 0.45.3.
- No extra Python packages.

## Install
```
cd ComfyUI/custom_nodes
git clone https://github.com/Magirad/ComfyUI-SingleEncoderEnhancer
```
Restart ComfyUI, then refresh the browser.

## Modes
- **normal**: encodes the instruction as written, with no text generation. Works with any CLIP.
- **enhance**: the CLIP rewrites the instruction as prose, and the prose is encoded. *Prompt length x* sets the
  target length as a multiple of your word count; text-to-image doubles it. If the model answers with JSON anyway, the
  fields are turned into prose, so JSON is never encoded in this mode.
- **json**: the CLIP writes a structured JSON prompt, and the JSON text is encoded.
  - Edit fields: `edit_instruction`, `reference_roles`, `target_description`, `preserve`.
  - Text-to-image fields: `description`, `subjects`, `background`, `composition`, `lighting`, `style`, `rendered_text`.
  - If the answer is not a valid object with the required keys, the node retries once. If the retry also fails, the
    node stops with an error and encodes nothing.

## Images and edits
- **Image sockets:** connect pictures to the image sockets; a new socket appears each time the last one is used.
  - Refer to the pictures as `<image1>`, `<image2>`, ... in the instruction.
  - A batch connected to one socket is split into single pictures.
  - With pictures connected the job is treated as an edit; without them, as text-to-image.
- **Resolution:** pictures are resized to about *Resolution*² pixels, keeping the aspect ratio. The rewrite and the
  encode see the same resized pictures.
- **VAE:** connect a VAE to add reference latents to both conditionings. The VAE gets all channels, including alpha;
  the vision tower gets the picture composited over white. Without a VAE the pictures only guide the rewrite.
- **`latent` output:** an empty latent in the VAE's layout, at image 1's size, or *Resolution*² without pictures.

## Writer and refiner
- **refiner_clip** (optional): a second text model that can do the writing. The main CLIP always encodes.
- **Writer**:
  - *auto* and *main*: the main CLIP writes when it can, so no second model is loaded.
  - *refiner*: the refiner writes.
  - If the chosen model cannot write, the other one does. If neither can, the instruction is encoded as written.

## Other options
- **System prompt** (optional input): your own rewrite guidance. The model's own system turn is kept, and the node
  still adds the output-format rules for the selected mode, so guidance that asks for a different format cannot break
  the encode.
- **Negative prompt**: in Enhance and JSON it is also given to the rewrite as things to avoid.
- **Encode negative**:
  - *on*: encode the negative prompt.
  - *off*: skip that encode and output a zeroed negative.
  - *auto*: off while the negative prompt is empty, on otherwise.
  - A zeroed negative gives the same image only at CFG 1.
- **Sampling** off (default): the rewrite is deterministic.
  - When it is on, the seed, temperature, top k, top p and min p settings apply.
  - Repetition and presence penalties apply either way. JSON mode switches them off.
- **Thinking**: reasoning before writing, on models whose chat template supports it. The reasoning shows in
  `raw_output` and is not encoded.
- **Fast encode (bf16)**:
  - Encodes in bf16 instead of ComfyUI's fp32 activations. That is faster, but the conditioning differs slightly.
  - Off reproduces the stock encode exactly.
  - Ignored on devices without fast bf16.
- **Force rewrite**: redoes the rewrite, the encode and the image features on every run. Use it for timing tests.
- **Keep model loaded**: off (default) unloads the CLIP after the node to free VRAM for the diffusion model.

## Outputs
- `prepared_prompt`: the text that was encoded.
- `positive` and `negative`: the conditionings.
- `raw_output`: the model's full answer, including any reasoning.
- `latent`: the empty latent described under Images and edits.

## Caches
Every cache is cleared by any change to what it depends on: the weights, patches, layer, precision, text or pictures.
- **Rewrite:** an unchanged request reuses its stored rewrite.
- **Vision features:** each picture's features are computed once and shared by the rewrite and the encode. They are
  reused on later runs.
- **Encode:** encodes are reused for unchanged tokens, including edit encodes with pictures.
- **Reference latents:** an unchanged picture's reference latent is reused.

## Limits
- If no connected CLIP can write text (for example T5, CLIP-L, the Flux or SD3 encoders, or truncated encoders),
  Enhance and JSON fall back to Normal. The instruction is encoded as written and `raw_output` says why.
- Batches are not capped. Every picture adds roughly 1,000 tokens at 1024², so many pictures make the rewrite and
  the encode slow and memory-heavy.
- GGUF CLIPs: the ComfyUI-GGUF loader loads vision weights (mmproj) only for Qwen2-VL. Qwen3-VL or Gemma GGUF CLIPs
  therefore cannot see the pictures.
- Detection or grounding VL models (for example WeVisOne) do not write prompts. They tend to echo the input.

## Tests
From the ComfyUI portable folder:
```
python_embeded\python.exe -s ComfyUI\custom_nodes\ComfyUI-SingleEncoderEnhancer\tests\verify.py
```

## License
GPL-3.0, the same licence as ComfyUI. See [LICENSE](LICENSE).
