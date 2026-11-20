import { app } from "/scripts/app.js";

const TYPE = "SingleEncoderEnhancer";

// The image sockets are added and removed by the frontend's own Autogrow handling; here they only get short labels.
function labelImages(node) {
    for (const input of node.inputs ?? []) if (input.name.startsWith("images.image_")) {
        input.label = `image ${input.name.split("_").at(-1)}`;
    }
}

// Seed/temperature/top-k... only matter while Sampling is on (like Generate Text's sampling mode).
const SAMPLING = ["seed", "control_after_generate", "temperature", "top_k", "top_p", "min_p"];
// Used by the Enhance/JSON rewrite only; the penalties apply with Sampling on or off.
const REWRITE = ["max_length", "do_sample", "repetition_penalty", "presence_penalty", "thinking", "force_rewrite"];

function hook(node, widget) {
    if (!widget || widget.enhancerHook) return;
    const original = widget.callback;
    widget.callback = function (...args) { const result = original?.apply(this, args); refresh(node); return result; };
    widget.enhancerHook = true;
}

// Widgets that do not apply are greyed out (disabled), never hidden: hiding removes their rows, so the node
// would change size every time Mode, Sampling or the refiner connection changes.
function updateWidgets(node) {
    const get = name => node.widgets?.find(widget => widget.name === name);
    const mode = get("mode"), sample = get("do_sample"), seed = get("seed"), negative = get("encode_negative");
    hook(node, mode);
    hook(node, sample);
    const normal = mode?.value === "normal";
    const refiner = node.inputs?.find(input => input.name === "refiner_clip");
    for (const widget of node.widgets ?? []) {
        if (widget.hidden) widget.hidden = false;   // earlier versions of this script hid widgets
        if (SAMPLING.includes(widget.name)) widget.disabled = normal || !sample?.value;
        else if (REWRITE.includes(widget.name)) widget.disabled = normal;
        else if (widget.name === "enhance_length") widget.disabled = mode?.value !== "enhance";
        else if (widget.name === "writer") widget.disabled = normal || refiner?.link == null;   // needs a refiner CLIP
    }
    // While the seed is not used (Sampling off or Normal) a constant is sent, so a seed set to randomize/increment
    // does not make ComfyUI run this node and everything after it again on every queue.
    if (seed && !seed.enhancerSeed) {
        const original = seed.serializeValue;
        seed.serializeValue = function (...args) {
            if (get("mode")?.value === "normal" || !get("do_sample")?.value) return 0;
            return original ? original.apply(this, args) : this.value;
        };
        seed.enhancerSeed = true;
    }
    // workflows saved while Encode negative was an on/off switch
    if (negative && typeof negative.value === "boolean") negative.value = negative.value ? "on" : "off";
    node.setDirtyCanvas?.(true, true);
}

const pending = new WeakSet();
function refresh(node) {
    if (pending.has(node)) return;
    pending.add(node);
    requestAnimationFrame(() => { pending.delete(node); updateWidgets(node); labelImages(node); });
}

app.registerExtension({
    name: "single_encoder_enhancer.ui",
    beforeRegisterNodeDef(nodeType, data) {
        if (data.name !== TYPE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            if (this.size[0] < 440) this.setSize([440, this.size[1]]);   // width set once, at creation only
            return result;
        };
        for (const event of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[event];
            nodeType.prototype[event] = function (...args) {
                const result = original?.apply(this, args);
                refresh(this);
                return result;
            };
        }
    },
});
