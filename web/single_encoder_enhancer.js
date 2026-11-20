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

// Widgets that do not apply are hidden. Like Generate Text, the node then shrinks or grows from the bottom by exactly
// the rows that were hidden or shown, so the instruction box keeps its size (a multiline box takes any spare height,
// so keeping the node size would make the box grow instead).
function updateWidgets(node) {
    const before = node.computeSize?.()[1];
    const get = name => node.widgets?.find(widget => widget.name === name);
    const mode = get("mode"), sample = get("do_sample"), seed = get("seed"), negative = get("encode_negative");
    hook(node, mode);
    hook(node, sample);
    const normal = mode?.value === "normal";
    const refiner = node.inputs?.find(input => input.name === "refiner_clip");
    const size = [...node.size];
    for (const widget of node.widgets ?? []) {
        let unused = null;
        if (SAMPLING.includes(widget.name)) unused = normal || !sample?.value;
        else if (REWRITE.includes(widget.name)) unused = normal;
        else if (widget.name === "enhance_length") unused = mode?.value !== "enhance";
        else if (widget.name === "writer") unused = normal || refiner?.link == null;   // needs a refiner CLIP
        if (unused === null) continue;
        widget.hidden = unused;
        widget.disabled = unused;   // also greyed out where a frontend ignores hidden
    }
    // change the height by the rows that appeared or disappeared; the user's extra height (and width) stays
    const after = node.computeSize?.()[1];
    if (before != null && after != null && after !== before) node.setSize?.([size[0], Math.max(after, size[1] + after - before)]);
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
