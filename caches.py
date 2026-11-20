from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import logging
import weakref

import torch


def copy_features(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: copy_features(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_features(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_features(item) for item in value)
    return value


class Stored:
    """CPU snapshot of a tensor and the device it lived on."""
    __slots__ = ("tensor", "device")

    def __init__(self, tensor):
        self.tensor, self.device = tensor.detach().to("cpu", copy=True), tensor.device


def to_store(value):
    if torch.is_tensor(value):
        return Stored(value)
    if isinstance(value, dict):
        return {key: to_store(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_store(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_store(item) for item in value)
    return value


def from_store(value):
    if isinstance(value, Stored):
        return value.tensor.to(value.device, copy=True)
    if isinstance(value, dict):
        return {key: from_store(item) for key, item in value.items()}
    if isinstance(value, list):
        return [from_store(item) for item in value]
    if isinstance(value, tuple):
        return tuple(from_store(item) for item in value)
    return value


_MEMO = None


@contextmanager
def hash_memo():
    # One node run: the same pixels (same memory, not modified since) are hashed once, however many caches ask.
    global _MEMO
    previous, _MEMO = _MEMO, {}
    try:
        yield
    finally:
        _MEMO = previous


def pixel_key(pixels):
    memo, tag = _MEMO, None
    if memo is not None:
        # ComfyUI runs nodes in inference mode, where tensors have no version counter: the sum stands in for it,
        # so pixels changed in place since the last hash are hashed again.
        change = float(pixels.sum(dtype=torch.float64)) if pixels.is_inference() else pixels._version
        tag = (pixels.data_ptr(), tuple(pixels.shape), pixels.stride(), pixels.dtype, str(pixels.device), change)
        entry = memo.get(tag)
        if entry is not None:
            return entry[1]
    cpu = pixels.detach().to(device="cpu").contiguous()
    key = (tuple(cpu.shape), cpu.dtype, hashlib.sha256(cpu.view(torch.uint8).numpy()).digest())
    if memo is not None:
        memo[tag] = (pixels, key)   # the reference keeps this memory from being reused during the run
    return key


def stored_bytes(value):
    if isinstance(value, Stored):
        return value.tensor.numel() * value.tensor.element_size()
    if isinstance(value, dict):
        return sum(stored_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(stored_bytes(item) for item in value)
    return 0


class ImageStore:
    """Results that depend only on a picture and one set of weights, kept across runs on the CPU (LRU)."""

    def __init__(self, limit, max_bytes=None):
        self.limit, self.max_bytes = limit, max_bytes
        self.entries = OrderedDict()

    def lookup(self, key, owner):
        entry = self.entries.get(key)
        if entry is None or entry[0]() is not owner:
            return None
        self.entries.move_to_end(key)
        return from_store(entry[1])

    def store(self, key, owner, value):
        for stale in [k for k, entry in self.entries.items() if entry[0]() is None]:   # weights were freed
            del self.entries[stale]
        stored = to_store(value)
        self.entries[key] = (weakref.ref(owner), stored, stored_bytes(stored))
        self.entries.move_to_end(key)
        while len(self.entries) > self.limit or (self.max_bytes and len(self.entries) > 1
                                                 and sum(entry[2] for entry in self.entries.values()) > self.max_bytes):
            self.entries.popitem(last=False)


# 16 sockets, and the rewrite and the encode may ask with different per-picture options: 32 entries, at most 1.5 GB of RAM
VISION_STORE = ImageStore(32, max_bytes=3 << 29)
LATENT_STORE = ImageStore(32, max_bytes=1 << 30)   # reference latents of the last 32 pictures, at most 1 GB
REWRITE_STORE = ImageStore(32)   # (raw_output, prepared_prompt) of the last 32 rewrites


def rewrite_key(writer, mode, instruction, system_prompt, negative_prompt, enhance_length, thinking, options, pictures):
    # Everything the rewrite depends on. With sampling off the most likely token is taken, so the seed,
    # temperature and top-k/top-p/min-p cannot change the text and are left out (a changing seed still reuses).
    sampling = ("do_sample", "max_length", "repetition_penalty", "presence_penalty")
    if options.get("do_sample"):
        sampling += ("seed", "temperature", "top_k", "top_p", "min_p")
    return (id(writer.cond_stage_model), writer.patcher.patches_uuid, mode, instruction, system_prompt, negative_prompt,
            float(enhance_length), bool(thinking), tuple((name, options.get(name)) for name in sampling),
            tuple(pixel_key(picture) for picture in pictures))


class VisionFeatureCache:
    """Stands in for a text encoder's preprocess_embed: each picture passes the vision tower once, also across runs."""

    def __init__(self, preprocess, store=None, store_key=None, owner=None):
        self.preprocess = preprocess
        self.store, self.store_key, self.owner = store, store_key, owner
        self.entries = {}
        self.active = True
        self.hits = self.misses = self.restored = 0

    def __call__(self, embed, device):
        data = embed.get("data")
        if not self.active or embed.get("type") != "image" or not torch.is_tensor(data) or data.requires_grad:
            return self.preprocess(embed, device=device)
        try:   # per-picture options (token budget, block type) change the features
            options = repr(sorted((k, v) for k, v in embed.items() if k != "data"))
        except TypeError:
            return self.preprocess(embed, device=device)
        picture = pixel_key(data)
        key = (*picture[:2], str(data.device), str(device), options, picture[2])
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        if self.store is not None:   # same picture, same weights in an earlier run
            found = self.store.lookup((self.store_key, key), self.owner)
            if found is not None:
                self.entries[key] = found
                self.restored += 1
                return found
        result = self.preprocess(embed, device=device)
        # Isolate snapshots from producer work buffers; keep native dtype/device and any grid metadata.
        self.entries[key] = copy_features(result)
        if self.store is not None:
            self.store.store((self.store_key, key), self.owner, result)
        self.misses += 1
        return result

    def close(self):
        self.active = False
        self.entries.clear()


@contextmanager
def vision_feature_cache(clip, across_runs=True):
    # Any single-model text encoder whose transformer turns pictures into embeddings through preprocess_embed.
    name = getattr(clip.cond_stage_model, "clip", None)
    transformer = getattr(getattr(clip.cond_stage_model, name, None), "transformer", None) if isinstance(name, str) else None
    path = f"{name}.transformer.preprocess_embed"
    # Scheduled hooks can change vision weights between passes; another preprocess patch may change its contract.
    if (not callable(getattr(transformer, "preprocess_embed", None)) or clip.use_clip_schedule
            or clip.patcher.forced_hooks is not None or path in clip.patcher.object_patches):
        yield clip
        return
    # Same weights object and same patches (LoRA) -> same features for the same picture, also in a later run.
    store_key = (name, id(clip.cond_stage_model), clip.patcher.patches_uuid)
    cached_clip = clip.clone()
    cache = VisionFeatureCache(cached_clip.patcher.get_model_object(path), VISION_STORE if across_runs else None,
                               store_key, clip.cond_stage_model)
    cached_clip.patcher.add_object_patch(path, cache)
    try:
        yield cached_clip
    finally:
        cache.close()
        if cache.misses or cache.hits or cache.restored:
            logging.info("Single Encoder Enhancer vision cache: %d picture(s) computed, %d reused in this run, %d reused from earlier runs.",
                         cache.misses, cache.hits, cache.restored)


class CachedVAE:
    """The connected VAE with its reference-latent encodes kept across runs; everything else passes through."""

    def __init__(self, vae, store=LATENT_STORE):
        self.__dict__.update(_vae=vae, _store=store)

    def __getattr__(self, name):
        return getattr(self._vae, name)

    def __setattr__(self, name, value):
        setattr(self._vae, name, value)

    def encode(self, pixels, *args, **kwargs):
        vae = self._vae
        if args or kwargs or not torch.is_tensor(pixels) or pixels.requires_grad:
            return vae.encode(pixels, *args, **kwargs)
        owner = vae.first_stage_model
        key = (id(owner), vae.patcher.patches_uuid, pixel_key(pixels))
        found = self._store.lookup(key, owner)
        if found is not None:
            logging.info("Single Encoder Enhancer: reference latent reused from an earlier run.")
            return found
        latent = vae.encode(pixels)
        self._store.store(key, owner, latent)
        return latent


LATENT_SHAPES = OrderedDict()


def blank_latent_shape(vae, width, height):
    # The VAE's own latent layout for an empty latent: learned from one blank encode per VAE and size, then remembered.
    vae = getattr(vae, "_vae", vae)
    owner = getattr(vae, "first_stage_model", None)
    key = (id(owner), width, height)
    entry = LATENT_SHAPES.get(key) if owner is not None else None
    if entry is not None and entry[0]() is owner:
        return entry[1]
    shape = tuple(vae.encode(torch.ones([1, height, width, 3])).shape)
    if owner is not None:
        LATENT_SHAPES[key] = (weakref.ref(owner), shape)
        while len(LATENT_SHAPES) > 32:
            LATENT_SHAPES.popitem(last=False)
    return shape


def plain_tokens(value):
    # Hashable snapshot of the tokens: text as it is, pictures by a hash of their pixels; None for anything else.
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if torch.is_tensor(value):
        return None if value.requires_grad else ("tensor", *pixel_key(value))
    if isinstance(value, dict):
        items = tuple((key, plain_tokens(item)) for key, item in sorted(value.items(), key=lambda kv: str(kv[0])))
        return None if any(item is None and value[key] is not None for key, item in items) else items
    if isinstance(value, (list, tuple)):
        items = tuple(plain_tokens(item) for item in value)
        return None if any(item is None and raw is not None for item, raw in zip(items, value)) else items
    return None


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


class TextEncodeCache:
    """Conditioning of unchanged encodes (text, and text with pictures), kept across runs: a repeat loads no model."""
    limit = 16
    max_bytes = 1 << 30   # long multi-picture sequences: the oldest entries go once 1 GB is held

    def __init__(self):
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    def key(self, clip, tokens, unprojected, add_dict, fast_encode):
        snapshot = plain_tokens(tokens)
        if snapshot is None:
            return None
        try:
            extra = hashlib.sha256(repr((unprojected, sorted(add_dict.items()))).encode()).digest()
        except TypeError:
            return None
        # Same weights object, same patches (LoRA), same layer and precision -> same conditioning.
        return (id(clip.cond_stage_model), clip.patcher.patches_uuid, clip.layer_idx, bool(fast_encode),
                hashlib.sha256(repr(snapshot).encode()).digest(), extra)

    def lookup(self, key, owner):
        entry = self.entries.get(key)
        if entry is None or entry[0]() is not owner:
            return None
        self.entries.move_to_end(key)
        return copy_features(entry[1])

    def store(self, key, owner, conditioning):
        stored = copy_features(conditioning)
        self.entries[key] = (weakref.ref(owner), stored, tensor_bytes(stored))
        self.entries.move_to_end(key)
        while len(self.entries) > self.limit or (len(self.entries) > 1 and
                                                 sum(entry[2] for entry in self.entries.values()) > self.max_bytes):
            self.entries.popitem(last=False)


TEXT_ENCODE_CACHE = TextEncodeCache()


@contextmanager
def text_encode_cache(clip, fast_encode=False, cache=TEXT_ENCODE_CACHE):
    # Scheduled hooks make the encode depend on more than the tokens; leave those uncached.
    if clip.use_clip_schedule or clip.patcher.forced_hooks is not None:
        yield clip
        return
    encode = clip.encode_from_tokens_scheduled
    owner = clip.cond_stage_model
    hits, misses = cache.hits, cache.misses

    def cached(tokens, unprojected=False, add_dict={}, show_pbar=True):
        key = cache.key(clip, tokens, unprojected, add_dict, fast_encode)
        if key is not None:
            found = cache.lookup(key, owner)
            if found is not None:
                cache.hits += 1
                return found
        conditioning = encode(tokens, unprojected=unprojected, add_dict=add_dict, show_pbar=show_pbar)
        if key is not None:
            cache.misses += 1
            cache.store(key, owner, conditioning)
        return conditioning

    clip.encode_from_tokens_scheduled = cached
    try:
        yield clip
    finally:
        del clip.encode_from_tokens_scheduled
        if cache.hits - hits:
            logging.info("Single Encoder Enhancer text cache: %d encode(s) reused, %d computed.", cache.hits - hits, cache.misses - misses)
