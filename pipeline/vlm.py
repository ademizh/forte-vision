"""
Local VLM (Qwen2.5-VL-7B-Instruct, Apache-2.0). Runs fully offline from VLM_PATH; load once with get_vlm().

  choice_probs(prompts, n)           P(answer letter) for (system, frames, question) prompts - no generated text
  frame_question_probs(sys, frames, questions)   many yes/no questions per frame, the frame encoded once (KV cache)
  generate(prompts)                  short free text (the one-sentence description of an event)
"""
import os
import threading
import time

from .frames import Frame

MODEL_PATH = os.environ.get("VLM_PATH", "C:/models/qwen2.5-vl-7b")
QUANT = os.environ.get("VLM_QUANT", "")            # "" | "4bit"
MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "200"))
# Each 640x360 frame is ~300 visual tokens; cap per-image pixels so windows stay cheap.
MAX_PIXELS = int(os.environ.get("VLM_MAX_PIXELS", "921600"))   # full 464x848 frames; thresholds calibrated at this size


def _patch_vision_attention():
    """transformers 4.49 builds a (all frames x all frames) mask in the vision encoder, which needs
    >10 GB for 12 frames and grows with the square of the frame count. Frames never attend to each
    other (cu_seqlens marks the boundaries), so attend one segment at a time: same result
    (checked: max diff 7e-8), tiny memory."""
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as m

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
        n = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(n, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos().float(), emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = m.apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)      # heads, seq, dim
        bounds = cu_seqlens.tolist()
        out = torch.empty_like(q)
        for a, b in zip(bounds[:-1], bounds[1:]):
            out[:, a:b] = F.scaled_dot_product_attention(q[:, a:b], k[:, a:b], v[:, a:b], dropout_p=0.0)
        return self.proj(out.transpose(0, 1).reshape(n, -1))

    m.Qwen2_5_VLVisionSdpaAttention.forward = forward


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_lock = threading.Lock()
_instance = None


def get_vlm():
    global _instance
    with _lock:
        if _instance is None:
            _instance = QwenVLM(MODEL_PATH)
    return _instance


class QwenVLM:
    def __init__(self, path: str):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model folder not found: {path} (set VLM_PATH)")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available")

        _patch_vision_attention()
        t0 = time.time()
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "cuda", "attn_implementation": "sdpa",
                  "local_files_only": True}
        if QUANT == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            path, min_pixels=128 * 28 * 28, max_pixels=MAX_PIXELS, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.torch = torch
        print(f"[vlm] loaded {path} in {time.time() - t0:.0f}s")

    def _build(self, system: str, frames: list[Frame], user: str):
        content = []
        for i, f in enumerate(frames):
            content.append({"type": "text", "text": f"Frame {i}, timestamp {f.t:.1f} s:"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": user})
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: list[tuple[str, list[Frame], str]], max_new_tokens: int | None = None) -> list[str]:
        """prompts: list of (system_text, frames, user_text). Returns raw answers, same order.
        On GPU out-of-memory, splits the batch in half and retries instead of crashing."""
        torch = self.torch
        texts = [self._build(s, fr, u) for s, fr, u in prompts]
        images = [f.image for _, fr, _ in prompts for f in fr]
        try:
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
                                          do_sample=False, temperature=None, top_p=None, top_k=None)
            out = out[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(out, skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(prompts) > 1:
                half = len(prompts) // 2
                print(f"[vlm] out of memory with batch {len(prompts)}, retrying as {half}+{len(prompts) - half}")
                return self.generate(prompts[:half], max_new_tokens) + self.generate(prompts[half:], max_new_tokens)
            raise RuntimeError("GPU out of memory on a single window: lower VLM_MAX_PIXELS "
                               "(e.g. 230400 -> 160000) or WINDOW_SEC") from None
        finally:
            torch.cuda.empty_cache()

    # ---- probabilistic inference (Zhang et al., NeurIPS 2024, "Why are VLMs bad at image classification") ----
    # Instead of letting the model WRITE an answer (and a made-up confidence), we read the probability it
    # assigns to each answer letter in a single forward pass. This gives a real, tunable score, never
    # produces an invalid answer, and is faster than generation.

    def _letter_ids(self, n: int) -> list[int]:
        tok = self.processor.tokenizer
        ids = []
        for letter in LETTERS[:n]:
            t = tok.encode(letter, add_special_tokens=False)
            if len(t) != 1:
                raise ValueError(f"letter {letter!r} is not a single token")
            ids.append(t[0])
        return ids

    def choice_probs(self, prompts: list[tuple[str, list[Frame], str]], n_options: int,
                     return_features: bool = False):
        """Probability of each option letter (A, B, ...) for each prompt; rows sum to 1.
        With return_features=True also returns the last-token hidden state per prompt (for linear probes)."""
        torch = self.torch
        letter_ids = self._letter_ids(n_options)
        texts = [self._build(s, fr, u) + "" for s, fr, u in prompts]
        images = [f.image for _, fr, _ in prompts for f in fr]
        head, captured = self.model.lm_head, {}

        class LastTokenHead(torch.nn.Module):          # compute logits for the last position only
            def __init__(self, h):
                super().__init__()
                self.h = h

            def forward(self, x):
                captured["feat"] = x[:, -1, :].float()
                return self.h(x[:, -1:, :])

        try:
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
            self.model.lm_head = LastTokenHead(head)
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits[:, -1, :]
            probs = torch.softmax(logits[:, letter_ids].float(), dim=-1).cpu().tolist()
            feats = captured["feat"].cpu().numpy() if return_features else None
        except torch.cuda.OutOfMemoryError:
            self.model.lm_head = head
            torch.cuda.empty_cache()
            if len(prompts) == 1:
                raise RuntimeError("GPU out of memory on a single window: lower VLM_MAX_PIXELS") from None
            half = len(prompts) // 2
            a = self.choice_probs(prompts[:half], n_options, return_features)
            b = self.choice_probs(prompts[half:], n_options, return_features)
            if not return_features:
                return a + b
            import numpy as np
            return a[0] + b[0], np.concatenate([a[1], b[1]])
        finally:
            self.model.lm_head = head
            torch.cuda.empty_cache()
        return (probs, feats) if return_features else probs

    # ---- many yes/no questions about the same frame: encode the frame ONCE, reuse it for every question ----
    # The prompt is split right after the image (<|vision_end|> is a special token, so the split is exact).
    # The frame part is run once and kept in the KV cache; each question then costs only ~30 tokens.
    # Same probabilities as choice_probs on the full prompt (checked on a small Qwen2.5-VL), ~10x faster
    # for 13 questions, because the image tokens (~1000 per frame) dominate the cost.

    def _prefix_probs(self, inputs, suffixes: list[list[int]], letter_ids: list[int]):
        """inputs: processor output for the shared prefix (batch of frames). suffixes: token ids per question.
        Returns probs[frame][question][option]."""
        torch = self.torch
        from transformers import DynamicCache
        model, head = self.model, self.model.lm_head

        class LastTokenHead(torch.nn.Module):
            def __init__(self, h):
                super().__init__()
                self.h = h

            def forward(self, x):
                return self.h(x[:, -1:, :])

        dev = inputs["input_ids"].device
        bsz, plen = inputs["input_ids"].shape
        out = [[None] * len(suffixes) for _ in range(bsz)]
        cache = DynamicCache()
        try:
            model.lm_head = LastTokenHead(head)
            with torch.inference_mode():
                model.rope_deltas = None
                model(**inputs, past_key_values=cache, use_cache=True)
                for qi, suf in enumerate(suffixes):
                    ids = torch.tensor([suf] * bsz, device=dev)
                    am = torch.cat([inputs["attention_mask"], torch.ones_like(ids)], dim=1)
                    logits = model(input_ids=ids, attention_mask=am, past_key_values=cache, use_cache=True,
                                   cache_position=torch.arange(plen, plen + len(suf), device=dev)).logits[:, -1, :]
                    p = torch.softmax(logits[:, letter_ids].float(), dim=-1).cpu().tolist()
                    for b in range(bsz):
                        out[b][qi] = p[b]
                    cache.crop(plen)
        finally:
            model.lm_head = head
            model.rope_deltas = None
        return out

    def frame_question_probs(self, system: str, frames: list[Frame], questions: list[str], n_options: int = 2):
        """P(option) for every (frame, question) pair: result[frame][question] = [pA, pB, ...]."""
        torch = self.torch
        letter_ids = self._letter_ids(n_options)
        dummy = Frame(0.0, frames[0].image)                    # same text for every frame (no timestamp)
        marker = "<|vision_end|>"
        prefix = None
        suffix_ids = []
        for q in questions:
            full = self._build(system, [dummy], q)
            cut = full.index(marker) + len(marker)
            if prefix is None:
                prefix = full[:cut]
            elif full[:cut] != prefix:
                raise RuntimeError("prompt prefix differs between questions")
            suffix_ids.append(self.processor.tokenizer.encode(full[cut:], add_special_tokens=False))
        try:
            inputs = self.processor(text=[prefix] * len(frames), images=[f.image for f in frames],
                                    return_tensors="pt", padding=True).to("cuda")
            return self._prefix_probs(inputs, suffix_ids, letter_ids)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(frames) == 1:
                raise RuntimeError("GPU out of memory on a single frame: lower VLM_MAX_PIXELS") from None
            h = len(frames) // 2
            return (self.frame_question_probs(system, frames[:h], questions, n_options)
                    + self.frame_question_probs(system, frames[h:], questions, n_options))
        finally:
            torch.cuda.empty_cache()


# ---------------- prompts ----------------

SYSTEM = (
    "You are a compliance auditor reviewing silent CCTV footage from a bank cash desk. "
    "The camera is mounted above. People: a cashier behind the counter, clients, sometimes cash collectors. "
    "Frames are in chronological order and each has its timestamp. "
    "Report only what is clearly visible. Never guess about amounts, documents, computer systems "
    "or anything not visible. Something not visible in these sampled frames is not proof that it did "
    "not happen. Do not identify people or describe faces."
)


OBSERVE_PROMPT = (
    "In one sentence in Russian, describe what the people in these frames do with cash, bags, the safe, "
    "keys, doors, documents or the computer. Only visible facts. Do not judge and do not mention rules."
)


# Factual yes/no questions about ONE frame. Facts are far more reliable than "is this a violation?":
# the VLM only reports what is visible; rules in code (over time) decide what is a violation.
FACTS = {
    # who is where
    "person_visible": "Is at least one person visible anywhere in this frame?",
    "cashier_at_desk": "Is the cashier (the employee) at their workplace behind the cash desk, sitting or standing there?",
    "client_at_desk": "Is a client (a visitor, not the employee) in the room on the client side of the counter?",
    "outsider_behind_counter": "Is any person other than the cashier on the cashier's side of the counter or next to the safe?",
    "collector_present": "Is a cash collector (a person bringing or taking a cash bag, often in uniform) in the room?",
    # cash and valuables
    "cash_on_desk": "Are there any banknotes, cash or stacks of money lying anywhere on the desk or table, even small or partly hidden?",
    "cash_in_hands": "Is someone holding, counting or handing over banknotes?",
    "collection_bag_visible": "Is a cash collection bag (a sealed bag for transporting cash) visible anywhere?",
    "safe_open": "Is the door of the safe open?",
    "key_in_safe_lock": "Is there a key inserted in the lock of the safe?",
    # workplace
    "personal_item_on_desk": "Is a personal item (handbag, backpack, mobile phone, clothing) lying on or right next to the cash desk?",
    "service_door_open": "Is the door of the room open?",
    "computer_screen_on": "Is the computer screen switched on and showing a program?",
    "drink_over_cash": "Is someone holding a cup or drink directly above money on the desk?",
}


def fact_prompt(question: str) -> str:
    """One fact as a 2-option question, scored by answer probability (no JSON, no null)."""
    return (f"Look at THIS frame only. {question}\n"
            "A. Yes\nB. No\nAnswer with the letter only.")
