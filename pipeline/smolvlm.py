"""
SmolVLM2 (HuggingFaceTB/SmolVLM2-2.2B-Instruct) for event verification, loaded once and reused.

    vlm = SmolVLM(path, device="cuda", dtype="bfloat16", longest_edge=1152, image_splitting=True)
    probs = vlm.choice(frames, question, options)     # probability of each option (sums to 1)
    text = vlm.generate(frames, prompt, max_new_tokens=96)

Loading follows the model card for transformers >= 4.50 (AutoProcessor + AutoModelForImageTextToText, chat
template with {"type": "image"} entries, images passed to the processor as one list per prompt).

Why answer-letter probabilities: the model is never asked to WRITE a confidence. One forward pass gives the
next-token probabilities of the option letters ("A", " A", ...), renormalised over the options. With
both_orders the options are asked in two orders and averaged (cancels the bias towards the first letter).
These numbers are scores, not calibrated probabilities: thresholds are calibrated on clean videos.
"""
from __future__ import annotations

import time

LETTERS = "ABCDEFGHIJ"


def _dtype(torch, name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


class SmolVLM:
    name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    def __init__(self, path: str, device: str = "cuda", dtype: str = "bfloat16", longest_edge: int = 1152,
                 image_splitting: bool = True, both_orders: bool = True):
        import os

        if not os.path.isdir(path):
            raise FileNotFoundError(f"SmolVLM2 folder not found: {path}  (download HuggingFaceTB/SmolVLM2-2.2B-Instruct "
                                    f"and set models.vlm_path in configs/verified.yaml or SMOLVLM_PATH)")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available: set models.device: cpu (slow) or install a CUDA build of torch")
        t0 = time.time()
        self.torch = torch
        self.device = device
        self.dtype = _dtype(torch, dtype if device.startswith("cuda") else "float32")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        ip = self.processor.image_processor
        if hasattr(ip, "size") and isinstance(ip.size, dict) and "longest_edge" in ip.size:
            ip.size = {"longest_edge": int(longest_edge)}
        if hasattr(ip, "do_image_splitting"):
            ip.do_image_splitting = bool(image_splitting)
        kw = {"torch_dtype": self.dtype, "local_files_only": True}
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(path, _attn_implementation="sdpa", **kw)
        except (ValueError, ImportError):
            self.model = AutoModelForImageTextToText.from_pretrained(path, **kw)
        self.model = self.model.to(device).eval()
        self.tok = self.processor.tokenizer
        self.both_orders = both_orders
        self.calls = 0
        self.load_sec = round(time.time() - t0, 1)
        print(f"[smolvlm] loaded {path} in {self.load_sec}s on {device}")

    # ---- prompt
    def _inputs(self, frames, user_text: str):
        content = []
        for k, f in enumerate(frames):
            content.append({"type": "text", "text": f"Frame {k} ({f.t:.1f} s):"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[[f.image for f in frames]] if frames else None,
                                return_tensors="pt")
        out = {}
        for k, v in inputs.items():
            v = v.to(self.device)
            out[k] = v.to(self.dtype) if v.is_floating_point() else v
        return out

    def _letter_ids(self, n: int) -> list[list[int]]:
        ids = []
        for L in LETTERS[:n]:
            alts = set()
            for variant in (L, " " + L):
                t = self.tok.encode(variant, add_special_tokens=False)
                if len(t) == 1:
                    alts.add(t[0])
            if not alts:
                raise ValueError(f"no single-token id for option letter {L}")
            ids.append(sorted(alts))
        return ids

    def _forward_probs(self, frames, user_text: str, n: int) -> list[float]:
        torch = self.torch
        try:
            inputs = self._inputs(frames, user_text)
            with torch.inference_mode():
                logits = self.model(**inputs).logits[0, -1].float()
            self.calls += 1
            p = torch.softmax(logits, dim=-1)
            mass = [float(sum(p[i] for i in alts)) for alts in self._letter_ids(n)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(frames) > 1:
                print(f"[smolvlm] out of memory with {len(frames)} frames: retrying with {len(frames) // 2}")
                return self._forward_probs(frames[::2], user_text, n)
            raise RuntimeError("GPU out of memory on one image: lower vlm.longest_edge or set image_splitting: false")
        tot = sum(mass)
        return [m / tot for m in mass] if tot > 0 else [1.0 / n] * n

    def choice(self, frames, question: str, options: list[str]) -> list[float]:
        n = len(options)

        def ask(order):
            lines = "\n".join(f"{LETTERS[j]}. {options[o]}" for j, o in enumerate(order))
            pr = self._forward_probs(frames, f"{question}\n{lines}\nAnswer with the letter only.", n)
            out = [0.0] * n
            for j, o in enumerate(order):
                out[o] = pr[j]
            return out
        fwd = ask(list(range(n)))
        if not self.both_orders:
            return fwd
        rev = ask(list(range(n))[::-1])
        return [(a + b) / 2 for a, b in zip(fwd, rev)]

    def generate(self, frames, prompt: str, max_new_tokens: int = 96) -> str:
        torch = self.torch
        inputs = self._inputs(frames, prompt)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        self.calls += 1
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def memory(self) -> dict:
        torch = self.torch
        if not torch.cuda.is_available():
            return {}
        return {"allocated_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}


class MockVLM:
    """Deterministic fake with the same interface (tests / plumbing). answers: {question substring: [pA, pB, pC]}."""
    name = "mock"

    def __init__(self, answers: dict | None = None, default=(0.34, 0.33, 0.33)):
        self.answers, self.default, self.calls, self.log = answers or {}, list(default), 0, []

    def choice(self, frames, question, options):
        self.calls += 2
        self.log.append({"question": question, "times": [f.t for f in frames]})
        for k, v in self.answers.items():
            if k in question:
                return list(v)[:len(options)]
        return self.default[:len(options)]

    def generate(self, frames, prompt, max_new_tokens=96):
        self.calls += 1
        return '{"observed": true, "reason": "mock", "supporting_frames": [0], "uncertainty": "mock"}'

    def memory(self):
        return {}
