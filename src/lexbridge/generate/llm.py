"""LLM wrapper: Ollama → OpenAI-compatible API → deterministic MockLLM.

Every call passes a ``task`` and a structured ``context``. Real backends only read the prompt;
the mock uses the context so the whole pipeline runs offline and deterministically.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from loguru import logger

from ..config import get_config


class BaseLLM:
    name = "base"

    def complete(self, prompt: str, *, system: str = "", task: str = "answer", context: dict | None = None) -> str:
        raise NotImplementedError

    def complete_json(self, prompt: str, *, system: str = "", task: str, context: dict | None = None) -> dict:
        raw = self.complete(prompt, system=system, task=task, context=context)
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


def _post(url: str, payload: dict, headers: dict | None = None, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class OllamaLLM(BaseLLM):  # pragma: no cover - needs a running Ollama
    def __init__(self, model: str, url: str, temperature: float = 0.0):
        self.model, self.url, self.temperature, self.name = model, url.rstrip("/"), temperature, f"ollama:{model}"

    def complete(self, prompt, *, system="", task="answer", context=None):
        r = _post(f"{self.url}/api/generate", {"model": self.model, "prompt": prompt, "system": system,
                                                "stream": False, "options": {"temperature": self.temperature}})
        return r.get("response", "")


class OpenAICompatLLM(BaseLLM):  # pragma: no cover - needs an API
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 800):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.temperature, self.max_tokens, self.name = temperature, max_tokens, f"openai:{model}"

    def complete(self, prompt, *, system="", task="answer", context=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = _post(f"{self.base_url}/chat/completions",
                  {"model": self.model, "messages": msgs, "temperature": self.temperature, "max_tokens": self.max_tokens},
                  headers={"Authorization": f"Bearer {self.api_key}"})
        return r["choices"][0]["message"]["content"]


class HFLocalLLM(BaseLLM):  # pragma: no cover - needs model weights
    """Local CPU generation with transformers (e.g. Qwen2.5-1.5B-Instruct). Greedy decoding."""

    def __init__(self, model: str, max_new_tokens: int = 400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
        self.model.eval()
        self.max_new_tokens, self.name = max_new_tokens, f"hf:{model}"

    def complete(self, prompt, *, system="", task="answer", context=None):
        import torch
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        limit = 96 if task in ("transferability", "verify") else self.max_new_tokens
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=limit, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class MockLLM(BaseLLM):
    """Deterministic stand-in. Extractive: it only ever repeats text from the context it is given."""
    name = "mock"

    def complete(self, prompt, *, system="", task="answer", context=None):
        context = context or {}
        fn = getattr(self, f"_task_{task}", None)
        return fn(context) if fn else ""

    # --- C3 transferability -------------------------------------------------------------
    @staticmethod
    def _task_transferability(ctx: dict) -> str:
        d = ctx.get("diff", {})
        if d.get("best_sub_ratio", 0) >= 0.95 and d.get("n_new_subsections", 1) > 1:
            label, why = "merged", "the old text survives verbatim as one sub-section; other sub-sections were added"
        elif d.get("ingredient_ratio", 0) >= 0.9 and d.get("punishment_ratio", 1) < 0.9:
            label, why = "punishment_changed", "ingredients are unchanged but the punishment clause differs"
        elif d.get("ratio", 0) >= 0.85:
            label, why = "reworded", "the provisions differ only in wording"
        else:
            label, why = "ingredients_changed", "the definition of the offence differs substantially"
        return json.dumps({"label": label, "changed_elements": d.get("changed_elements", []), "rationale": why})

    # --- Generation ---------------------------------------------------------------------
    @staticmethod
    def _task_answer(ctx: dict) -> str:
        lines = []
        for s in ctx.get("statutes", []):
            lines.append(f"Under {s['code']} Section {s['section']} ({s['title']}): {_first_sentence(s['text'], ctx.get('query', ''))} "
                         f"[{s['code']} §{s['section']}]")
        for c in ctx.get("chunks", [])[:4]:
            sent = _best_sentence(c["text"], ctx.get("query", "") + " " + " ".join(ctx.get("keywords", [])))
            if sent:
                tag = f" ({c['tag']})" if c.get("tag") else ""
                lines.append(f"{sent} [{c['chunk_id']}]{tag}")
        if not lines:
            return "I could not find support for an answer in the retrieved material."
        return "\n".join(lines)

    @staticmethod
    def _task_parametric(ctx: dict) -> str:
        """No-RAG baseline: mimics an LLM whose memory is dominated by IPC-era text."""
        out = []
        for sec, title in ctx.get("parametric", []):
            out.append(f"Section {sec} deals with {title}." if title else f"I am not sure what Section {sec} covers.")
        return " ".join(out) or "I do not have enough information to answer."

    @staticmethod
    def _task_verify(ctx: dict) -> str:
        from .verifier import lexical_support
        return json.dumps({"supported": lexical_support(ctx.get("claim", ""), ctx.get("evidence", "")) >= 0.5})


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\[SYNTHETIC TEST FIXTURE[^\]]*\]\s*", "", text)
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if len(s.strip()) > 20]


def _toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+|\d+", s.lower()) if len(w) > 2}


def _best_sentence(text: str, query: str) -> str:
    q = _toks(query)
    sents = _sentences(text)
    if not sents:
        return ""
    return max(sents, key=lambda s: (len(q & _toks(s)), -abs(len(s) - 160)))


def _first_sentence(text: str, query: str = "") -> str:
    s = _sentences(text)
    return s[0] if s else text


_LLM: BaseLLM | None = None


def get_llm(backend: str | None = None) -> BaseLLM:
    global _LLM
    cfg = get_config().get("llm", {})
    backend = backend or os.environ.get("LLM_BACKEND") or cfg.get("backend", "auto")
    if backend == "mock":
        return MockLLM()
    if _LLM is not None and backend == "auto":
        return _LLM
    llm: BaseLLM | None = None
    if backend in ("auto", "openai") and cfg.get("base_url"):
        llm = OpenAICompatLLM(cfg["base_url"], cfg.get("api_key", ""), cfg.get("model", "gpt-4o-mini"),
                              cfg.get("temperature", 0.0), cfg.get("max_tokens", 800))
    elif backend in ("auto", "ollama"):
        try:
            urllib.request.urlopen(f"{cfg.get('ollama_url', 'http://localhost:11434')}/api/tags", timeout=0.5)
            llm = OllamaLLM(cfg.get("ollama_model"), cfg.get("ollama_url"), cfg.get("temperature", 0.0))
        except Exception:
            llm = None
    if llm is None and backend in ("auto", "hf") and cfg.get("hf_model"):
        try:
            llm = HFLocalLLM(cfg["hf_model"], cfg.get("max_tokens", 400))
        except Exception as e:  # pragma: no cover
            logger.warning(f"local HF model unavailable ({type(e).__name__}: {e})")
    if llm is None:
        logger.info("LLM backend: deterministic MockLLM")
        llm = MockLLM()
    else:
        logger.info(f"LLM backend: {llm.name}")
    if backend == "auto":
        _LLM = llm
    return llm
