"""The judge's LLM: GPT-4o-mini through OpenRouter (https://openrouter.ai), temperature 0. Key: OPENROUTER_API_KEY in
the environment or in a .env file (see .env.example). Every call is logged in USAGE (tokens, seconds, US$)."""
import json
import os
import re
import time

GPT4O_MINI = "openai/gpt-4o-mini"
PRICES = {GPT4O_MINI: (0.15, 0.60)}                              # $ per 1M tokens in, out (if OpenRouter gives no cost)
USAGE = []
_client = None


def client():
    global _client
    if _client is None:
        try:
            from dotenv import find_dotenv, load_dotenv
            load_dotenv(find_dotenv(usecwd=True))
        except ImportError:
            pass
        from openai import OpenAI
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=120,
                         max_retries=0)
    return _client


def ask(prompt, step="", attempts=4, model=GPT4O_MINI):
    t0 = time.time()
    for attempt in range(1, attempts + 1):
        try:
            resp = client().chat.completions.create(model=model, max_tokens=8000, temperature=0,
                                                    messages=[{"role": "user", "content": prompt}],
                                                    extra_body={"usage": {"include": True}})
            if resp.choices and resp.usage:
                break
            raise RuntimeError("empty response")
        except Exception as e:
            print(f"  [API {step!r} attempt {attempt}/{attempts}] {type(e).__name__}: {str(e)[:200]}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(15 * attempt)
    u, (p_in, p_out) = resp.usage, PRICES.get(model, (0, 0))
    cost = getattr(u, "cost", None)
    USAGE.append({"model": model, "step": step, "in": u.prompt_tokens, "out": u.completion_tokens, "sec": time.time() - t0,
                  "usd": cost if cost is not None else (u.prompt_tokens * p_in + u.completion_tokens * p_out) / 1e6})
    return (resp.choices[0].message.content or "").strip()


def ask_json(prompt, step="", model=GPT4O_MINI):
    """First JSON value in the reply, or None."""
    raw = ask(prompt, step, model=model)
    text = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        start = min(i for i in (text.find("["), text.find("{")) if i >= 0)
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except Exception:
        print(f"  [not JSON] {raw[:300]!r}")
        return None


def ask_list(prompt, n, step, model=GPT4O_MINI):
    """A JSON list of n items (3 tries), else None."""
    for _ in range(3):
        out = ask_json(prompt, step, model)
        if isinstance(out, list) and len(out) == n:
            return out
    return None


def usd():
    return sum(r["usd"] for r in USAGE)
