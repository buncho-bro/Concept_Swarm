import urllib.request
import json
import re

def query_ollama(model, prompt, system_prompt=None, timeout=60):
    """
    ローカルの Ollama API (http://localhost:11434/api/generate) に問い合わせを行う
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as res:
            data = json.loads(res.read().decode("utf-8"))
            raw_text = data.get("response", "")
            return raw_text
    except Exception as e:
        print(f"[Ollama Client Error] {e}")
        return None

def extract_code_block(text):
    """
    LLMが出力したMarkdownコードブロック (```python ... ```) から純粋なコードを抽出
    """
    if not text:
        return ""
    # ```python ... ``` または ``` ... ```
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # コードブロックがない場合はそのまま返す
    return text.strip()
