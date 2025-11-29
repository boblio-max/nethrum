from typing import Dict, Any
import re
import os
from g4f.client import Client
class Secretary:
    """
    Light NL parser + optional LLM wrapper.
    Set OPENAI_API_KEY env var to enable OpenAI calls and set llm_provider='openai'
    when instantiating in ui.py.
    """
    def __init__(self, llm_provider: str = None, llm_key: str = None):
        self.name = 'AI_Secretary'
        self.llm_provider = llm_provider
        self.llm_key = llm_key or os.environ.get('OPENAI_API_KEY')

    def parse(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        res = {'targets': [], 'action': None, 'symbol': None, 'params': {}}
        agents = ['infra', 'research', 'quant', 'backtest', 'risk', 'portfolio', 'execution']
        lowered = text.lower()
        if 'run pipeline on' in lowered or 'run pipeline' in lowered or 'run tests on' in lowered:
            res['action'] = 'pipeline'
        # detect ticker-like token (1-5 letters, case-insensitive, convert to uppercase)
        # First try uppercase match, then try case-insensitive for words after "on" or standalone tickers
        m = re.search(r"\b([A-Z]{1,5})\b", text)
        if m:
            res['symbol'] = m.group(1)
        else:
            # Try to find ticker after common keywords like "on", "for", etc.
            m2 = re.search(r"\bon\s+([a-zA-Z]{1,5})\b", text, re.IGNORECASE)
            if m2:
                res['symbol'] = m2.group(1).upper()
            else:
                # Last resort: find any 1-5 letter word that looks like a ticker
                m3 = re.search(r"\b([a-zA-Z]{1,5})\b(?=\s*$|\s+\w+=)", text)
                if m3 and m3.group(1).lower() not in ['run', 'on', 'for', 'the', 'and', 'all']:
                    res['symbol'] = m3.group(1).upper()
        for a in agents:
            if a in lowered:
                res['targets'].append(a)
        for kv in re.findall(r"(\w+)=([\w\.\-]+)", text):
            k, v = kv
            try:
                vnum = int(v); res['params'][k] = vnum; continue
            except: pass
            try:
                vfloat = float(v); res['params'][k] = vfloat; continue
            except: pass
            res['params'][k] = v
        if res['action'] == 'pipeline' and not res['targets']:
            res['targets'] = agents
        if 'all' in lowered or 'everything' in lowered:
            res['targets'] = agents
        return res

    def llm_brainstorm(self, prompt: str) -> str:
        if self.llm_provider == 'openai' and self.llm_key:
            try:
                import openai
                openai.api_key = self.llm_key
                r = openai.ChatCompletion.create(
                    model='gpt-4o-mini',
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=300
                )
                return r['choices'][0]['message']['content']
            except Exception as e:
                return f"LLM call failed: {e}"
        return f"[Secretary] brainstorm for: {prompt}\n- Try increasing lookback\n- Test momentum + mean-reversion mix"

    def makeSense(self, prompt: str):
            client = Client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Brainstorm a plan for {prompt}, so that it maximizes output and minimizes risk."}],
                web_search=False
            )
            return (response.choices[0].message.content)