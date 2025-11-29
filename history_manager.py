from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime


class HistoryManager:
    """Simple run history manager with optional JSON persistence."""

    def __init__(self, path: Optional[str] = None, max_items: int = 1000):
        self.path = path or os.path.join(os.getcwd(), 'history.json')
        self.max_items = max_items
        self._runs: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._runs = data[-self.max_items:]
            except Exception:
                # ignore load errors, start fresh
                self._runs = []

    def _persist(self):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self._runs[-self.max_items:], f, default=str, indent=2)
        except Exception:
            pass

    def record_run(self, symbol: str, run_type: str, results: Dict[str, Any]):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'type': run_type,
            'results': results
        }
        self._runs.append(entry)
        if len(self._runs) > self.max_items:
            self._runs = self._runs[-self.max_items:]
        self._persist()

    def get_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        return list(self._runs[-limit:])

    def clear(self):
        self._runs = []
        self._persist()

