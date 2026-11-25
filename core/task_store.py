"""Thread-safe JSON persistence for async image and video task queues."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple


class JsonTaskStore:
    """Thread-safe JSON dictionary store with automatic completed-history trimming."""

    ACTIVE_STATUSES = {"pending", "running", "queued", "in_progress", "processing", "completed", "succeeded"}

    def __init__(self, file_path: str, max_completed_history: int = 10) -> None:
        self.file_path = file_path
        self.max_completed_history = max(1, int(max_completed_history))
        self._lock = threading.RLock()

    def _read_unlocked(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _trim_unlocked(self, tasks: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        active: Dict[str, Dict[str, Any]] = {}
        finished: list[Tuple[str, Dict[str, Any]]] = []
        for tid, info in tasks.items():
            status = str(info.get("status", "pending")).lower()
            if status in self.ACTIVE_STATUSES:
                active[tid] = info
            else:
                finished.append((tid, info))

        if len(finished) > self.max_completed_history:
            finished.sort(key=lambda item: str(item[1].get("submitted_at", "1970-01-01 00:00:00")), reverse=True)
            finished = finished[: self.max_completed_history]

        merged = dict(active)
        for tid, info in finished:
            merged[tid] = info
        return merged

    def _write_unlocked(self, tasks: Dict[str, Dict[str, Any]]) -> None:
        parent = os.path.dirname(os.path.abspath(self.file_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        trimmed = self._trim_unlocked(tasks)
        temp_path = f"{self.file_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(temp_path, self.file_path)

    def get_tasks(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return self._read_unlocked()

    def upsert_task(self, task_id: str, data: Dict[str, Any]) -> None:
        with self._lock:
            tasks = self._read_unlocked()
            tasks[task_id] = data
            self._write_unlocked(tasks)

    def mutate(self, fn: Callable[[Dict[str, Dict[str, Any]]], Any]) -> Any:
        with self._lock:
            tasks = self._read_unlocked()
            result = fn(tasks)
            self._write_unlocked(tasks)
            return result

    def next_numbered_id(
        self,
        prefix: str = "任务",
        exclude_prefixes: Tuple[str, ...] = ("批量任务",),
    ) -> str:
        with self._lock:
            tasks = self._read_unlocked()
            nums: list[int] = []
            for key in tasks.keys():
                if any(key.startswith(ex) for ex in exclude_prefixes if ex != prefix):
                    continue
                if key.startswith(prefix):
                    suffix = key[len(prefix) :]
                    if suffix.isdigit():
                        nums.append(int(suffix))
            return f"{prefix}{max(nums, default=0) + 1}"

    @staticmethod
    def is_expired(
        submitted_at_str: str,
        ttl_seconds: int = 7200,
        now: Optional[datetime] = None,
    ) -> bool:
        """Check if a timestamp string '%Y-%m-%d %H:%M:%S' is older than ttl_seconds (default 2h)."""
        if not submitted_at_str:
            return False
        try:
            submitted = datetime.strptime(submitted_at_str.strip(), "%Y-%m-%d %H:%M:%S")
            ref_now = now or datetime.now()
            return (ref_now - submitted).total_seconds() > ttl_seconds
        except ValueError:
            return False
