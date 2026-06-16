from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .utils import ensure_dir


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


class RunLogger:
    def __init__(self, path: Path | None = None, *, verbose: bool = True) -> None:
        self.path = path
        self.verbose = verbose
        self._handle: TextIO | None = None
        if path:
            ensure_dir(path.parent)
            self._handle = path.open("a", encoding="utf-8")

    def __call__(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        if self.verbose:
            print(line, flush=True)
        if self._handle:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class PhaseTimer:
    logger: RunLogger
    name: str

    def __enter__(self) -> "PhaseTimer":
        self.start = time.perf_counter()
        self.logger(f"INICIO {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self.start
        status = "ERROR" if exc_type else "FIN"
        self.logger(f"{status} {self.name} ({format_seconds(elapsed)})")

