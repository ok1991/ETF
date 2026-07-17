"""Filesystem layout for source, runtime state, artifacts, and public output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    runtime: Path
    data: Path
    state: Path
    logs: Path
    artifacts: Path
    calibration: Path
    public: Path
    web: Path

    @classmethod
    def discover(cls) -> "RuntimePaths":
        root = Path(__file__).resolve().parents[1]
        runtime = root / ".runtime"
        artifacts = root / "artifacts"
        return cls(
            root=root,
            runtime=runtime,
            data=runtime / "data",
            state=runtime / "state",
            logs=runtime / "logs",
            artifacts=artifacts,
            calibration=artifacts / "calibration",
            public=root / "public",
            web=root / "web",
        )

    def ensure(self) -> None:
        for path in (self.data, self.state, self.logs, self.calibration, self.public):
            path.mkdir(parents=True, exist_ok=True)


PATHS = RuntimePaths.discover()
