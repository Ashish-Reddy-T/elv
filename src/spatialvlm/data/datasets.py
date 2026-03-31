"""Dataset loaders for navigation and spatial QA corpora."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records from either JSON array/object or JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL rows must be JSON objects.")
            records.append(row)
        return records

    parsed = json.loads(text)
    if isinstance(parsed, list):
        if not all(isinstance(x, dict) for x in parsed):
            raise ValueError("JSON list dataset must contain JSON objects.")
        return parsed
    if isinstance(parsed, dict):
        # Common pattern: {"episodes": [...]}
        if "episodes" in parsed and isinstance(parsed["episodes"], list):
            episodes = parsed["episodes"]
            if not all(isinstance(x, dict) for x in episodes):
                raise ValueError("'episodes' must contain JSON objects.")
            return episodes
        return [parsed]

    raise ValueError("Unsupported dataset JSON structure.")


@dataclass(frozen=True)
class NavSample:
    """Standardized navigation sample."""

    instruction: str
    episode_id: str
    source: str
    payload: dict[str, Any]


def _pick_instruction(record: dict[str, Any]) -> str:
    for key in ("instruction", "instructions", "command", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item
    raise ValueError("Missing instruction text in record.")


def _pick_episode_id(record: dict[str, Any], idx: int) -> str:
    value = record.get("episode_id", record.get("id", record.get("path_id", idx)))
    return str(value)


class _BaseNavDataset(Dataset[NavSample]):
    source_name: str = "base"

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        split: str | None = None,
        limit: int | None = None,
    ) -> None:
        rows = list(records)
        if split is not None:
            rows = [r for r in rows if str(r.get("split", "")).lower() == split.lower()]
        if limit is not None:
            rows = rows[:limit]
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> NavSample:
        row = self._rows[idx]
        return NavSample(
            instruction=_pick_instruction(row),
            episode_id=_pick_episode_id(row, idx),
            source=self.source_name,
            payload=row,
        )


class R2RCEDataset(_BaseNavDataset):
    """R2R-CE loader."""

    source_name = "r2r-ce"

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        split: str | None = None,
        limit: int | None = None,
    ) -> R2RCEDataset:
        return cls(_read_json_or_jsonl(Path(path)), split=split, limit=limit)


class RxRCEDataset(_BaseNavDataset):
    """RxR-CE loader."""

    source_name = "rxr-ce"

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        split: str | None = None,
        limit: int | None = None,
    ) -> RxRCEDataset:
        return cls(_read_json_or_jsonl(Path(path)), split=split, limit=limit)


class SQA3DDataset(_BaseNavDataset):
    """SQA3D loader.

    Keeps same NavSample shape for compatibility with shared collators.
    """

    source_name = "sqa3d"

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        split: str | None = None,
        limit: int | None = None,
    ) -> SQA3DDataset:
        return cls(_read_json_or_jsonl(Path(path)), split=split, limit=limit)


def build_dataset(
    name: str,
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
) -> Dataset[NavSample]:
    """Factory for supported benchmark datasets."""
    key = name.strip().lower()
    if key in {"r2r", "r2r-ce", "vln-r2r"}:
        return R2RCEDataset.from_file(path, split=split, limit=limit)
    if key in {"rxr", "rxr-ce", "vln-rxr"}:
        return RxRCEDataset.from_file(path, split=split, limit=limit)
    if key in {"sqa3d"}:
        return SQA3DDataset.from_file(path, split=split, limit=limit)
    raise ValueError(f"Unsupported dataset name: {name}")


def iter_instructions(dataset: Iterable[NavSample]) -> Iterable[str]:
    """Yield instruction strings from standardized samples."""
    for sample in dataset:
        yield sample.instruction
