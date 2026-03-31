"""Tests for dataset loaders in the data pipeline."""

from __future__ import annotations

import json

import pytest

from spatialvlm.data.datasets import (
    R2RCEDataset,
    RxRCEDataset,
    SQA3DDataset,
    _read_json_or_jsonl,
    build_dataset,
    iter_instructions,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_jsonl_records(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "go left", "episode_id": "1"}),
                json.dumps({"instruction": "go right", "episode_id": "2"}),
            ]
        ),
        encoding="utf-8",
    )
    rows = _read_json_or_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["instruction"] == "go left"


def test_read_json_episodes_structure(tmp_path):
    path = tmp_path / "episodes.json"
    _write_json(
        path,
        {
            "episodes": [
                {"instruction": "walk to chair", "episode_id": "e1"},
                {"instruction": "turn around", "episode_id": "e2"},
            ]
        },
    )
    rows = _read_json_or_jsonl(path)
    assert len(rows) == 2
    assert rows[1]["episode_id"] == "e2"


def test_r2r_dataset_split_and_limit(tmp_path):
    path = tmp_path / "r2r.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "a", "episode_id": "1", "split": "train"}),
                json.dumps({"instruction": "b", "episode_id": "2", "split": "val"}),
                json.dumps({"instruction": "c", "episode_id": "3", "split": "train"}),
            ]
        ),
        encoding="utf-8",
    )
    ds = R2RCEDataset.from_file(path, split="train", limit=1)
    assert len(ds) == 1
    sample = ds[0]
    assert sample.source == "r2r-ce"
    assert sample.instruction == "a"


def test_dataset_accepts_instruction_lists(tmp_path):
    path = tmp_path / "rxr.json"
    _write_json(
        path,
        [{"instructions": ["first instruction", "fallback"], "episode_id": "42", "split": "train"}],
    )
    ds = RxRCEDataset.from_file(path, split="train")
    assert ds[0].instruction == "first instruction"


def test_missing_instruction_raises(tmp_path):
    path = tmp_path / "broken.json"
    _write_json(path, [{"episode_id": "1"}])
    ds = SQA3DDataset.from_file(path)
    with pytest.raises(ValueError, match="Missing instruction"):
        _ = ds[0]


def test_build_dataset_aliases_and_iter_instructions(tmp_path):
    path = tmp_path / "sqa3d.json"
    _write_json(path, [{"instruction": "where is the sofa?", "id": "x"}])

    ds = build_dataset("vln-r2r", path)
    assert isinstance(ds, R2RCEDataset)

    ds2 = build_dataset("sqa3d", path)
    instructions = list(iter_instructions(ds2))
    assert instructions == ["where is the sofa?"]


def test_build_dataset_unsupported_name(tmp_path):
    path = tmp_path / "data.json"
    _write_json(path, [{"instruction": "x"}])
    with pytest.raises(ValueError, match="Unsupported dataset name"):
        build_dataset("unknown-dataset", path)
