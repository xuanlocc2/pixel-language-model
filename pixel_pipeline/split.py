from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairSplit:
    train_pair_ids: list[int]
    val_pair_ids: list[int]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train_pair_ids": self.train_pair_ids,
            "val_pair_ids": self.val_pair_ids,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PairSplit":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            train_pair_ids=[int(x) for x in payload["train_pair_ids"]],
            val_pair_ids=[int(x) for x in payload["val_pair_ids"]],
        )


def add_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values("id").reset_index(drop=True).copy()
    if len(sorted_df) % 2 != 0:
        raise ValueError(f"Expected even number of rows, got {len(sorted_df)}")
    pair_ids = []
    for i in range(0, len(sorted_df), 2):
        left = sorted_df.iloc[i]
        right = sorted_df.iloc[i + 1]
        if int(right["id"]) != int(left["id"]) + 1:
            raise ValueError(f"Broken pair IDs at sorted rows {i}/{i + 1}")
        if left["language"] == "English" or right["language"] != "English":
            raise ValueError(
                f"Expected non-English then English at sorted rows {i}/{i + 1}, "
                f"got {left['language']} then {right['language']}"
            )
        pair_id = i // 2
        pair_ids.extend([pair_id, pair_id])
    sorted_df["pair_id"] = pair_ids
    return sorted_df


def make_pair_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_train_pairs: int | None = None,
    max_val_pairs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, PairSplit]:
    with_pairs = add_pair_id(df)
    pair_ids = np.asarray(sorted(with_pairs["pair_id"].unique()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    shuffled = pair_ids.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(pair_ids) * val_ratio)))
    val_ids = sorted(int(x) for x in shuffled[:n_val])
    train_ids = sorted(int(x) for x in shuffled[n_val:])

    if max_val_pairs is not None:
        val_ids = val_ids[: int(max_val_pairs)]
    if max_train_pairs is not None:
        train_ids = train_ids[: int(max_train_pairs)]

    train_df = with_pairs[with_pairs["pair_id"].isin(train_ids)].drop(columns=["pair_id"])
    val_df = with_pairs[with_pairs["pair_id"].isin(val_ids)].drop(columns=["pair_id"])
    train_df = train_df.sort_values("id").reset_index(drop=True)
    val_df = val_df.sort_values("id").reset_index(drop=True)
    return train_df, val_df, PairSplit(train_pair_ids=train_ids, val_pair_ids=val_ids)
