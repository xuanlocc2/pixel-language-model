from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from PIL import ImageFont


ROOT = Path(r"E:/Document/ML/introml-project-dkd-2526-2")
TRAIN = ROOT / "train.csv"
TEST = ROOT / "test.csv"
FONT = ROOT / "font-times-new-roman" / "font-times-new-roman" / "times.ttf"


def qstats(values: pd.Series | np.ndarray) -> str:
    s = pd.Series(values).astype(float)
    qs = s.quantile([0, 0.25, 0.5, 0.75, 1.0])
    return (
        f"min={qs.loc[0]:.0f}, p25={qs.loc[0.25]:.0f}, "
        f"median={qs.loc[0.5]:.0f}, mean={s.mean():.1f}, "
        f"p75={qs.loc[0.75]:.0f}, max={qs.loc[1.0]:.0f}"
    )


def wc(text: str) -> int:
    return len(str(text).split())


def date_in(text: str) -> str | None:
    m = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", str(text))
    return m.group(0) if m else None


def edge_category(ch: str) -> str:
    if ch.isalpha():
        return "letter"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    return ch


def analyze_pairs(df: pd.DataFrame, name: str) -> None:
    broken = []
    pair_langs: Counter[tuple[str, str]] = Counter()
    same_dates = 0
    for i in range(0, len(df), 2):
        if i + 1 >= len(df):
            broken.append((i, "dangling"))
            continue
        a = df.iloc[i]
        b = df.iloc[i + 1]
        pair_langs[(a["language"], b["language"])] += 1
        ok = a["language"] != "English" and b["language"] == "English"
        ok = ok and int(b["id"]) == int(a["id"]) + 1
        if date_in(a["context"]) == date_in(b["context"]):
            same_dates += 1
        if not ok:
            broken.append((int(a["id"]), a["language"], int(b["id"]), b["language"]))
    print(f"\n[{name}] consecutive pair analysis")
    print(f"pairs={len(df)//2}, broken={len(broken)}, same_date_pairs={same_dates}/{len(df)//2}")
    print("pair language counts:", dict(pair_langs))
    if broken[:5]:
        print("first broken examples:", broken[:5])


def analyze_lengths(df: pd.DataFrame, name: str, has_target: bool) -> None:
    print(f"\n[{name}] lengths")
    c_chars = df["context"].str.len()
    c_words = df["context"].map(wc)
    print("context chars:", qstats(c_chars))
    print("context words:", qstats(c_words))
    print("max_width:", qstats(df["max_width"]))
    if has_target:
        t_chars = df["target"].str.len()
        t_words = df["target"].map(wc)
        full_chars = c_chars + 1 + t_chars
        full_words = c_words + t_words
        c_ratio = c_chars / full_chars
        t_ratio = t_chars / full_chars
        print("target chars:", qstats(t_chars))
        print("target words:", qstats(t_words))
        print("full paragraph chars:", qstats(full_chars))
        print("context/full char ratio:", qstats(c_ratio * 100))
        print("target/full char ratio:", qstats(t_ratio * 100))
        print("context/full word ratio:", qstats((c_words / full_words) * 100))
        print("leading/trailing whitespace in context:", int(df["context"].str.match(r"^\s").sum()), int(df["context"].str.match(r".*\s$").sum()))
        print("leading/trailing whitespace in target:", int(df["target"].str.match(r"^\s").sum()), int(df["target"].str.match(r".*\s$").sum()))
        print("target starts:", dict(Counter(edge_category(str(x)[0]) for x in df["target"] if str(x))))
        print("context ends:", dict(Counter(edge_category(str(x)[-1]) for x in df["context"] if str(x))))
        print("target final punctuation:", dict(Counter(str(x)[-1] for x in df["target"] if str(x))))


def analyze_render_widths(train: pd.DataFrame) -> None:
    font = ImageFont.truetype(str(FONT), 24)
    widths = []
    heights = []
    clipped = []
    for _, row in train.iterrows():
        bbox = font.getbbox(str(row["target"]))
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        widths.append(width)
        heights.append(height)
        clipped.append(width + 10 > int(row["max_width"]))
    widths = pd.Series(widths)
    heights = pd.Series(heights)
    width_delta = train["max_width"].astype(int) - widths
    margin_delta = train["max_width"].astype(int) - (widths + 10)
    print("\n[train] render geometry for target with times.ttf size 24")
    print("target bbox width:", qstats(widths))
    print("target bbox height:", qstats(heights))
    print("max_width - bbox_width:", qstats(width_delta))
    print("max_width - (bbox_width + 10 margin):", qstats(margin_delta))
    print(f"bbox wider than max_width: {int((width_delta < 0).sum())}/{len(train)}")
    print(f"10px right margin capped by render_only.py: {sum(clipped)}/{len(train)}")
    tight = width_delta.abs().le(3).sum()
    print(f"near exact glyph width (+/-3 px): {tight}/{len(train)}")
    worst = pd.DataFrame(
        {
            "id": train["id"],
            "language": train["language"],
            "max_width": train["max_width"],
            "bbox_width": widths,
            "delta": width_delta,
            "target": train["target"],
        }
    ).sort_values("delta").head(5)
    print("smallest deltas:")
    for _, r in worst.iterrows():
        print(f"  id={int(r.id)} lang={r.language} delta={int(r.delta)} maxw={int(r.max_width)} bbox={int(r.bbox_width)} target={r.target[:90]!r}")


def analyze_dates(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_dates = train["context"].map(date_in)
    test_dates = test["context"].map(date_in)
    print("\n[dates]")
    print(f"train rows with date={train_dates.notna().sum()}/{len(train)}, unique_dates={train_dates.nunique()}")
    print(f"test rows with date={test_dates.notna().sum()}/{len(test)}, unique_dates={test_dates.nunique()}")
    overlap = sorted(set(train_dates.dropna()) & set(test_dates.dropna()))
    print(f"date overlap train/test={len(overlap)}")
    if overlap[:10]:
        print("first overlapping dates:", overlap[:10])


def rough_similarity(train: pd.DataFrame, test: pd.DataFrame) -> None:
    en_train = train[train["language"] == "English"].copy()
    en_test = test[test["language"] == "English"].copy()

    def tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    train_tokens = [(int(r.id), tokens(r.context)) for r in en_train.itertuples(index=False)]
    rows = []
    for r in en_test.itertuples(index=False):
        tt = tokens(r.context)
        best = (0.0, None)
        for tid, trt in train_tokens:
            union = len(tt | trt)
            score = len(tt & trt) / union if union else 0.0
            if score > best[0]:
                best = (score, tid)
        rows.append(best[0])
    print("\n[English context nearest-neighbor lexical overlap test->train]")
    print("best Jaccard:", qstats(pd.Series(rows) * 100))
    print(f"test English rows with best overlap >= 70%: {sum(x >= 0.70 for x in rows)}/{len(rows)}")


def print_examples(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\n[examples]")
    for name, df in [("train", train), ("test", test)]:
        print(f"{name} first two pairs:")
        for i in range(0, 4, 2):
            a, b = df.iloc[i], df.iloc[i + 1]
            target_a = f" | target={a['target'][:90]!r}" if "target" in df.columns else ""
            target_b = f" | target={b['target'][:90]!r}" if "target" in df.columns else ""
            print(f"  id={int(a.id)} {a.language}: context={a.context[:115]!r}{target_a}")
            print(f"  id={int(b.id)} {b.language}: context={b.context[:115]!r}{target_b}")


def main() -> None:
    train = pd.read_csv(TRAIN, encoding="utf-8")
    test = pd.read_csv(TEST, encoding="utf-8")
    print("[files]")
    print("train shape:", train.shape, "columns:", list(train.columns))
    print("test shape:", test.shape, "columns:", list(test.columns))
    print("train id range:", int(train["id"].min()), int(train["id"].max()), "contiguous:", train["id"].tolist() == list(range(int(train["id"].min()), int(train["id"].max()) + 1)))
    print("test id range:", int(test["id"].min()), int(test["id"].max()), "contiguous:", test["id"].tolist() == list(range(int(test["id"].min()), int(test["id"].max()) + 1)))
    print("train nulls:", train.isna().sum().to_dict())
    print("test nulls:", test.isna().sum().to_dict())
    print("train language counts:", train["language"].value_counts().to_dict())
    print("test language counts:", test["language"].value_counts().to_dict())
    print("encoding sanity sample:", train.loc[0, "context"][:120])
    analyze_pairs(train, "train")
    analyze_pairs(test, "test")
    analyze_lengths(train, "train", True)
    analyze_lengths(test, "test", False)
    analyze_render_widths(train)
    analyze_dates(train, test)
    rough_similarity(train, test)
    print_examples(train, test)


if __name__ == "__main__":
    main()
