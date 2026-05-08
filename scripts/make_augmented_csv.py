from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixel_pipeline import TextRenderer


DEFAULT_ROOT = Path(r"E:/Document/ML/introml-project-dkd-2526-2")
DEFAULT_FONT = DEFAULT_ROOT / "font-times-new-roman" / "font-times-new-roman" / "times.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--out-csv", type=Path, default=Path("artifacts/train_augmented.csv"))
    parser.add_argument("--augment-per-pair", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-prefix-ratio", type=float, default=0.65)
    parser.add_argument("--max-prefix-ratio", type=float, default=0.88)
    parser.add_argument("--min-prefix-words", type=int, default=8)
    parser.add_argument("--min-target-words", type=int, default=3)
    parser.add_argument("--target-width-margin", type=int, default=4)
    parser.add_argument("--max-target-width", type=int, default=1800)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--include-original", action="store_true")
    return parser.parse_args()


def split_full_text(
    full_text: str,
    ratio: float,
    min_prefix_words: int,
    min_target_words: int,
) -> tuple[str, str] | None:
    words = str(full_text).split()
    if len(words) < min_prefix_words + min_target_words:
        return None
    split_index = int(round(len(words) * ratio))
    split_index = max(min_prefix_words, split_index)
    split_index = min(len(words) - min_target_words, split_index)
    context = " ".join(words[:split_index]).strip()
    target = " ".join(words[split_index:]).strip()
    if not context or not target:
        return None
    return context, target


def target_width(renderer: TextRenderer, text: str, margin: int) -> int:
    width, _ = renderer.text_size(text)
    return max(1, int(width) + int(margin))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    renderer = TextRenderer(args.font)
    train = pd.read_csv(args.root / "train.csv", encoding="utf-8").sort_values("id").reset_index(drop=True)
    if args.max_pairs is not None:
        train = train.head(int(args.max_pairs) * 2)

    rows = []
    next_id = 0
    if args.include_original:
        for row in train.itertuples(index=False):
            rows.append(
                {
                    "id": next_id,
                    "language": row.language,
                    "context": row.context,
                    "target": row.target,
                    "max_width": int(row.max_width),
                }
            )
            next_id += 1

    accepted = 0
    skipped = 0
    for i in range(0, len(train), 2):
        pair_rows = [train.iloc[i], train.iloc[i + 1]]
        for _ in range(args.augment_per_pair):
            ratio = rng.uniform(args.min_prefix_ratio, args.max_prefix_ratio)
            candidate_rows = []
            ok = True
            for source in pair_rows:
                full = f"{source['context']} {source['target']}"
                split = split_full_text(
                    full,
                    ratio=ratio,
                    min_prefix_words=args.min_prefix_words,
                    min_target_words=args.min_target_words,
                )
                if split is None:
                    ok = False
                    break
                context, target = split
                width = target_width(renderer, target, args.target_width_margin)
                if width > args.max_target_width:
                    ok = False
                    break
                candidate_rows.append(
                    {
                        "language": source["language"],
                        "context": context,
                        "target": target,
                        "max_width": width,
                    }
                )
            if not ok:
                skipped += 1
                continue
            for candidate in candidate_rows:
                candidate["id"] = next_id
                rows.append(candidate)
                next_id += 1
            accepted += 1

    out = pd.DataFrame(rows, columns=["id", "language", "context", "target", "max_width"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(
        f"saved={args.out_csv.resolve()} rows={len(out)} "
        f"pairs={len(out)//2} accepted_aug_pairs={accepted} skipped_aug_pairs={skipped}"
    )


if __name__ == "__main__":
    main()
