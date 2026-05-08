from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixel_pipeline import PatchTokenizer, PixelPairDataset, TextRenderer, build_pair_table


DEFAULT_ROOT = Path(r"E:/Document/ML/introml-project-dkd-2526-2")
DEFAULT_FONT = DEFAULT_ROOT / "font-times-new-roman" / "font-times-new-roman" / "times.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--patch-width", type=int, default=4)
    parser.add_argument("--limit-tokenizer-rows", type=int, default=0)
    parser.add_argument(
        "--tokenizer-out",
        type=Path,
        default=Path("artifacts/patch_tokenizer_w4.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.root / "train.csv"
    test_path = args.root / "test.csv"
    train = pd.read_csv(train_path, encoding="utf-8")
    test = pd.read_csv(test_path, encoding="utf-8")

    renderer = TextRenderer(args.font)
    tokenizer = PatchTokenizer(patch_width=args.patch_width)

    train_pairs = build_pair_table(train)
    test_pairs = build_pair_table(test)
    print(f"train rows={len(train)} pair records={len(train_pairs)} pairs={len(train_pairs)//2}")
    print(f"test rows={len(test)} pair records={len(test_pairs)} pairs={len(test_pairs)//2}")

    fit_df = train if args.limit_tokenizer_rows <= 0 else train.head(args.limit_tokenizer_rows)
    for row in fit_df.itertuples(index=False):
        target_img = renderer.render_target(row.target, int(row.max_width))
        tokenizer.fit_image(target_img)
    tokenizer.save(args.tokenizer_out)
    print(f"patch_width={args.patch_width} vocab_size={tokenizer.vocab_size}")
    print(f"saved tokenizer={args.tokenizer_out.resolve()}")

    row = train.iloc[0]
    context_img = renderer.render_context(row["context"])
    target_img = renderer.render_target(row["target"], int(row["max_width"]))
    ids = tokenizer.encode_image(target_img)
    decoded = tokenizer.decode_ids(ids, width=int(row["max_width"]))
    mismatch = int(np.abs((target_img > 128).astype(np.int16) - (decoded > 128).astype(np.int16)).sum())
    print(
        "sample0",
        f"context_shape={context_img.shape}",
        f"target_shape={target_img.shape}",
        f"target_tokens={len(ids)}",
        f"roundtrip_mismatch_pixels={mismatch}",
    )

    pixels = renderer.text_to_pixels(row["target"], int(row["id"]), int(row["max_width"]))
    print(
        "pixels",
        f"count={len(pixels)}",
        f"sample_id_minmax=({pixels['sample_id'].min()},{pixels['sample_id'].max()})",
        f"row_minmax=({pixels['row_id'].min()},{pixels['row_id'].max()})",
        f"col_minmax=({pixels['col_id'].min()},{pixels['col_id'].max()})",
    )

    dataset = PixelPairDataset(train, renderer=renderer, tokenizer=tokenizer)
    item0 = dataset[0]
    item1 = dataset[1]
    print(
        "dataset[0]",
        f"id={item0['sample_id']}",
        f"lang={item0['language']}",
        f"paired={item0['paired_language']}",
        f"ctx={item0['context_image'].shape}",
        f"pair_ctx={item0['paired_context_image'].shape}",
        f"target_ids={len(item0['target_ids'])}",
    )
    print(
        "dataset[1]",
        f"id={item1['sample_id']}",
        f"lang={item1['language']}",
        f"paired={item1['paired_language']}",
        f"ctx={item1['context_image'].shape}",
        f"pair_ctx={item1['paired_context_image'].shape}",
        f"target_ids={len(item1['target_ids'])}",
    )

    try:
        from pixel_pipeline import collate_pixel_pairs

        batch = collate_pixel_pairs([item0, item1])
        print(
            "torch batch",
            f"context_images={tuple(batch['context_images'].shape)}",
            f"paired_context_images={tuple(batch['paired_context_images'].shape)}",
            f"decoder_input_ids={tuple(batch['decoder_input_ids'].shape)}",
        )
    except ModuleNotFoundError as exc:
        print(f"torch batch skipped: {exc}")


if __name__ == "__main__":
    main()
