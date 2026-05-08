from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .patch_tokenizer import PatchTokenizer
from .rendering import TextRenderer


LANGUAGE_TO_ID = {
    "English": 0,
    "French": 1,
    "German": 2,
    "Italian": 3,
    "Spanish": 4,
}


@dataclass(frozen=True)
class PairRecord:
    pair_id: int
    sample_id: int
    paired_sample_id: int
    row_index: int
    paired_row_index: int
    language: str
    paired_language: str


def build_pair_table(df: pd.DataFrame) -> dict[int, PairRecord]:
    if len(df) % 2 != 0:
        raise ValueError(f"Expected even number of rows, got {len(df)}")

    table: dict[int, PairRecord] = {}
    sorted_df = df.sort_values("id").reset_index(drop=True)
    for i in range(0, len(sorted_df), 2):
        left = sorted_df.iloc[i]
        right = sorted_df.iloc[i + 1]
        if int(right["id"]) != int(left["id"]) + 1:
            raise ValueError(f"Broken consecutive IDs at rows {i}/{i+1}")
        if left["language"] == "English" or right["language"] != "English":
            raise ValueError(
                f"Expected non-English then English at rows {i}/{i+1}, "
                f"got {left['language']} then {right['language']}"
            )
        pair_id = i // 2
        left_id = int(left["id"])
        right_id = int(right["id"])
        table[left_id] = PairRecord(
            pair_id=pair_id,
            sample_id=left_id,
            paired_sample_id=right_id,
            row_index=i,
            paired_row_index=i + 1,
            language=str(left["language"]),
            paired_language=str(right["language"]),
        )
        table[right_id] = PairRecord(
            pair_id=pair_id,
            sample_id=right_id,
            paired_sample_id=left_id,
            row_index=i + 1,
            paired_row_index=i,
            language=str(right["language"]),
            paired_language=str(left["language"]),
        )
    return table


class PixelPairDataset:
    def __init__(
        self,
        df: pd.DataFrame | str | Path,
        renderer: TextRenderer,
        tokenizer: PatchTokenizer | None = None,
        use_pair_context: bool = True,
        require_target: bool | None = None,
    ) -> None:
        if isinstance(df, (str, Path)):
            df = pd.read_csv(df, encoding="utf-8")
        self.df = df.sort_values("id").reset_index(drop=True).copy()
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.use_pair_context = use_pair_context
        self.has_target = "target" in self.df.columns
        if require_target is True and not self.has_target:
            raise ValueError("Dataset requires target column, but none was found")
        if require_target is False and self.has_target:
            pass
        self.pair_table = build_pair_table(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[int(index)]
        sample_id = int(row["id"])
        pair = self.pair_table[sample_id]
        paired_row = self.df.iloc[pair.paired_row_index]

        context_image = self.renderer.render_context(row["context"], binary=False)
        paired_context_image = self.renderer.render_context(
            paired_row["context"], binary=False
        )
        if not self.use_pair_context:
            paired_context_image = np.zeros((self.renderer.image_height, 1), dtype=np.uint8)

        item = {
            "sample_id": sample_id,
            "pair_id": pair.pair_id,
            "language": str(row["language"]),
            "language_id": LANGUAGE_TO_ID[str(row["language"])],
            "paired_language": str(paired_row["language"]),
            "max_width": int(row["max_width"]),
            "context_image": context_image,
            "paired_context_image": paired_context_image,
            "context_width": int(context_image.shape[1]),
            "paired_context_width": int(paired_context_image.shape[1]),
        }

        if self.has_target:
            target_image = self.renderer.render_target(
                row["target"], max_width=int(row["max_width"]), binary=False
            )
            item["target_image"] = target_image
            item["target_width"] = int(target_image.shape[1])
            if self.tokenizer is not None:
                item["target_ids"] = self.tokenizer.encode_image(target_image)
                item["target_patch_count"] = int(len(item["target_ids"]))

        return item


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for collate_pixel_pairs. Install torch on the 4090 server."
        ) from exc
    return torch


def _pad_images(torch, images: list[np.ndarray]) -> tuple:
    heights = {img.shape[0] for img in images}
    if len(heights) != 1:
        raise ValueError(f"All images must have same height, got {heights}")
    max_width = max(int(img.shape[1]) for img in images)
    batch = torch.zeros((len(images), 1, images[0].shape[0], max_width), dtype=torch.float32)
    widths = torch.zeros((len(images),), dtype=torch.long)
    for i, img in enumerate(images):
        width = int(img.shape[1])
        widths[i] = width
        arr = torch.from_numpy((img > 128).astype(np.float32))
        batch[i, 0, :, :width] = arr
    return batch, widths


def collate_pixel_pairs(batch: list[dict]) -> dict:
    torch = _require_torch()
    context_images, context_widths = _pad_images(
        torch, [item["context_image"] for item in batch]
    )
    paired_images, paired_widths = _pad_images(
        torch, [item["paired_context_image"] for item in batch]
    )

    out = {
        "sample_ids": torch.tensor([item["sample_id"] for item in batch], dtype=torch.long),
        "pair_ids": torch.tensor([item["pair_id"] for item in batch], dtype=torch.long),
        "language_ids": torch.tensor([item["language_id"] for item in batch], dtype=torch.long),
        "max_widths": torch.tensor([item["max_width"] for item in batch], dtype=torch.long),
        "context_images": context_images,
        "paired_context_images": paired_images,
        "context_widths": context_widths,
        "paired_context_widths": paired_widths,
        "languages": [item["language"] for item in batch],
        "paired_languages": [item["paired_language"] for item in batch],
    }

    if "target_ids" in batch[0]:
        pad_id = 0
        bos_id = 1
        max_len = max(len(item["target_ids"]) for item in batch)
        decoder_input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        target_lengths = torch.zeros((len(batch),), dtype=torch.long)
        for i, item in enumerate(batch):
            ids = torch.as_tensor(item["target_ids"], dtype=torch.long)
            target_lengths[i] = len(ids)
            decoder_input_ids[i, 0] = bos_id
            if len(ids) > 1:
                decoder_input_ids[i, 1 : len(ids)] = ids[:-1]
            labels[i, : len(ids)] = ids
        out["decoder_input_ids"] = decoder_input_ids
        out["labels"] = labels
        out["target_lengths"] = target_lengths
        out["target_widths"] = torch.tensor(
            [item["target_width"] for item in batch], dtype=torch.long
        )

    return out


def _to_pix2struct_pil(image: np.ndarray, invert: bool = True) -> Image.Image:
    arr = ((image > 128).astype(np.uint8) * 255)
    if invert:
        arr = 255 - arr
    return Image.fromarray(arr, mode="L").convert("RGB")


class Pix2StructPairCollator:
    def __init__(
        self,
        model_name: str = "google/pix2struct-base",
        max_patches: int = 1024,
        invert_images: bool = True,
        keep_raw_images: bool = False,
    ) -> None:
        try:
            from transformers import AutoImageProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Pix2StructPairCollator requires transformers. Install with: "
                "pip install transformers accelerate sentencepiece protobuf"
            ) from exc

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.max_patches = int(max_patches)
        self.invert_images = bool(invert_images)
        self.keep_raw_images = bool(keep_raw_images)

    def _process_images(self, images: list[np.ndarray]) -> dict:
        pil_images = [_to_pix2struct_pil(img, invert=self.invert_images) for img in images]
        encoded = self.image_processor(
            images=pil_images,
            return_tensors="pt",
            max_patches=self.max_patches,
        )
        return {
            "flattened_patches": encoded["flattened_patches"],
            "attention_mask": encoded["attention_mask"].bool(),
        }

    def __call__(self, batch: list[dict]) -> dict:
        base = collate_pixel_pairs(batch)
        context_inputs = self._process_images([item["context_image"] for item in batch])
        paired_inputs = self._process_images(
            [item["paired_context_image"] for item in batch]
        )

        base["pix_context_flattened_patches"] = context_inputs["flattened_patches"]
        base["pix_context_attention_mask"] = context_inputs["attention_mask"]
        base["pix_paired_flattened_patches"] = paired_inputs["flattened_patches"]
        base["pix_paired_attention_mask"] = paired_inputs["attention_mask"]

        if not self.keep_raw_images:
            base.pop("context_images", None)
            base.pop("paired_context_images", None)
            base.pop("context_widths", None)
            base.pop("paired_context_widths", None)
        return base
