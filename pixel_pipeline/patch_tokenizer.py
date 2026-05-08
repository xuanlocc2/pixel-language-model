from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


def _binary_patch_key(patch: np.ndarray, threshold: int) -> bytes:
    bits = (patch > threshold).astype(np.uint8).reshape(-1)
    return np.packbits(bits, bitorder="big").tobytes()


@dataclass
class PatchTokenizer:
    patch_width: int = 4
    image_height: int = 32
    threshold: int = 128
    token_to_key: dict[int, bytes] = field(default_factory=dict)
    key_to_token: dict[bytes, int] = field(default_factory=dict)

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKENS["<pad>"]

    @property
    def bos_id(self) -> int:
        return SPECIAL_TOKENS["<bos>"]

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKENS["<eos>"]

    @property
    def unk_id(self) -> int:
        return SPECIAL_TOKENS["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(SPECIAL_TOKENS) + len(self.key_to_token)

    def _next_id(self) -> int:
        return len(SPECIAL_TOKENS) + len(self.key_to_token)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError(f"Expected grayscale image [H,W], got shape={image.shape}")
        if image.shape[0] != self.image_height:
            raise ValueError(f"Expected height {self.image_height}, got {image.shape[0]}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image

    def iter_patch_keys(self, image: np.ndarray) -> Iterable[bytes]:
        image = self._normalize_image(image)
        width = image.shape[1]
        n_patches = int(np.ceil(width / self.patch_width))
        padded_width = n_patches * self.patch_width
        if padded_width != width:
            padded = np.zeros((self.image_height, padded_width), dtype=np.uint8)
            padded[:, :width] = image
            image = padded

        for start in range(0, padded_width, self.patch_width):
            patch = image[:, start : start + self.patch_width]
            yield _binary_patch_key(patch, self.threshold)

    def fit_image(self, image: np.ndarray) -> None:
        for key in self.iter_patch_keys(image):
            if key not in self.key_to_token:
                token_id = self._next_id()
                self.key_to_token[key] = token_id
                self.token_to_key[token_id] = key

    def fit_images(self, images: Iterable[np.ndarray]) -> None:
        for image in images:
            self.fit_image(image)

    def encode_image(
        self,
        image: np.ndarray,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> np.ndarray:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for key in self.iter_patch_keys(image):
            ids.append(self.key_to_token.get(key, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return np.asarray(ids, dtype=np.int64)

    def decode_ids(
        self,
        token_ids: Iterable[int],
        width: int | None = None,
        skip_special: bool = True,
    ) -> np.ndarray:
        patches: list[np.ndarray] = []
        blank = np.zeros((self.image_height, self.patch_width), dtype=np.uint8)
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in (self.pad_id, self.bos_id, self.eos_id):
                if skip_special:
                    continue
                patches.append(blank.copy())
                continue
            key = self.token_to_key.get(token_id)
            if key is None:
                patches.append(blank.copy())
                continue
            bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8), bitorder="big")
            patch = bits[: self.image_height * self.patch_width].reshape(
                self.image_height, self.patch_width
            )
            patches.append((patch * 255).astype(np.uint8))

        if not patches:
            img = np.zeros((self.image_height, 1), dtype=np.uint8)
        else:
            img = np.concatenate(patches, axis=1)
        if width is not None:
            width = int(width)
            if img.shape[1] < width:
                padded = np.zeros((self.image_height, width), dtype=np.uint8)
                padded[:, : img.shape[1]] = img
                img = padded
            else:
                img = img[:, :width]
        return img

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "patch_width": self.patch_width,
            "image_height": self.image_height,
            "threshold": self.threshold,
            "keys": [
                {
                    "token_id": token_id,
                    "key_b64": base64.b64encode(key).decode("ascii"),
                }
                for token_id, key in sorted(self.token_to_key.items())
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PatchTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(
            patch_width=int(payload["patch_width"]),
            image_height=int(payload["image_height"]),
            threshold=int(payload["threshold"]),
        )
        for item in payload["keys"]:
            token_id = int(item["token_id"])
            key = base64.b64decode(item["key_b64"])
            tokenizer.token_to_key[token_id] = key
            tokenizer.key_to_token[key] = token_id
        return tokenizer
