"""Labelled (image, phrase, mask) datasets for the adaptation contract: the digest-pinned FoodSeg103 sample, the
record contract and its structural validation, image-disjoint splitting, and the BYOD loader.

A record is ``{id, image, prompt, mask}`` where ``image`` is a PIL image (sides within the pipeline's ceilings),
``prompt`` the phrase the mask answers (normalised like ``format_prompts`` normalises a query) and ``mask`` a
boolean array (or a Pillow-decodable image whose non-zero pixels are the mask) of the image's height × width
with at least one true pixel.

The default sample is drawn from FoodSeg103 (Wu et al. 2021, LARC-CMU-SMU; **Apache-2.0**; a curated sample of
Recipe1M food photographs with pixel-wise ingredient masks) as converted to parquet by the Hugging Face Hub at an
immutable revision: the first ``CORPUS_ROW_GROUPS`` row groups of the validation shard are read with HTTPS range
requests (about 5 MB each; the shard's declared size is checked first and every row group's content is refused
unless its SHA-256 matches the pin). Each image yields one record: the ingredient class that covers the most
pixels becomes the phrase (its FoodSeg103 name, e.g. ``bread``, ``chicken duck``, ``steak``) and that class's
pixels the mask; images whose largest class is ``background`` or ``other ingredients`` are skipped (none of the
800 are). The domain gap to the model's PhraseCut training distribution is the point of the sample: the frozen
model must already know what the phrase means, and the adapter can only sharpen where it draws the boundary.
"""
# ruff: noqa: E501  -- record and pin literals are kept on single lines

from __future__ import annotations

import csv
import hashlib
import io
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .pipeline import MODEL_ID, format_prompts, validate_image

CORPUS_NAME = "FoodSeg103 (validation split), first eight parquet row groups"
CORPUS_REPO = "EduardoPacheco/FoodSeg103"
CORPUS_REVISION = "176acc3edd2432ee126bda6fb01469eadeb018df"  # refs/convert/parquet commit on the Hub
CORPUS_FILE = "default/validation/0000.parquet"
CORPUS_BYTES = 115_186_173
CORPUS_ROWS = 2_135
CORPUS_ROW_GROUPS = 8  # of 22; 100 images each
CORPUS_LICENSE = "Apache-2.0 (FoodSeg103, LARC-CMU-SMU; Wu, Fu, Liu, Lim, Hoi, Sun, ACM MM 2021; images curated from Recipe1M)"
CORPUS_URL = f"https://huggingface.co/datasets/{CORPUS_REPO}/resolve/{CORPUS_REVISION}/{CORPUS_FILE}"
# SHA-256 over the concatenated image bytes + label-mask bytes + UTF-8 id of each row, in row order, and that byte total.
ROW_GROUP_PINS: dict[int, tuple[str, int]] = {
    0: ("931827f16c33a0548ad26c456c8495fc3949c79f7950869eabaff44f748e8f78", 5_090_593),
    1: ("a784bb0f23e413eef8d0dad90e504f398bde1a44d2110dced06bdea8953f08bf", 5_207_382),
    2: ("f8a8e15d28c6678d344324b3c972008e642878b25ebff9f481196707c39fc32e", 5_467_905),
    3: ("adebbbc59740e87c5eb0578f029174872bd95b7fa5b1178c708e61c766b73fa5", 6_071_430),
    4: ("fd181120d4966d20e47742232e4102d45247b3f14cf7e2a8733b9d08d1aa152d", 4_758_783),
    5: ("c3af4dda9af15abf54d5f4494fd07df925d57c1d9ec654eb30a6c1bd30f50b61", 5_020_081),
    6: ("0597cb610e859eb400c746dfba37abc446afd34d897b573b1c68453c5c9d80d9", 4_784_703),
    7: ("9e9aac90865e09124fa6c3e89cd676ef2dcd65f87532c002574a545671bae4a9", 5_899_238),
}
DEFAULT_CACHE_DIR = Path("weights") / "foodseg103"

# The FoodSeg103 class vocabulary (dataset card, ids 0..103); 0 = background and 103 = other ingredients never become a phrase.
FOODSEG103_CLASSES = ("background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage", "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra", "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom", "salad", "other ingredients")
EXCLUDED_CLASS_IDS = (0, 103)

SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 600, "validation": 60, "test": 140}  # of the 800 images the eight row groups hold
SAMPLE_DIGEST = "d8123852b1f1c8dd7054c77ab36cf408738f48c7902678109d620165c2a6d4bf"  # dataset_digest over the three default splits together; tests pin it
MIN_RECORDS = 8
MAX_RECORDS = 5_000
MIN_MASK_PIXELS = 1
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _HttpRangeFile(io.RawIOBase):
    """A seekable read-only view of one HTTPS object served with `Range` requests (what `pyarrow` needs to read a
    parquet footer and a few row groups without downloading the file)."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos = url, size, 0
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "clipseg-segmentation-pipeline"})
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 (pinned https URL)
            if response.status != 206:
                raise ValueError(f"{self.url}: server ignored the Range request (HTTP {response.status})")
            data = response.read()
        self.fetched += len(data)
        self.pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _declared_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "clipseg-segmentation-pipeline"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 (pinned https URL)
        length = response.headers.get("Content-Length")
    if length is None:
        raise ValueError(f"{url}: no Content-Length in the HEAD response")
    return int(length)


def _group_digest(rows: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    digest, total = hashlib.sha256(), 0
    for row in rows:
        for chunk in (row["image"]["bytes"], row["label"]["bytes"], str(row["id"]).encode("utf-8")):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def fetch_corpus(
    *, cache_dir: str | Path | None = None, groups: Sequence[int] | None = None, opener: Any = None
) -> dict[int, list[dict[str, Any]]]:
    """Return the pinned row groups as lists of `{image, label, id}` (JPEG/PNG bytes, label-mask PNG bytes, source
    id), from the cache (one parquet file per row group) or the Hub (footer + the row groups it needs, over range
    requests). Every row group's decoded content is refused unless its SHA-256 and byte total match
    `ROW_GROUP_PINS`; a fresh fetch also checks the shard's declared size and row count."""
    import pyarrow.parquet as pq

    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    wanted = list(groups) if groups is not None else sorted(ROW_GROUP_PINS)
    out: dict[int, list[dict[str, Any]]] = {}
    reader = None
    for group in wanted:
        if group not in ROW_GROUP_PINS:
            raise ValueError(f"row group {group} has no pin; pinned groups are {sorted(ROW_GROUP_PINS)}")
        local = cache / f"validation-rg{group}.parquet"
        rows: list[dict[str, Any]] | None = None
        if local.is_file():
            rows = pq.read_table(local).to_pylist()
            if _group_digest(rows) != ROW_GROUP_PINS[group]:
                rows = None  # stale or corrupt cache: refetch
        if rows is None:
            if reader is None:
                if opener is not None:
                    reader = pq.ParquetFile(opener(CORPUS_URL))
                else:
                    declared = _declared_size(CORPUS_URL)
                    if declared != CORPUS_BYTES:
                        raise ValueError(f"{CORPUS_FILE}: declared size {declared} != pinned {CORPUS_BYTES}")
                    reader = pq.ParquetFile(io.BufferedReader(_HttpRangeFile(CORPUS_URL, CORPUS_BYTES), buffer_size=1 << 20))
                if reader.metadata.num_rows != CORPUS_ROWS:
                    raise ValueError(f"{CORPUS_FILE}: {reader.metadata.num_rows} rows, pinned {CORPUS_ROWS}")
            table = reader.read_row_group(group, columns=["image", "label", "id"])
            rows = table.to_pylist()
            digest, total = _group_digest(rows)
            if (digest, total) != ROW_GROUP_PINS[group]:
                raise ValueError(f"{CORPUS_FILE} row group {group}: sha256 {digest} / {total} bytes != pinned {ROW_GROUP_PINS[group]}")
            pq.write_table(table, local)
        out[group] = [{"image": r["image"]["bytes"], "label": r["label"]["bytes"], "id": int(r["id"])} for r in rows]
    return out


def largest_class(label: np.ndarray) -> int | None:
    """The FoodSeg103 class id covering the most pixels, background and 'other ingredients' excluded; None if no
    other class is present."""
    ids, counts = np.unique(np.asarray(label), return_counts=True)
    candidates = [(int(c), int(i)) for i, c in zip(ids, counts, strict=True) if int(i) not in EXCLUDED_CLASS_IDS and 0 <= int(i) < len(FOODSEG103_CLASSES)]
    if not candidates:
        return None
    return max(candidates)[1]


def read_corpus(groups: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Decode the verified row groups into records: one per image, the largest ingredient class as the phrase and
    its pixels as the mask (images with no ingredient class besides background / other are skipped)."""
    out = []
    for group in sorted(groups):
        for index, row in enumerate(groups[group]):
            label = np.asarray(Image.open(io.BytesIO(row["label"])))
            class_id = largest_class(label)
            if class_id is None:
                continue
            image = Image.open(io.BytesIO(row["image"]))
            image.load()
            out.append(
                {
                    "id": f"foodseg103-val-{group * 100 + index}",
                    "image": image.convert("RGB"),
                    "prompt": FOODSEG103_CLASSES[class_id],
                    "mask": label == class_id,
                    "class_id": class_id,
                    "source_id": row["id"],
                    "source_row_group": group,
                }
            )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]], *, seed: int = SAMPLE_SEED, sizes: Mapping[str, int] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Seeded image-level draw: shuffle the records and cut `sizes` (train / validation / test) in order."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    pool = [dict(r) for r in records]
    random.Random(seed).shuffle(pool)
    needed = sum(sizes.values())
    if len(pool) < needed:
        raise ValueError(f"only {len(pool)} records available, need {needed}")
    out, cursor = {}, 0
    for name, count in sizes.items():
        out[name] = pool[cursor : cursor + count]
        cursor += count
    return out


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, seed: int = SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    return build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cache_dir)), seed=seed)


# ---------------------------------------------------------------------------------------------------------
# Record contract
# ---------------------------------------------------------------------------------------------------------


def _open(image: Any, where: str) -> Image.Image:
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{where}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{where}: must be a PIL.Image.Image or a file path")
    return image


def coerce_mask(mask: Any, size: tuple[int, int], where: str = "mask") -> np.ndarray:
    """A boolean height × width array from a boolean/integer array or a Pillow-decodable image (non-zero = true)."""
    if isinstance(mask, str | Path):
        mask = _open(mask, where)
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L")) > 0
    else:
        try:
            array = np.asarray(mask)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{where}: must be a boolean array or an image") from exc
        if array.dtype != bool:
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"{where}: must be a boolean or numeric array")
            array = array != 0
    width, height = size
    if array.ndim != 2 or array.shape != (height, width):
        raise ValueError(f"{where}: shape {array.shape} does not match the image's height x width {(height, width)}")
    if int(array.sum()) < MIN_MASK_PIXELS:
        raise ValueError(f"{where}: the mask has no true pixel")
    return array


def _check_record(record: Any, index: int) -> dict[str, Any]:
    where = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{where} must be a mapping with id/image/prompt/mask")
    for key in ("id", "image", "prompt", "mask"):
        if key not in record:
            raise ValueError(f"{where} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{where}: id must match {_ID_RE.pattern}")
    try:
        image = validate_image(_open(record["image"], f"{where}.image"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc
    if not isinstance(record["prompt"], str):
        raise ValueError(f"{where}: prompt must be a str")
    try:
        prompt = format_prompts([record["prompt"]])[0]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc
    mask = coerce_mask(record["mask"], image.size, f"{where}.mask")
    item = {"id": rid, "image": image, "prompt": prompt, "mask": mask}
    for key in ("class_id", "source_id", "source_row_group"):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a labelled-mask dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, prompt, mask} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked, ids = [], set()
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        checked.append(item)
    areas = [float(r["mask"].mean()) for r in checked]
    widths = [r["image"].width for r in checked]
    heights = [r["image"].height for r in checked]
    prompts = sorted({r["prompt"] for r in checked})
    return {
        "records": checked,
        "n_records": len(checked),
        "n_prompts": len(prompts),
        "prompts": prompts,
        "mask_area_fraction": {"min": min(areas), "max": max(areas), "mean": sum(areas) / len(areas)},
        "image_width": {"min": min(widths), "max": max(widths)},
        "image_height": {"min": min(heights), "max": max(heights)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size-prefixed) — the identity a split is made disjoint on."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes())


def mask_digest(mask: np.ndarray) -> str:
    array = np.asarray(mask, dtype=bool)
    return _sha256_bytes(f"{array.shape[1]}x{array.shape[0]}:".encode() + np.packbits(array).tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent SHA-256 over (id, image digest, prompt, mask digest)."""
    parts = sorted(f"{r['id']}:{image_digest(r['image'])}:{r['prompt']}:{mask_digest(r['mask'])}" for r in records)
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no image (by decoded-pixel digest) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["image"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]], *, val_fraction: float = 0.15, test_fraction: float = 0.2, seed: int = 0
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test after de-duplicating images."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    random.Random(seed).shuffle(unique)
    n = len(unique)
    n_test = max(1, round(n * test_fraction))
    n_val = round(n * val_fraction)
    if n - n_test - n_val < 1:
        raise ValueError(f"{n} distinct images are too few to split into train/validation/test")
    return {"test": unique[:n_test], "validation": unique[n_test : n_test + n_val], "train": unique[n_test + n_val :]}


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Records from a directory or zip holding images, mask images and a `masks.csv` with the columns `file`,
    `mask` and `prompt` (and optionally `id`); every listed file must exist and every image file must be listed
    (mask files are those a `mask` column names)."""
    source = Path(path)
    members: dict[str, bytes] = {}
    if source.is_dir():
        for file in sorted(source.rglob("*")):
            if file.is_file():
                members[file.name] = file.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    members[Path(info.filename).name] = archive.read(info)  # flattened; no extractall
    else:
        raise ValueError(f"{source} is neither a directory nor a zip file")
    if "masks.csv" not in members:
        raise ValueError("BYOD data must include masks.csv with the columns file, mask and prompt")
    rows = list(csv.DictReader(io.StringIO(members["masks.csv"].decode("utf-8-sig"))))
    if not rows or any(column not in rows[0] for column in ("file", "mask", "prompt")):
        raise ValueError("masks.csv must have the columns file, mask and prompt")
    out = []
    for row in rows:
        name = Path(str(row.get("file", "")).strip()).name
        mask_name = Path(str(row.get("mask", "")).strip()).name
        for needed in (name, mask_name):
            if needed not in members:
                raise ValueError(f"masks.csv names a missing file: {needed}")
        try:
            image = Image.open(io.BytesIO(members[name]))
            image.load()
            mask = Image.open(io.BytesIO(members[mask_name]))
            mask.load()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"BYOD file is not a decodable image: {name} / {mask_name}") from exc
        rid = str(row.get("id", "") or "").strip()
        out.append({"id": rid or re.sub(r"[^A-Za-z0-9_.:-]", "_", Path(name).stem)[:64], "image": image.convert("RGB"), "prompt": str(row.get("prompt", "")), "mask": mask})
    listed = {Path(str(r.get(k, "")).strip()).name for r in rows for k in ("file", "mask")}
    unlisted = [n for n in members if n != "masks.csv" and n not in listed]
    if unlisted:
        raise ValueError(f"{len(unlisted)} file(s) have no masks.csv row, e.g. {unlisted[0]}")
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """A summary table (id, image size, prompt, mask area fraction, provenance) in the BYOD `masks.csv` column
    layout plus extras (`file`/`mask` name the id; the images themselves are not written)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "file", "mask", "prompt", "width", "height", "mask_area_fraction", "class_id", "source_id", "source_row_group"])
        for r in records:
            writer.writerow([r["id"], f"{r['id']}.jpg", f"{r['id']}_mask.png", r["prompt"], r["image"].width, r["image"].height, round(float(np.asarray(r["mask"], dtype=bool).mean()), 4), r.get("class_id", ""), r.get("source_id", ""), r.get("source_row_group", "")])
    return out
