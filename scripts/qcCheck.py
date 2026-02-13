from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

INPUT_CSV = Path("data/labels/labels.csv")
OUTPUT_CSV = Path("data/labels/labels_qc.csv")
RAW_ROOT = Path("data/raw")

MIN_PIXELS = 1_000_000
TOO_DARK = 0.15
TOO_BRIGHT = 0.90


def compute_exposure(img_rgb: np.ndarray) -> float:
    return float(img_rgb.mean() / 255.0)


def try_find_image(image_id: str, class_label: str, filepath_value: str | None) -> Path | None:
    if isinstance(filepath_value, str) and filepath_value.strip():
        p = Path(filepath_value.strip())
        if p.exists():
            return p
        if not p.is_absolute():
            p2 = Path(".") / p
            if p2.exists():
                return p2

    if isinstance(class_label, str) and class_label.strip():
        class_folder = RAW_ROOT / class_label.strip()
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = class_folder / f"{image_id}{ext}"
            if candidate.exists():
                return candidate

    for match in RAW_ROOT.rglob(f"{image_id}.*"):
        if match.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            return match

    return None


def qc_one_image(path: Path):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            width, height = im.size
            arr = np.array(im)

        exposure = compute_exposure(arr)

        qc_status = "completed"
        notes = []

        if width * height < MIN_PIXELS:
            qc_status = "working"
            notes.append(f"low_resolution({width}x{height})")

        if exposure < TOO_DARK:
            qc_status = "working"
            notes.append("too_dark")

        if exposure > TOO_BRIGHT:
            qc_status = "working"
            notes.append("too_bright")

        qc_notes = ";".join(notes) if notes else "ok"
        return width, height, exposure, qc_status, qc_notes

    except Exception as e:
        return None, None, None, "failed", f"unreadable:{type(e).__name__}"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Could not find {INPUT_CSV.as_posix()}")

    df = pd.read_csv(INPUT_CSV)
    df.columns = [c.strip() for c in df.columns]

    required = ["imageID", "class", "hawkID", "filepath", "qcStatus", "qcNotes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    for col in ["width", "height", "exposureScore"]:
        if col not in df.columns:
            df[col] = np.nan

    new_filepaths = []
    widths = []
    heights = []
    exposures = []
    statuses = []
    notes_list = []

    for _, row in df.iterrows():
        image_id = str(row["imageID"]).strip()
        class_label = str(row["class"]).strip()
        filepath_val = row["filepath"]

        found_path = try_find_image(image_id, class_label, filepath_val)

        if found_path is None:
            new_filepaths.append(str(filepath_val) if pd.notna(filepath_val) else "")
            widths.append(None)
            heights.append(None)
            exposures.append(None)
            statuses.append("failed")
            notes_list.append("file_not_found_in_repo")
            continue

        try:
            rel_path = found_path.relative_to(Path("."))
            new_filepaths.append(rel_path.as_posix())
        except Exception:
            new_filepaths.append(found_path.as_posix())

        w, h, exp, qc_status, qc_notes = qc_one_image(found_path)
        widths.append(w)
        heights.append(h)
        exposures.append(exp)
        statuses.append(qc_status)
        notes_list.append(qc_notes)

    df["filepath"] = new_filepaths
    df["width"] = widths
    df["height"] = heights
    df["exposureScore"] = exposures
    df["qcStatus"] = statuses
    df["qcNotes"] = notes_list

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote QC results to: {OUTPUT_CSV.as_posix()}")
    print(df[["imageID", "class", "filepath", "qcStatus", "qcNotes", "exposureScore"]].head())


if __name__ == "__main__":
    main()
