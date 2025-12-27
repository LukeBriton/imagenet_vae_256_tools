from __future__ import annotations
from pathlib import Path
import json
import numpy as np

def export_filtered_repacked(
    feat_dir: str,
    lab_dir: str,
    out_dir: str,
    target_classes: list[int] | None = None,   # e.g. [264, 609, 835] or None for all
    keep_prob: float | None = None,            # e.g. 0.1 (approx keep 10%)
    max_per_class: int | None = None,          # e.g. 500
    seed: int = 42,
    shuffle_files: bool = True,
    out_batch: int = 256,                      # output shard size (choose 32/64/128/256/512)
    remap_labels: bool = False,                # remap selected classes -> 0..K-1
    write_provenance_jsonl: bool = True,       # provenance can get large if you keep lots of samples
):
    feat_dir = Path(feat_dir)
    lab_dir  = Path(lab_dir)
    out_dir  = Path(out_dir)

    out_feat = out_dir / "imagenet256_features"
    out_lab  = out_dir / "imagenet256_labels"
    out_feat.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)  # reproducible RNG :contentReference[oaicite:2]{index=2}
    target_set = None if target_classes is None else set(map(int, target_classes))

    # Optional label remapping (useful if you keep only a subset of classes)
    label_map = None
    if remap_labels:
        if target_set is None:
            raise ValueError("remap_labels=True requires target_classes to be provided.")
        sorted_cls = sorted(target_set)
        label_map = {c: i for i, c in enumerate(sorted_cls)}

    # Deterministic file list + optional seeded shuffle
    files = sorted(feat_dir.glob("*.npy"), key=lambda p: p.name)
    if shuffle_files:
        rng.shuffle(files)  # deterministic due to seed :contentReference[oaicite:3]{index=3}

    counts: dict[int, int] = {}   # selected sample counts by *original* label
    buf_x: list[np.ndarray] = []
    buf_y: list[int] = []
    out_idx = 0
    total_selected = 0

    # provenance: one line per output shard, to avoid gigantic single JSON
    prov_f = None
    if write_provenance_jsonl:
        prov_f = (out_dir / "provenance.jsonl").open("w", encoding="utf-8")

    def flush():
        """Write one output shard from current buffers."""
        nonlocal out_idx, buf_x, buf_y, total_selected

        if not buf_y:
            return

        X = np.stack(buf_x, axis=0).astype(np.float32, copy=False)  # (B,4,32,32)
        Y = np.asarray(buf_y, dtype=np.int64)                       # (B,)

        stem = f"{out_idx:06d}_0"  # rename here
        np.save(out_feat / f"{stem}.npy", X)  # np.save writes .npy format :contentReference[oaicite:4]{index=4}
        np.save(out_lab  / f"{stem}.npy", Y)

        total_selected += len(buf_y)
        out_idx += 1
        buf_x.clear()
        buf_y.clear()

    # To record provenance per output shard we accumulate sources for *current* buffer.
    buf_src: list[tuple[str, int]] = []  # (source_stem, row)
    def flush_with_prov():
        nonlocal buf_src
        if not buf_y:
            return
        if prov_f is not None:
            line = {
                "out_stem": f"{out_idx:06d}_0",
                "sources": [{"stem": s, "row": int(r)} for (s, r) in buf_src],
            }
            prov_f.write(json.dumps(line) + "\n")
        buf_src.clear()
        flush()

    for f in files:
        stem = f.stem
        y_path = lab_dir / f"{stem}.npy"
        if not y_path.exists():
            raise FileNotFoundError(f"Missing label file for {stem}: {y_path}")

        # memmap read is efficient for big datasets :contentReference[oaicite:5]{index=5}
        x = np.load(f, mmap_mode="r", allow_pickle=False)       # (32,4,32,32)
        y = np.load(y_path, mmap_mode="r", allow_pickle=False)  # (32,)

        order = rng.permutation(len(y))  # deterministic within-file order :contentReference[oaicite:6]{index=6}
        for j in order:
            lbl = int(y[j])

            if target_set is not None and lbl not in target_set:
                continue

            if keep_prob is not None and rng.random() > float(keep_prob):
                continue

            if max_per_class is not None and counts.get(lbl, 0) >= int(max_per_class):
                continue

            # accepted
            counts[lbl] = counts.get(lbl, 0) + 1

            out_lbl = lbl if label_map is None else label_map[lbl]
            buf_y.append(out_lbl)

            # copy sample out of memmap
            buf_x.append(np.array(x[j], copy=True))

            if prov_f is not None:
                buf_src.append((stem, int(j)))

            if len(buf_y) >= out_batch:
                flush_with_prov()

    flush_with_prov()

    if prov_f is not None:
        prov_f.close()

    manifest = {
        "seed": seed,
        "target_classes": sorted(list(target_set)) if target_set is not None else None,
        "keep_prob": keep_prob,
        "max_per_class": max_per_class,
        "shuffle_files": shuffle_files,
        "out_batch": out_batch,
        "remap_labels": remap_labels,
        "label_map": label_map,  # None unless remap_labels=True
        "total_selected_samples": int(sum(counts.values())),
        "total_output_shards": int(out_idx),
        "counts_by_class": {str(k): int(v) for k, v in sorted(counts.items())},
        "output_features_dir": str(out_feat),
        "output_labels_dir": str(out_lab),
        "provenance_jsonl": "provenance.jsonl" if write_provenance_jsonl else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Done. seed={seed}, selected {manifest['total_selected_samples']} samples "
          f"-> wrote {manifest['total_output_shards']} shard(s).")
    print("Example kept counts:", sorted(counts.items())[:10])


if __name__ == "__main__":
    export_filtered_repacked(
        feat_dir="./imagenet_feature/imagenet256_features",
        lab_dir="./imagenet_feature/imagenet256_labels",
        out_dir="shapenet_overlapped_feature",
        target_classes=[402, 404, 412, 413, 414, 423, 435, 440, 444, 448, 453, 466, 468, 472, 484, 487, 495, 504, 508, 510, 511, 520, 528, 532, 534, 546, 547, 553, 554, 559, 564, 576, 579, 609, 619, 620, 625, 627, 632, 636, 637, 648, 650, 654, 659, 664, 665, 670, 671, 703, 705, 707, 717, 721, 728, 732, 736, 737, 742, 748, 751, 759, 761, 763, 764, 765, 779, 780, 782, 790, 809, 814, 817, 820, 827, 829, 831, 833, 846, 851, 871, 878, 881, 894, 895, 897, 898, 900, 907, 914],
        keep_prob=None,                  # e.g. 0.1
        max_per_class=1500,             # large cap means “keep all available”
        seed=42,
        shuffle_files=True,
        out_batch=32,
        remap_labels=False,              # set True if you want labels 0..K-1
        write_provenance_jsonl=True,
    )

# Done. seed=42, selected 116632 samples -> wrote 456 shard(s).