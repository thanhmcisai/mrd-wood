#!/usr/bin/env python3
"""
Report species/class-name overlap between WoodID datasets.

The primary source is the dataset folder structure used by WoodDataset:
<dataset>/<species_name>/<images>. For VN26, each magnification subset is treated
as a dataset but can also be collapsed by normalized species name.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from wood_spatial.config import ALL_DATASETS, V4_CSV


DEFAULT_DATASETS = [
    "WRD25", "DTSR14", "PCA11",
    "BFS46", "FSDM41", "GOIMAI", "WOODAUTH", "BD11",
    "VN26_x10", "VN26_x20", "VN26_x50",
]


def normalize_species(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[_\\/-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s.strip()


def species_from_folder(root: Path) -> list[str]:
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            out.append(d.name)
    return out


def collect_species(datasets: list[str]) -> dict[str, pd.DataFrame]:
    result = {}
    for ds in datasets:
        info = ALL_DATASETS.get(ds)
        if not info:
            continue
        root = Path(info["root"])
        names = species_from_folder(root)
        result[ds] = pd.DataFrame({
            "dataset": ds,
            "species_raw": names,
            "species_norm": [normalize_species(x) for x in names],
        })
    return result


def pairwise_overlap(species: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    detail = []
    datasets = list(species)
    for a in datasets:
        set_a = set(species[a]["species_norm"])
        raw_a = dict(zip(species[a]["species_norm"], species[a]["species_raw"]))
        for b in datasets:
            set_b = set(species[b]["species_norm"])
            raw_b = dict(zip(species[b]["species_norm"], species[b]["species_raw"]))
            inter = sorted(set_a & set_b)
            union = set_a | set_b
            rows.append({
                "dataset_a": a,
                "dataset_b": b,
                "n_a": len(set_a),
                "n_b": len(set_b),
                "n_overlap": len(inter),
                "jaccard": (len(inter) / len(union)) if union else 0.0,
            })
            for s in inter:
                detail.append({
                    "dataset_a": a,
                    "dataset_b": b,
                    "species_norm": s,
                    "species_a": raw_a.get(s, ""),
                    "species_b": raw_b.get(s, ""),
                })
    return pd.DataFrame(rows), pd.DataFrame(detail)


def run(datasets: list[str], out_dir: Path, save: bool = True):
    species = collect_species(datasets)
    species_rows = pd.concat(species.values(), ignore_index=True) if species else pd.DataFrame()
    summary, detail = pairwise_overlap(species)
    matrix = summary.pivot(index="dataset_a", columns="dataset_b", values="n_overlap")
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        species_rows.to_csv(out_dir / "species_by_dataset.csv", index=False)
        summary.to_csv(out_dir / "species_overlap_pairwise.csv", index=False)
        detail.to_csv(out_dir / "species_overlap_detail.csv", index=False)
        matrix.to_csv(out_dir / "species_overlap_matrix.csv")
    return species_rows, summary, detail, matrix


def main():
    parser = argparse.ArgumentParser(description="Check species-name overlap between datasets.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--out-dir", default=str(V4_CSV))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    species_rows, summary, detail, matrix = run(args.datasets, Path(args.out_dir), save=not args.no_save)
    print("\n=== Species counts ===")
    if len(species_rows):
        print(species_rows.groupby("dataset")["species_norm"].nunique().to_string())
    else:
        print("No species folders found. Check WOOD_DATASETS_DIR / dataset paths.")
    print("\n=== Overlap matrix (counts) ===")
    print(matrix.fillna(0).astype(int).to_string())
    nonself = summary[summary["dataset_a"] < summary["dataset_b"]].sort_values(
        ["n_overlap", "jaccard"], ascending=False
    )
    print("\n=== Non-zero pair overlaps ===")
    nz = nonself[nonself["n_overlap"] > 0]
    print(nz.to_string(index=False) if len(nz) else "No exact normalized-name overlaps.")
    if not args.no_save:
        print(f"\nSaved overlap CSVs to {Path(args.out_dir)}")


if __name__ == "__main__":
    main()
