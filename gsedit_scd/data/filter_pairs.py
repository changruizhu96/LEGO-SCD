from __future__ import annotations
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from collections import Counter, defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path.cwd()  # repository root: run the tools from the repository root

from gsedit_scd.data.filter_edits import load_mask, split_boundary_ignore

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Keep only complete nuisance-matched semantic-event episodes whose "
            "positive/ignore masks satisfy the boundary-aware artifact gate."
        )
    )
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--boundary-ignore-radius", type=int, default=5)
    parser.add_argument("--maximum-far-ignore-positive-ratio", type=float, default=1.0)
    parser.add_argument(
        "--expected-conditions",
        nargs="+",
        default=("identity", "relighting", "pose", "relighting_pose"),
    )
    return parser.parse_args()


def resolve_row_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main():
    args = parse_args()
    if args.output_list.exists() or args.output_audit.exists():
        raise FileExistsError("Output list or audit already exists")
    if args.boundary_ignore_radius < 0:
        raise ValueError("--boundary-ignore-radius must be non-negative")
    if args.maximum_far_ignore_positive_ratio < 0.0:
        raise ValueError(
            "--maximum-far-ignore-positive-ratio must be non-negative"
        )
    expected_conditions = set(args.expected_conditions)
    if not expected_conditions:
        raise ValueError("--expected-conditions must not be empty")

    with args.input_list.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    if not rows:
        raise ValueError("Input list contains no rows")

    groups = defaultdict(list)
    def audit_row(row):
        positive = load_mask(resolve_row_path(row["positive_mask"]))
        ignore = load_mask(resolve_row_path(row["ignore_mask"]))
        near, far = split_boundary_ignore(
            positive,
            ignore,
            args.boundary_ignore_radius,
        )
        positive_pixels = int(positive.sum())
        far_ratio = int(far.sum()) / max(positive_pixels, 1)
        individual_pass = (
            positive_pixels > 0
            and far_ratio <= args.maximum_far_ignore_positive_ratio
        )
        return {
            **row,
            "positive_pixels": positive_pixels,
            "yellow_pixels": int(ignore.sum()),
            "near_yellow_pixels": int(near.sum()),
            "far_yellow_pixels": int(far.sum()),
            "far_yellow_positive_ratio": far_ratio,
            "individual_pass": individual_pass,
        }

    with ThreadPoolExecutor(max_workers=4) as reader:
        audits = list(reader.map(audit_row, rows))
    for index, row in enumerate(rows):
        groups[(row["object_id"], row["camera_name"], row["event"])].append(index)

    kept_indices = set()
    kept_episodes = 0
    for indices in groups.values():
        conditions = [rows[index]["condition"] for index in indices]
        complete = (
            set(conditions) == expected_conditions
            and len(conditions) == len(expected_conditions)
        )
        passed = complete and all(audits[index]["individual_pass"] for index in indices)
        if passed:
            kept_indices.update(indices)
            kept_episodes += 1
        for index in indices:
            audits[index]["complete_episode"] = complete
            audits[index]["status"] = "train" if passed else "excluded"
            if passed:
                reason = ""
            elif not complete:
                reason = "incomplete_or_duplicate_conditions"
            elif not audits[index]["individual_pass"]:
                reason = "far_yellow_gt_relative_limit"
            else:
                reason = "paired_condition_failed"
            audits[index]["filter_reason"] = reason

    output_rows = [row for index, row in enumerate(rows) if index in kept_indices]
    args.output_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    with args.output_list.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    with args.output_audit.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(audits[0]))
        writer.writeheader()
        writer.writerows(audits)

    print(json.dumps({
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "input_episodes": len(groups),
        "output_episodes": kept_episodes,
        "kept_object_count": len({row["object_id"] for row in output_rows}),
        "condition_counts": dict(Counter(row["condition"] for row in output_rows)),
        "output_list": str(args.output_list),
        "output_audit": str(args.output_audit),
    }, indent=2))


if __name__ == "__main__":
    main()
