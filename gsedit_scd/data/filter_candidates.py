from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reject candidates whose interventions are not persistently clean, "
            "then collapse candidates that claim the same Gaussian support."
        )
    )
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--event-audit", type=Path, required=True)
    parser.add_argument("--event-manifest", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--minimum-passing-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-passing-event-types", type=int, default=2)
    parser.add_argument("--duplicate-containment", type=float, default=0.9)
    parser.add_argument("--duplicate-jaccard", type=float, default=0.6)
    return parser.parse_args()


def duplicate_groups(candidate_ids, gaussian_sets, containment_limit, jaccard_limit):
    adjacency = defaultdict(set)
    edges = []
    candidate_ids = sorted(candidate_ids)
    for index, left in enumerate(candidate_ids):
        for right in candidate_ids[index + 1:]:
            intersection = len(gaussian_sets[left] & gaussian_sets[right])
            if not intersection:
                continue
            minimum = min(len(gaussian_sets[left]), len(gaussian_sets[right]))
            union = len(gaussian_sets[left] | gaussian_sets[right])
            containment = intersection / max(minimum, 1)
            jaccard = intersection / max(union, 1)
            if containment >= containment_limit or jaccard >= jaccard_limit:
                adjacency[left].add(right)
                adjacency[right].add(left)
                edges.append({
                    "left": left,
                    "right": right,
                    "intersection": intersection,
                    "containment": containment,
                    "jaccard": jaccard,
                })

    seen = set()
    groups = []
    for candidate_id in candidate_ids:
        if candidate_id in seen or candidate_id not in adjacency:
            continue
        stack = [candidate_id]
        seen.add(candidate_id)
        group = []
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        groups.append(sorted(group))
    return groups, edges


def main():
    args = parse_args()
    if args.output_list.exists() or args.output_summary.exists():
        raise FileExistsError("Output list or summary already exists")
    if not 0.0 <= args.minimum_passing_fraction <= 1.0:
        raise ValueError("--minimum-passing-fraction must lie in [0, 1]")
    if args.minimum_passing_event_types <= 0:
        raise ValueError("--minimum-passing-event-types must be positive")
    for name, value in (
        ("--duplicate-containment", args.duplicate_containment),
        ("--duplicate-jaccard", args.duplicate_jaccard),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")

    with args.input_list.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    with args.event_audit.open(newline="", encoding="utf-8") as handle:
        event_rows = list(csv.DictReader(handle))
    if not rows or not event_rows:
        raise ValueError("Input list and event audit must not be empty")

    audit_by_object = defaultdict(list)
    for row in event_rows:
        audit_by_object[row["object_id"]].append(row)
    input_object_ids = sorted({row["object_id"] for row in rows})
    statistics = {}
    consistent = []
    for object_id in input_object_ids:
        object_rows = audit_by_object[object_id]
        if not object_rows:
            raise ValueError(f"Missing event audit rows for {object_id}")
        passing_rows = [row for row in object_rows if row["status"] == "train"]
        passing_event_types = sorted({row["event"] for row in passing_rows})
        passing_fraction = len(passing_rows) / len(object_rows)
        is_consistent = (
            passing_fraction > args.minimum_passing_fraction
            and len(passing_event_types) >= args.minimum_passing_event_types
        )
        statistics[object_id] = {
            "event_rows": len(object_rows),
            "passing_event_rows": len(passing_rows),
            "passing_fraction": passing_fraction,
            "passing_event_types": passing_event_types,
            "consistent": is_consistent,
        }
        if is_consistent:
            consistent.append(object_id)

    manifest = json.loads(args.event_manifest.read_text())
    latent_map = torch.load(
        manifest["object_map"], map_location="cpu", weights_only=False
    )
    components = {
        int(component["component_id"]): component
        for component in latent_map["components"]
    }
    objects = {row["object_id"]: row for row in manifest["objects"]}
    gaussian_sets = {
        object_id: set(
            components[int(objects[object_id]["anchor_component_id"])][
                "gaussian_ids"
            ].tolist()
        )
        for object_id in consistent
    }
    groups, edges = duplicate_groups(
        consistent,
        gaussian_sets,
        args.duplicate_containment,
        args.duplicate_jaccard,
    )

    rejected_duplicates = set()
    group_records = []
    for group in groups:
        representative = max(
            group,
            key=lambda object_id: (
                statistics[object_id]["passing_fraction"],
                statistics[object_id]["passing_event_rows"],
                int(objects[object_id]["majority_owned_gaussian_count"]),
                object_id,
            ),
        )
        rejected = sorted(set(group) - {representative})
        rejected_duplicates.update(rejected)
        group_records.append({
            "members": group,
            "representative": representative,
            "rejected": rejected,
        })

    selected = sorted(set(consistent) - rejected_duplicates)
    selected_set = set(selected)
    output_rows = [row for row in rows if row["object_id"] in selected_set]
    args.output_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_list.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "protocol": {
            "minimum_passing_fraction_exclusive": args.minimum_passing_fraction,
            "minimum_passing_event_types": args.minimum_passing_event_types,
            "duplicate_containment": args.duplicate_containment,
            "duplicate_jaccard": args.duplicate_jaccard,
            "selection_order": (
                "passing_fraction, passing_rows, majority_owned_gaussian_count, "
                "object_id"
            ),
            "target_gt_used": False,
        },
        "input_object_count": len(input_object_ids),
        "consistent_object_count": len(consistent),
        "selected_object_count": len(selected),
        "input_row_count": len(rows),
        "selected_row_count": len(output_rows),
        "consistent_objects": consistent,
        "selected_objects": selected,
        "rejected_inconsistent": sorted(set(input_object_ids) - set(consistent)),
        "duplicate_groups": group_records,
        "duplicate_edges": edges,
        "object_statistics": statistics,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
