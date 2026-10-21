from __future__ import annotations
import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a label-free semantic-event selection containing every "
            "candidate in an exhaustive FastGS candidate manifest."
        )
    )
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_review(candidate_manifest_path, candidate_manifest):
    candidates = candidate_manifest.get("candidates") or []
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if not candidate_ids:
        raise ValueError("Candidate manifest contains no candidates")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate manifest contains duplicate candidate IDs")
    manifest_path = str(candidate_manifest_path.resolve())
    return {
        "candidate_manifest": manifest_path,
        "review_type": "automatic_all_exhaustive_candidates",
        "review_note": (
            "Retains every exhaustive automatic candidate. Human review and "
            "target GT are not used; intervention editability is evaluated "
            "downstream."
        ),
        "source_candidate_manifest": manifest_path,
        "target_gt_used": False,
        "objects": [
            {"candidate_id": candidate_id, "name": candidate_id}
            for candidate_id in candidate_ids
        ],
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidate_manifest = json.loads(args.candidate_manifest.read_text())
    review = build_review(args.candidate_manifest, candidate_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, indent=2))
    print(json.dumps({
        "review_type": review["review_type"],
        "candidate_count": len(review["objects"]),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
