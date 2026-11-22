"""Compare aligned GGUF teacher-forced scores on the same tokens and prefixes."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path


def compare(reference, candidate):
    matched = count = 0
    ref_sum = candidate_sum = max_delta = 0.0
    first_differences = []
    with reference.open() as a, candidate.open() as b:
        baseline = csv.DictReader(a, delimiter="\t")
        trial = csv.DictReader(b, delimiter="\t")
        for x, y in itertools.zip_longest(baseline, trial):
            if x is None or y is None:
                raise ValueError("The files have different lengths")
            if any(x[key] != y[key] for key in ("document", "position", "input_id", "next_id")):
                raise ValueError("The scored prefixes differ")
            ref_lp, new_lp = float(x["logp_next"]), float(y["logp_next"])
            if not math.isfinite(ref_lp) or not math.isfinite(new_lp):
                raise ValueError("Found a nonfinite token likelihood")
            count += 1
            if x["argmax_id"] == y["argmax_id"]:
                matched += 1
            elif len(first_differences) < 10:
                first_differences.append(dict(document=int(x["document"]), position=int(x["position"]),
                                              reference=int(x["argmax_id"]), candidate=int(y["argmax_id"])))
            ref_sum += ref_lp
            candidate_sum += new_lp
            max_delta = max(max_delta, abs(ref_lp - new_lp))
    if count == 0:
        raise ValueError("No scores found")
    return dict(positions=count, matches=matched, agreement=matched/count,
                reference_mean_nll=-ref_sum/count, candidate_mean_nll=-candidate_sum/count,
                max_logp_delta_at_recorded_precision=max_delta, first_differences=first_differences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.candidate), indent=2))
