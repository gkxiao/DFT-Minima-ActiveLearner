#!/usr/bin/env python3

import argparse
import numpy as np

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from sklearn.linear_model import LogisticRegression

HARTREE_TO_KCAL = 627.509474


def parse_sdf(filename):

    mols = []

    with open(filename, "r") as f:
        content = f.read()

    blocks = content.split("$$$$")

    for block in blocks:

        block = block.strip()

        if not block:
            continue

        lines = block.splitlines()

        mol = {
            "Title": lines[0].strip()
        }

        i = 0

        while i < len(lines):

            line = lines[i]

            if line.startswith(">") and "<" in line and ">" in line:

                try:
                    prop = line[
                        line.index("<") + 1:
                        line.rindex(">")
                    ]
                except Exception:
                    i += 1
                    continue

                i += 1

                while (
                    i < len(lines)
                    and not lines[i].strip()
                ):
                    i += 1

                if i < len(lines):

                    value = lines[i].strip()

                    try:
                        value = float(value)
                    except Exception:
                        pass

                    mol[prop] = value

            i += 1

        mols.append(mol)

    return mols


def relative_energy(values):

    e_min = min(values)

    return [
        v - e_min
        for v in values
    ]


def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():

    parser = argparse.ArgumentParser(
        description="Conformer Recovery Analysis v2.0"
    )

    parser.add_argument(
        "--crest",
        required=True,
        help="crest_ensemble.sdf"
    )

    parser.add_argument(
        "--spe",
        required=True,
        help="ensemble_spe.sdf"
    )

    parser.add_argument(
        "--ewin",
        type=float,
        default=6.0,
        help="Target Psi4 relative energy window"
    )

    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.01,
        help="Bayesian probability threshold"
    )

    args = parser.parse_args()

    #
    # Read CREST ensemble
    #

    crest = parse_sdf(args.crest)

    crest = [
        m
        for m in crest
        if "Energy_xTB" in m
    ]

    if len(crest) == 0:
        raise RuntimeError(
            "No Energy_xTB property found in crest file."
        )

    xtb_abs = [
        m["Energy_xTB"] * HARTREE_TO_KCAL
        for m in crest
    ]

    xtb_rel = relative_energy(
        xtb_abs
    )

    for m, rel in zip(
        crest,
        xtb_rel
    ):
        m["E_xTB_Rel"] = rel

    crest_dict = {
        m["Title"]: m
        for m in crest
    }

    total_confs = len(crest)

    xtb_window = max(
        m["E_xTB_Rel"]
        for m in crest
    )

    #
    # xTB distribution
    #

    bins = [1, 2, 3, 4, 5, 6]

    xdist = {}

    for cutoff in bins:

        xdist[cutoff] = sum(
            1
            for m in crest
            if m["E_xTB_Rel"] <= cutoff
        )

    #
    # Read SPE file
    #

    spe = parse_sdf(args.spe)

    completed = []

    for mol in spe:

        title = mol.get("Title")

        if title not in crest_dict:
            continue

        if "Psi4_Energy (kcal/mol)" not in mol:
            continue

        merged = dict(
            crest_dict[title]
        )

        merged.update(mol)

        completed.append(merged)

    if len(completed) < 5:

        print(
            "\nToo few completed DFT conformers.\n"
        )

        return

    #
    # Psi4 relative energies
    #

    psi4_abs = [
        m["Psi4_Energy (kcal/mol)"]
        for m in completed
    ]

    psi4_rel = relative_energy(
        psi4_abs
    )

    for m, rel in zip(
        completed,
        psi4_rel
    ):

        m["E_Psi4_Rel"] = rel

    #
    # Ranking consistency
    #

    x = np.array([
        m["E_xTB_Rel"]
        for m in completed
    ])

    y = np.array([
        m["E_Psi4_Rel"]
        for m in completed
    ])

    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)

    r2 = r * r

    #
    # Hits
    #

    hits = [
        m
        for m in completed
        if m["E_Psi4_Rel"] <= args.ewin
    ]

    nhits = len(hits)

    if nhits > 0:

        recovery_frontier = max(
            m["E_xTB_Rel"]
            for m in hits
        )

    else:

        recovery_frontier = 0.0

    current_frontier = max(
        m["E_xTB_Rel"]
        for m in completed
    )

    frontier_margin = (
        current_frontier
        - recovery_frontier
    )

    remaining_space = (
        xtb_window
        - current_frontier
    )

    #
    # Consecutive misses
    #

    completed_sorted = sorted(
        completed,
        key=lambda m: m["E_xTB_Rel"]
    )

    last_hit_index = -1

    for i, m in enumerate(
        completed_sorted
    ):

        if (
            m["E_Psi4_Rel"]
            <= args.ewin
        ):
            last_hit_index = i

    consecutive_misses = (
        len(completed_sorted)
        - last_hit_index
        - 1
    )

    #
    # Compression ratio
    #

    xtb_confs_in_window = sum(
        1
        for m in crest
        if m["E_xTB_Rel"] <= args.ewin
    )

    compression_ratio = None

    if xtb_confs_in_window > 0:

        compression_ratio = (
            nhits / xtb_confs_in_window
        )

    #
    # Recovery efficiency
    #

    recovery_efficiency = []

    max_cutoff = int(
        np.ceil(xtb_window)
    )

    for cutoff in range(
        1,
        max_cutoff + 1
    ):

        recovered = sum(
            1
            for m in hits
            if m["E_xTB_Rel"] <= cutoff
        )

        frac = (
            100.0 * recovered / nhits
            if nhits > 0
            else 0.0
        )

        recovery_efficiency.append(
            (
                cutoff,
                recovered,
                frac
            )
        )

    #
    # Hit distribution
    #

    hit_xtb = [
        m["E_xTB_Rel"]
        for m in hits
    ]

    p90 = None
    p95 = None

    if len(hit_xtb) > 0:

        p90 = np.percentile(
            hit_xtb,
            90
        )

        p95 = np.percentile(
            hit_xtb,
            95
        )

    #
    # Coverage risk
    #

    occupancy = (
        100.0
        * recovery_frontier
        / xtb_window
        if xtb_window > 0
        else 0.0
    )

    if occupancy < 70:

        risk_level = "LOW"

    elif occupancy < 90:

        risk_level = "MEDIUM"

    elif occupancy < 95:

        risk_level = "HIGH"

    else:

        risk_level = "CRITICAL"

    #
    # Bayesian predictor
    #

    bayes_cutoff = None
    expected_remaining_hits = None

    labels = np.array([

        1 if m["E_Psi4_Rel"] <= args.ewin
        else 0

        for m in completed

    ])

    if len(set(labels)) >= 2:

        model = LogisticRegression(
            max_iter=1000
        )

        model.fit(
            x.reshape(-1, 1),
            labels
        )

        scan = np.linspace(
            0,
            xtb_window,
            2000
        )

        probs = model.predict_proba(
            scan.reshape(-1, 1)
        )[:, 1]

        for e, p in zip(
            scan,
            probs
        ):

            if p < args.prob_threshold:

                bayes_cutoff = e
                break

        done_titles = set(
            m["Title"]
            for m in completed
        )

        remaining = [
            m
            for m in crest
            if m["Title"]
            not in done_titles
        ]

        if len(remaining):

            rem_x = np.array([
                m["E_xTB_Rel"]
                for m in remaining
            ])

            rem_probs = model.predict_proba(
                rem_x.reshape(-1, 1)
            )[:, 1]

            expected_remaining_hits = (
                rem_probs.sum()
            )

        else:

            expected_remaining_hits = 0.0

    #
    # REPORT
    #

    section(
        "Input Ensemble Summary"
    )

    print(
        f"Total conformers                  : {total_confs}"
    )

    print(
        f"xTB energy window                 : {xtb_window:.2f} kcal/mol"
    )

    print()

    for cutoff in bins:

        print(
            f"Conformers <= {cutoff} kcal/mol        : "
            f"{xdist[cutoff]}"
        )

    section(
        "DFT Progress"
    )

    print(
        f"DFT completed                     : "
        f"{len(completed)} / {total_confs}"
    )

    print(
        f"Completion                        : "
        f"{100*len(completed)/total_confs:.1f}%"
    )

    print(
        f"Current Frontier                  : "
        f"{current_frontier:.2f} kcal/mol"
    )

    print(
        f"Remaining Search Space            : "
        f"{remaining_space:.2f} kcal/mol"
    )

    section(
        "Recovery Analysis"
    )

    print(
        f"Target Window                     : "
        f"{args.ewin:.2f} kcal/mol"
    )

    print(
        f"DFT Hits                          : "
        f"{nhits}"
    )

    print(
        f"Recovery Frontier                 : "
        f"{recovery_frontier:.2f} kcal/mol"
    )

    print(
        f"Coverage Beyond Recovery Frontier : "
        f"{frontier_margin:.2f} kcal/mol"
    )

    print(
        f"Consecutive Misses                : "
        f"{consecutive_misses}"
    )

    section(
        "Recovery Efficiency Analysis"
    )

    if compression_ratio is not None:

        print(
            f"DFT Window Compression Ratio      : "
            f"{compression_ratio:.3f}"
        )

    print()

    print(
        "Recovery Efficiency"
    )

    print(
        "-" * 50
    )

    for cutoff, recovered, frac in recovery_efficiency:

        print(
            f"xTB <= {cutoff:2d} kcal/mol : "
            f"{recovered:4d}/{nhits:<4d} "
            f"({frac:5.1f}%)"
        )

    print()

    print(
        "Among DFT Hits"
    )

    print(
        "-" * 50
    )

    if p90 is not None:

        print(
            f"90th percentile E_xTB_Rel        : "
            f"{p90:.2f} kcal/mol"
        )

        print(
            f"95th percentile E_xTB_Rel        : "
            f"{p95:.2f} kcal/mol"
        )

        print(
            f"Maximum E_xTB_Rel                : "
            f"{max(hit_xtb):.2f} kcal/mol"
        )

    section(
        "Coverage Risk Assessment"
    )

    print(
        f"Recovery Frontier Occupancy      : "
        f"{occupancy:.1f}%"
    )

    print(
        f"Coverage Risk Level              : "
        f"{risk_level}"
    )

    section(
        "Ranking Consistency"
    )

    print(
        f"Pearson R²                        : "
        f"{r2:.3f}"
    )

    print(
        f"Spearman rho                      : "
        f"{rho:.3f}"
    )

    section(
        "Bayesian Recovery Predictor"
    )

    if bayes_cutoff is not None:

        print(
            f"Suggested Bayesian Cutoff         : "
            f"{bayes_cutoff:.2f} kcal/mol"
        )

        print(
            f"Probability Threshold             : "
            f"{100*args.prob_threshold:.1f}%"
        )

        print(
            f"Expected Remaining Hits           : "
            f"{expected_remaining_hits:.2f}"
        )

    else:

        print(
            "Insufficient positive/negative examples."
        )

    section(
        "Recommendation"
    )

    if len(completed) / total_confs > 0.95:

        print(
            "DFT calculations are essentially complete."
        )

        print()

        if risk_level == "CRITICAL":

            print(
                "WARNING:"
            )

            print(
                "Recovery Frontier reaches the boundary "
                "of the sampled xTB space."
            )

            print()

            print(
                "The current data do NOT demonstrate "
                f"complete coverage of the target "
                f"DFT <= {args.ewin:.1f} kcal/mol ensemble."
            )

            print()

            print(
                "Recommended action:"
            )

            print(
                "Repeat CREST using a larger xTB energy "
                "window (e.g. 12-15 kcal/mol)."
            )

        elif risk_level == "HIGH":

            print(
                "Recovery Frontier is close to the "
                "boundary of the searched xTB space."
            )

            print(
                "Consider validating with a larger "
                "xTB window."
            )

        else:

            print(
                "Current xTB search window appears "
                "sufficient."
            )

    else:

        print(
            "DFT calculations are still in progress."
        )

        print()

        if expected_remaining_hits is not None:

            print(
                f"Expected Remaining Hits: "
                f"{expected_remaining_hits:.2f}"
            )

        if (
            rho > 0.90
            and frontier_margin > 2.0
            and consecutive_misses > 15
        ):

            print(
                "\nSTOP RECOMMENDED"
            )

        else:

            print(
                "\nCONTINUE CALCULATIONS"
            )


if __name__ == "__main__":
    main()
