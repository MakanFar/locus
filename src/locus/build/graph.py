"""Graph driver: build the train-only co-citation graph and run its gates.

Usage:  .venv/bin/python -m locus.build.graph
        .venv/bin/python -m locus.build.graph --corpus DIR
Exit code 0 iff every gate passes.

`--corpus DIR` reads a JSONL corpus in the contract `locus.data.ingest`
documents instead of `$LOCUS_DATA_DIR`, for the same reason `build.pipeline`
and `build.embed_inputs` take it: without a co-citation graph there is no
rescorer, so a custom corpus that could not build one could not run the
benchmark's actual result. The pair-instance band in step 3 is calibrated to
Gu et al.'s train split, so under `--corpus` it is reported rather than
enforced; the leakage, PPMI-identity and determinism gates are enforced on
every corpus. With no `--corpus`, nothing here behaves differently.
"""
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np

from locus import config
from locus.data.corpus import iter_papers
from locus.scoring.cocitation import CoGraph, count_sites


def _peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / 2**30  # ru_maxrss is bytes on darwin
    if sys.platform.startswith("linux"):
        return peak / 2**20  # ru_maxrss is kilobytes on linux
    raise SystemExit(
        f"ru_maxrss units are unknown on platform {sys.platform!r} -- the "
        "2 GB peak-RSS gate cannot be evaluated safely without guessing "
        "whether it reports bytes or kilobytes here"
    )


def _digest_of_graph(g: CoGraph) -> str:
    """A sha256 over everything that makes two CoGraph builds "the same"."""
    h = hashlib.sha256()
    h.update(g.counts.indptr.tobytes())
    h.update(g.counts.indices.tobytes())
    h.update(g.counts.data.tobytes())
    h.update(g.marginals.tobytes())
    for v in g.vocab:
        h.update(v.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


_CHILD_SCRIPT = (
    "import sys; "
    "from locus.build.graph import _digest_of_graph; "
    "from locus.scoring.cocitation import count_sites; "
    "from locus.data.corpus import iter_papers; "
    "from locus.data.ingest import jsonl_source; "
    "corpus = sys.argv[1] or None; "
    "src = jsonl_source(corpus) if corpus else None; "
    "g = count_sites(iter_papers('val', keep_text=False, source=src)); "
    "print(_digest_of_graph(g))"
)


def _run_second_val_build(hashseed: str, corpus: str | None = None) -> str:
    """Rebuild the val CoGraph in a fresh subprocess under a different
    PYTHONHASHSEED and return its digest.

    The only nondeterminism count_sites guards against is PYTHONHASHSEED-
    dependent set iteration order, which is fixed within a single process by
    construction -- so comparing two in-process builds is structurally
    incapable of catching a regression. A subprocess with a different hash
    seed actually can.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT, corpus or ""],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # --splits exists for the train-only-vs-full leakage control, which
    # needs a deliberately leaky graph to measure how much of the gain
    # careless construction would have manufactured. Anything other than the
    # default train-only build is refused a .npz under the canonical name.
    ap.add_argument("--splits", default="train")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--corpus", default=None,
        help="a JSONL corpus directory (locus.data.ingest); without it the "
             "upstream corpus at $LOCUS_DATA_DIR is used",
    )
    args = ap.parse_args(argv)
    splits = [x.strip() for x in args.splits.split(",")]
    leaky_by_design = bool(set(splits) & {"val", "test"})
    source = None
    if args.corpus is not None:
        from locus.data.ingest import jsonl_source

        source = jsonl_source(args.corpus)

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    scale_notes: list[str] = []

    print(f"[1/5] counting co-citations over {splits} "
          "(one streaming pass per split, ~10 min each)")
    counted_papers: set[str] = set()

    def _tracked():
        for split in splits:
            for citing_id, recs in iter_papers(
                split, keep_text=False, source=source
            ):
                counted_papers.add(citing_id)
                yield citing_id, recs

    graph = count_sites(_tracked())
    if not counted_papers:
        raise SystemExit(
            f"0 papers streamed for {splits} -- check "
            + (f"the corpus at {args.corpus}" if args.corpus
               else "LOCUS_DATA_DIR")
            + " and that those split files are populated"
        )
    instances = graph.n_total // 2
    print(f"      {len(counted_papers)} papers, {len(graph.vocab)} refids, "
          f"{graph.counts.nnz} directed nonzeros, {instances} pair instances")

    print("[2/5] leakage")
    # iter_papers only yields papers with at least one context in the split,
    # so every paper it names is held out. No need to re-read the split files.
    held_out: set[str] = set()
    for split in ("val", "test"):
        for citing_id, _recs in iter_papers(
            split, keep_text=False, source=source
        ):
            held_out.add(citing_id)
    overlap = counted_papers & held_out
    print(f"      {len(held_out)} held-out papers, {len(overlap)} overlap")
    if not held_out:
        failures.append(
            "leakage gate could not be evaluated: 0 held-out papers were "
            "found across val and test -- check LOCUS_DATA_DIR and that "
            "those split files are populated; this does NOT mean leakage "
            "was found"
        )
    elif overlap and not leaky_by_design:
        failures.append(
            f"LEAKAGE: {len(overlap)} papers counted into the graph also appear "
            f"in val or test, e.g. {sorted(overlap)[:5]}"
        )
    elif overlap:
        print(f"      leakage EXPECTED and present ({len(overlap)} papers): this "
              "build is the train-only-vs-full control, not a usable graph")

    print("[3/5] pair-instance sanity band")
    if not 1_600_000 <= instances <= 6_400_000:
        # Calibrated to Gu et al.'s train split, so it is a note rather than a
        # failure under --corpus: a 192-context corpus cannot have 1.6M pair
        # instances and is not defective for it.
        (scale_notes if args.corpus is not None else failures).append(
            f"pair instances {instances} outside 1.6M-6.4M; the projection is "
            "1.063 per context measured on test, so this suggests a structural "
            "break rather than a distributional surprise"
        )

    print("[4/5] alpha=1 identity against the spec's closed form")
    p1 = graph.ppmi(alpha=1.0, min_count=1)
    coo = graph.counts.tocoo()
    rng = np.random.default_rng(config.SEED)
    take = rng.choice(coo.nnz, size=min(1000, coo.nnz), replace=False)
    marg = graph.marginals.astype(np.float64)
    bad = 0
    for t in take:
        i, j, cij = int(coo.row[t]), int(coo.col[t]), float(coo.data[t])
        want = max(0.0, float(np.log(cij * graph.n_total / (marg[i] * marg[j]))))
        got = float(p1.matrix[i, j])
        if abs(want - got) > 1e-9:
            bad += 1
    print(f"      {len(take)} sampled nonzeros, {bad} mismatches")
    if bad:
        failures.append(
            f"alpha=1 PPMI disagrees with the spec's closed form on {bad} of "
            f"{len(take)} sampled nonzeros"
        )

    print("[5/5] determinism on val, across processes")
    a = count_sites(iter_papers("val", keep_text=False, source=source))
    digest_a = _digest_of_graph(a)
    # Any concrete seed differing from whatever this process happens to be
    # running under is enough to exercise a different set-iteration order.
    child_seed = "0" if os.environ.get("PYTHONHASHSEED") != "0" else "1"
    digest_b = _run_second_val_build(child_seed, args.corpus)
    same = digest_a == digest_b
    print(f"      parent/child val digests match: {same}")
    if not same:
        failures.append(
            "count_sites is not deterministic across processes (val digest "
            f"mismatch under a different PYTHONHASHSEED: {digest_a} != {digest_b})"
        )

    # A leaky control build must never land on the canonical filename that
    # every scorer loads by default.
    out = args.out or config.WORK_DIR / (
        "cocitation_full.npz" if leaky_by_design else "cocitation_train.npz"
    )

    peak = _peak_rss_gb()
    print(f"      peak RSS: {peak:.2f} GB")
    if peak >= 2.0:
        failures.append(f"peak RSS {peak:.2f} GB >= 2 GB")

    report = {
        "papers_counted": len(counted_papers),
        "vocab": len(graph.vocab),
        "directed_nonzeros": int(graph.counts.nnz),
        "pair_instances": int(instances),
        "n_total": int(graph.n_total),
        "held_out_papers": len(held_out),
        "leakage_overlap": len(overlap),
        "alpha1_mismatches": bad,
        "val_determinism": same,
        "peak_rss_gb": peak,
        "splits": splits,
        "leaky_by_design": leaky_by_design,
        "corpus": args.corpus,
        "failures": failures,
        "scale_notes": scale_notes,
    }
    tag = "" if not leaky_by_design else "_full"
    with open(config.WORK_DIR / f"graph_report{tag}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if scale_notes:
        print(
            "\nNOTE: these thresholds describe Gu et al.'s corpus, not the "
            "benchmark definition, and are reported rather than enforced "
            "under --corpus:"
        )
        for note in scale_notes:
            print(f"  - {note}")
    if failures:
        print("\nGATE FAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        print(f"      {out} NOT written -- one or more gates failed")
        return 1

    graph.save(out)
    print(f"      wrote {out}")
    print("\nAll graph gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
