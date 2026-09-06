"""The `locus` command-line front end.

Every verb below is documented in README.md and is backed by a
`main(argv: list[str] | None) -> int` -- either one that already existed
(`build/pipeline`, `eval/rank`, `experiments/completion`, ...) or one added
here for `fetch`, `doctor` and `ingest`, which had none.

This module is purely a dispatcher: it never does file-format or scoring
work itself, and it never imports anything but the standard library at
module scope. `build_parser()` only resolves which verb was asked for --
every target module (and therefore every heavy dependency it might need,
such as torch, transformers or gensim) is imported lazily, inside the
resolver a verb actually reaches, so `locus --help` succeeds in an
environment with only the base install.

`VERBS` maps a verb string -- a single word for a bare verb, two
space-separated words for one with a subcommand (e.g. `"eval rank"`, matching
`"eval rank".split()`) -- to a zero-argument *resolver*: a function that
imports and returns the callable that verb actually runs. Resolvers are not
called until their verb is invoked, but `tests/test_cli.py` calls every one
of them directly at test time, so a verb whose target fails to import is
caught in CI rather than by whoever runs `locus <verb>` first. This
repository has shipped a documented command that did not work, twice --
that failure mode is exactly what parametrising the test over this registry
forecloses: a verb cannot be added to `VERBS` without a passing import.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path

Resolver = Callable[[], Callable[[list[str] | None], int]]


# --- bare verbs: build/pipeline through eval/verify, forwarded as-is ------

def _resolve_build_pipeline() -> Callable[[list[str] | None], int]:
    from locus.build import pipeline

    return pipeline.main


def _resolve_build_graph() -> Callable[[list[str] | None], int]:
    from locus.build import graph

    return graph.main


def _resolve_build_embed_inputs() -> Callable[[list[str] | None], int]:
    from locus.build import embed_inputs

    return embed_inputs.main


def _resolve_build_export() -> Callable[[list[str] | None], int]:
    from locus.data import export

    return export.main


def _resolve_embed() -> Callable[[list[str] | None], int]:
    from locus.scoring import embed

    return embed.main


def _resolve_eval_rank() -> Callable[[list[str] | None], int]:
    from locus.eval import rank

    return rank.main


def _resolve_eval_swap() -> Callable[[list[str] | None], int]:
    from locus.eval import swap

    return swap.main


def _resolve_sweep() -> Callable[[list[str] | None], int]:
    from locus.eval import sweep

    return sweep.main


def _resolve_verify() -> Callable[[list[str] | None], int]:
    from locus.eval import verify

    return verify.main


def _resolve_exp_completion() -> Callable[[list[str] | None], int]:
    from locus.experiments import completion

    return completion.main


def _resolve_exp_influence() -> Callable[[list[str] | None], int]:
    from locus.experiments import influence

    return influence.main


def _resolve_exp_anchors() -> Callable[[list[str] | None], int]:
    from locus.experiments import anchor_analysis

    return anchor_analysis.main


# --- verbs with no existing main(): fetch, ingest, doctor, eval all -------

def _resolve_fetch() -> Callable[[list[str] | None], int]:
    from locus.data import fetch as fetch_mod

    def _main(argv: list[str] | None = None) -> int:
        ap = argparse.ArgumentParser(
            prog="locus fetch",
            description="Download a released artefact bundle and verify it "
            "against its manifest.",
        )
        ap.add_argument("dest", help="directory to fetch the bundle into")
        ap.add_argument(
            "--manifest-url", default=None,
            help="manifest to fetch; defaults to the deposited core "
            "bundle on Zenodo (locus.data.fetch.DEFAULT_MANIFEST_URL)",
        )
        ap.add_argument(
            "--no-verify", action="store_true",
            help="skip sha256 verification of each downloaded file",
        )
        args = ap.parse_args(argv)
        manifest = fetch_mod.fetch(
            args.manifest_url, args.dest, verify=not args.no_verify
        )
        n = len(manifest.get("files", {}))
        print(f"fetched {n} file(s) into {args.dest}")
        return 0

    return _main


def _resolve_bundle() -> Callable[[list[str] | None], int]:
    from locus.data import bundle

    return bundle.main


def _resolve_ingest() -> Callable[[list[str] | None], int]:
    from locus.data import ingest as ingest_mod

    def _main(argv: list[str] | None = None) -> int:
        ap = argparse.ArgumentParser(
            prog="locus ingest",
            description="Validate a directory against the JSONL corpus "
            "contract documented in locus.data.ingest: contexts.jsonl, "
            "papers.jsonl, splits/{train,val,test}.json.",
        )
        ap.add_argument(
            "root",
            help="directory holding contexts.jsonl, papers.jsonl, splits/",
        )
        ap.add_argument(
            "--max-unalignable-rate", type=float, default=0.5,
            help="see locus.data.ingest.validate (default: 0.5)",
        )
        args = ap.parse_args(argv)
        problems = ingest_mod.validate(
            args.root, max_unalignable_rate=args.max_unalignable_rate
        )
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            print(f"\n{len(problems)} problem(s) found in {args.root}",
                  file=sys.stderr)
            return 1
        print(f"{args.root}: valid")
        return 0

    return _main


def _resolve_eval_all() -> Callable[[list[str] | None], int]:
    from locus.eval import rank as rank_mod
    from locus.eval import swap as swap_mod

    def _main(argv: list[str] | None = None) -> int:
        ap = argparse.ArgumentParser(
            prog="locus eval all",
            description="Run both probes -- LOCUS-Rank then "
            "LOCUS-Swap -- against the same embeddings, with one exit code "
            "for both. `locus eval rank`/`locus eval swap` expose every "
            "knob (association measure, structural vectors, a custom "
            "slice); this convenience path takes the two embeddings tables "
            "and leaves everything else at its own default.",
        )
        ap.add_argument("--val-embeddings", required=True)
        ap.add_argument("--test-embeddings", required=True)
        ap.add_argument("--tag", required=True)
        ap.add_argument("--sweep", default=None, help="rank sweep, for alpha/min_count")
        args = ap.parse_args(argv)

        rank_argv = ["--embeddings", args.test_embeddings]
        if args.sweep is not None:
            rank_argv += ["--sweep", args.sweep]
        code = rank_mod.main(rank_argv)
        if code:
            return code

        swap_argv = [
            "--val-embeddings", args.val_embeddings,
            "--test-embeddings", args.test_embeddings,
            "--tag", args.tag,
        ]
        if args.sweep is not None:
            swap_argv += ["--sweep", args.sweep]
        return swap_mod.main(swap_argv)

    return _main


def _resolve_doctor() -> Callable[[list[str] | None], int]:
    from locus import config

    def _main(argv: list[str] | None = None) -> int:
        ap = argparse.ArgumentParser(
            prog="locus doctor",
            description="Report the environment this checkout is running "
            "in: Python version, which optional extras import, whether "
            "LOCUS_DATA_DIR/LOCUS_WORK_DIR are set, and whether an "
            "artefact bundle is present. Always exits 0 -- it is a report, "
            "not a gate.",
        )
        ap.parse_args(argv)

        print(f"Python:  {sys.version.split()[0]}  ({sys.executable})")

        print("\nOptional extras:")
        for module_name, extra in (
            ("torch", "gpu"), ("transformers", "gpu"),
            ("gensim", "node2vec"), ("boto3", "cloud"),
        ):
            found = importlib.util.find_spec(module_name) is not None
            status = (
                "installed" if found
                else f"not installed  (pip install -e '.[{extra}]')"
            )
            print(f"  {module_name:<12s} {status}")

        print("\nEnvironment:")
        raw_data_dir = os.environ.get("LOCUS_DATA_DIR")
        if raw_data_dir:
            exists = Path(raw_data_dir).is_dir()
            print(
                f"  LOCUS_DATA_DIR   {raw_data_dir}"
                f"  ({'exists' if exists else 'MISSING'})"
            )
        else:
            print("  LOCUS_DATA_DIR   not set  (needed for `locus build pipeline`)")
        work_dir = config.WORK_DIR
        explicit_work_dir = "LOCUS_WORK_DIR" in os.environ
        print(
            f"  LOCUS_WORK_DIR   {work_dir}"
            f"  ({'set' if explicit_work_dir else 'defaulted to ./work'})"
        )

        print("\nArtefact bundle:")
        frozen = [work_dir / f"index_{split}.pkl" for split in ("test", "val")]
        exported = [
            work_dir / "export" / f"locus_manifest_{split}.json"
            for split in ("test", "val")
        ]
        any_frozen = any(p.exists() for p in frozen)
        any_exported = any(p.exists() for p in exported)
        print(
            f"  frozen build artefacts   "
            f"{'present' if any_frozen else 'absent'}  ({work_dir})"
        )
        print(
            f"  portable export          "
            f"{'present' if any_exported else 'absent'}  ({work_dir / 'export'})"
        )
        if not any_frozen and not any_exported:
            print(
                "\n  Nothing built yet. `locus fetch <dest>` downloads the "
                "released bundle from Zenodo; `locus build pipeline` rebuilds "
                "it from the upstream corpus at LOCUS_DATA_DIR. See README.md."
            )
        return 0

    return _main


# --- the registry itself ---------------------------------------------------

# Single source of truth for both `VERBS` and `build_parser()`: a verb string
# cannot exist in one but not the other, because both are derived from these
# two mappings rather than kept in sync by hand.
_SIMPLE: dict[str, Resolver] = {
    "fetch": _resolve_fetch,
    "bundle": _resolve_bundle,
    "doctor": _resolve_doctor,
    "ingest": _resolve_ingest,
    "embed": _resolve_embed,
    "sweep": _resolve_sweep,
    "verify": _resolve_verify,
}
_GROUPED: dict[str, dict[str, Resolver]] = {
    "build": {
        "pipeline": _resolve_build_pipeline,
        "graph": _resolve_build_graph,
        "embed-inputs": _resolve_build_embed_inputs,
        "export": _resolve_build_export,
    },
    "eval": {
        "rank": _resolve_eval_rank,
        "swap": _resolve_eval_swap,
        "all": _resolve_eval_all,
    },
    "exp": {
        "completion": _resolve_exp_completion,
        "influence": _resolve_exp_influence,
        "anchors": _resolve_exp_anchors,
    },
}

VERBS: dict[str, Resolver] = {
    **_SIMPLE,
    **{
        f"{group} {child}": resolver
        for group, children in _GROUPED.items()
        for child, resolver in children.items()
    },
}


def build_parser() -> argparse.ArgumentParser:
    """The routing parser: it identifies which verb was asked for and
    nothing else.

    Leaf subparsers (`build pipeline`, `eval rank`, `fetch`, ...) declare no
    arguments of their own and set `add_help=False`, so a flag -- including
    `-h`/`--help` -- meant for the target callable is left in the unmatched
    arguments `main()` forwards to it, rather than being consumed or
    misinterpreted here. That is what lets `locus eval rank --help` show
    `eval/rank.py`'s own real argparse help instead of this router's.
    """
    parser = argparse.ArgumentParser(
        prog="locus", description="LOCUS: a within-paper citation-location "
        "discrimination benchmark."
    )
    verb_sub = parser.add_subparsers(dest="verb", required=True)
    for name in _SIMPLE:
        verb_sub.add_parser(name, add_help=False)
    for group, children in _GROUPED.items():
        group_parser = verb_sub.add_parser(group)
        child_sub = group_parser.add_subparsers(dest="subverb", required=True)
        for child in children:
            child_sub.add_parser(child, add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args, remainder = parser.parse_known_args(raw)
    key = args.verb if args.verb not in _GROUPED else f"{args.verb} {args.subverb}"
    target = VERBS[key]()
    return target(remainder)


if __name__ == "__main__":
    sys.exit(main())
