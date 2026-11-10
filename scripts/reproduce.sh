#!/usr/bin/env bash
# Run every experiment in the paper, one step at a time, in dependency order.
#
# This is docs/reproduce.md as a script: same commands, same order, same
# output names. Read that file for the numbers each step should print.
#
#   scripts/reproduce.sh tier0            # no corpus, no GPU: fetch + verify
#   scripts/reproduce.sh all              # every stage below, in order
#   scripts/reproduce.sh build rank       # just these stages
#   scripts/reproduce.sh --dry-run all    # print the commands, run nothing
#   scripts/reproduce.sh --force rank     # rerun steps whose output exists
#   scripts/reproduce.sh --list           # list the stages
#
# Stages (`all` runs them in this order; `tier0` is separate):
#   build       pools, hardness, headline set, co-citation graph, embed inputs
#   embed       SPECTER2 and SciNCL vectors (HAtten too, if its checkpoint is given)
#   rank        Tables 1 and 2: sweep, Rank and Swap for SPECTER2, SciNCL, HAtten, BM25
#   controls    Tables 10 and 7: easy slice, random/degree-matched anchors, raw counts,
#               the anchor-route ablation and anchor proximity
#   siblings    Table 10: sibling-titles control (embeds its own inputs)
#   node2vec    Table 10 and the appendix: node2vec association, with its q grid
#   completion  Table 3 A/B, Table 4 and Figure 3: sequential completion, influence,
#               anchors, and node2vec in completion
#   corpuswide  Table 3 C/D and Table 4: completion over corpus-wide retrieval
#   theta       sensitivity of the headline to the clustering threshold: four
#               rebuilt work directories (about an hour) and one comparison
#   export      the release files, round-tripped and checked against the floors
#
# Environment:
#   LOCUS_DATA_DIR   the upstream corpus (Gu et al. 2022); needed by `build`
#   LOCUS_WORK_DIR   where artefacts and reports go (default: ./work)
#   LOCUS_HATTEN_CKPT, LOCUS_HATTEN_VOCAB, LOCUS_HATTEN_LCR_SRC
#                    HAtten's checkpoint, vocabulary and src/prefetch directory
#                    (docs/scoring.md); without them the HAtten rows need
#                    work/embeddings_hatten_{val,test}.parquet to already exist
#   LOCUS, PYTHON    the executables (default: .venv/bin/locus, .venv/bin/python)
#
# A step is skipped when its output already exists, so an interrupted run
# resumes where it stopped. That also means stale outputs are trusted: after
# changing anything upstream, pass --force or start from an empty work
# directory. A step whose inputs are missing is skipped with the reason, and
# the run carries on with whatever does not depend on it. Each step's output
# is kept in $LOCUS_WORK_DIR/logs/. Exit status is 1 if any step failed.

set -u
cd "$(dirname "$0")/.."

ALL_STAGES="build embed rank controls siblings node2vec completion corpuswide theta export"

DRY=0
FORCE=0
STAGES=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --force) FORCE=1 ;;
    --list) echo "tier0 $ALL_STAGES" | tr ' ' '\n'; exit 0 ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    all) STAGES="$STAGES $ALL_STAGES" ;;
    tier0|build|embed|rank|controls|siblings|node2vec|completion|corpuswide|theta|export)
      STAGES="$STAGES $arg" ;;
    *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done
if [ -z "$STAGES" ]; then
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

export LOCUS_WORK_DIR="${LOCUS_WORK_DIR:-$PWD/work}"
W="$LOCUS_WORK_DIR"
if [ -z "${LOCUS:-}" ]; then
  if [ -x .venv/bin/locus ]; then LOCUS=.venv/bin/locus; else LOCUS=locus; fi
fi
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
mkdir -p "$W/logs"

SUMMARY=""
FAILED=0
note() { SUMMARY="$SUMMARY$(printf '%-8s %s' "$1" "$2")"$'\n'; }

# need LABEL FILE...  -- returns 1 (and records a skip) if any input is missing
need() {
  local label="$1"; shift
  local f
  for f in "$@"; do
    if [ ! -e "$f" ]; then
      echo "--- skip $label: missing $f"
      note skipped "$label (missing ${f#"$W"/})"
      return 1
    fi
  done
}

# step LABEL OUTPUT CMD...  -- run CMD unless OUTPUT exists; log to logs/LABEL.log
# OK_CODES lists the exit codes that count as success (default "0").
OK_CODES="0"
step() {
  local label="$1" out="$2"; shift 2
  local ok="$OK_CODES"; OK_CODES="0"
  if [ "$FORCE" = 0 ] && [ -e "$out" ]; then
    echo "--- done $label: ${out#"$W"/} exists"
    note cached "$label"
    return 0
  fi
  echo
  echo "=== $label"
  echo "    $*"
  if [ "$DRY" = 1 ]; then note dry-run "$label"; return 0; fi
  local start=$SECONDS code
  "$@" 2>&1 | tee "$W/logs/$label.log"
  code=${PIPESTATUS[0]}
  case " $ok " in
    *" $code "*)
      note ok "$label ($((SECONDS - start))s)" ;;
    *)
      echo "!!! $label exited $code (log: $W/logs/$label.log)"
      note FAILED "$label (exit $code)"
      FAILED=1
      return 1 ;;
  esac
}

# In a dry run nothing is produced, so input checks would skip every later step.
[ "$DRY" = 1 ] && need() { return 0; }

# sweep + Rank + Swap for one dense base: bases TAG VAL_EMB TEST_EMB
dense_base() {
  local tag="$1" val="$2" test="$3"
  need "$tag" "$val" "$test" "$W/cocitation_train.npz" || return 0
  step "sweep-$tag" "$W/sweep_val_$tag.json" \
    "$LOCUS" sweep --embeddings "$val" --out "$W/sweep_val_$tag.json" || return 0
  step "rank-$tag" "$W/rank_report_$tag.json" \
    "$LOCUS" eval rank --embeddings "$test" \
      --sweep "$W/sweep_val_$tag.json" --out "$W/rank_report_$tag.json"
  step "swap-$tag" "$W/swap_report_$tag.json" \
    "$LOCUS" eval swap --val-embeddings "$val" --test-embeddings "$test" \
      --sweep "$W/sweep_val_$tag.json" --tag "$tag"
}

stage_tier0() {
  step fetch "$W/release/locus_manifest_test.json" "$LOCUS" fetch "$W/release"
  step fetch-embeddings "$W/embeddings_test.parquet" \
    curl -fL -o "$W/embeddings_test.parquet" \
      https://zenodo.org/records/22432297/files/embeddings_test.parquet
  need verify "$W/release/locus_manifest_test.json" "$W/embeddings_test.parquet" || return 0
  step verify "$W/logs/verify.ok" \
    "$LOCUS" verify --export "$W/release" \
      --embeddings "$W/embeddings_test.parquet" \
      --graph "$W/release/cocitation_train.npz" \
    && [ "$DRY" = 0 ] && touch "$W/logs/verify.ok"
}

stage_build() {
  if [ -z "${LOCUS_DATA_DIR:-}" ] && [ "$DRY" = 0 ]; then
    echo "--- skip build: LOCUS_DATA_DIR is not set (the upstream corpus; docs/schema.md)"
    note skipped "build (LOCUS_DATA_DIR not set)"
    return 0
  fi
  step pipeline-test "$W/build_report_test.json" "$LOCUS" build pipeline --split test
  # val fails one gate (alignment failure above 15%) after writing every
  # artefact; docs/reproduce.md explains why that is left as it is.
  OK_CODES="0 1"
  step pipeline-val "$W/build_report_val.json" "$LOCUS" build pipeline --split val
  step graph "$W/cocitation_train.npz" "$LOCUS" build graph
  step embed-inputs-test "$W/embed_inputs_test.parquet" "$LOCUS" build embed-inputs --split test
  step embed-inputs-val "$W/embed_inputs_val.parquet" "$LOCUS" build embed-inputs --split val
  # The pairs two annotators label (docs/annotation_protocol.md); index only.
  step annotation-sample "$W/annotation/manifest.json" \
    "$LOCUS" exp annotation-sample --out "$W/annotation"
}

stage_embed() {
  local split
  need embed "$W/embed_inputs_test.parquet" "$W/embed_inputs_val.parquet" || return 0
  for split in val test; do
    step "embed-specter2-$split" "$W/embeddings_$split.parquet" \
      "$LOCUS" embed --input "$W/embed_inputs_$split.parquet" \
        --output "$W/embeddings_$split.parquet"
  done
  for split in val test; do
    step "embed-scincl-$split" "$W/embeddings_scincl_$split.parquet" \
      "$LOCUS" embed --model malteos/scincl \
        --input "$W/embed_inputs_$split.parquet" \
        --output "$W/embeddings_scincl_$split.parquet"
  done
  if [ -n "${LOCUS_HATTEN_CKPT:-}" ]; then
    for split in val test; do
      step "embed-hatten-$split" "$W/embeddings_hatten_$split.parquet" \
        "$PYTHON" -m locus.experiments.hatten --split "$split" \
          --ckpt "$LOCUS_HATTEN_CKPT" --vocab "${LOCUS_HATTEN_VOCAB:-}" \
          --lcr-src "${LOCUS_HATTEN_LCR_SRC:-}"
    done
  else
    echo "--- skip embed-hatten: LOCUS_HATTEN_CKPT is not set (docs/scoring.md)"
    note skipped "embed-hatten (LOCUS_HATTEN_CKPT not set)"
  fi
}

stage_rank() {
  # SPECTER2 keeps the default report names: every later stage reads
  # sweep_val.json as the selected alpha, min_count and lambda.
  if need specter2 "$W/embeddings_val.parquet" "$W/embeddings_test.parquet" \
      "$W/cocitation_train.npz"; then
    step sweep-specter2 "$W/sweep_val.json" \
      "$LOCUS" sweep --embeddings "$W/embeddings_val.parquet" \
    && {
      step rank-specter2 "$W/rank_report.json" \
        "$LOCUS" eval rank --embeddings "$W/embeddings_test.parquet"
      step swap-specter2 "$W/swap_report_specter2.json" \
        "$LOCUS" eval swap --val-embeddings "$W/embeddings_val.parquet" \
          --test-embeddings "$W/embeddings_test.parquet" --tag specter2
    }
  fi
  dense_base scincl "$W/embeddings_scincl_val.parquet" "$W/embeddings_scincl_test.parquet"
  dense_base hatten "$W/embeddings_hatten_val.parquet" "$W/embeddings_hatten_test.parquet"
  if need bm25 "$W/embed_inputs_val.parquet" "$W/embed_inputs_test.parquet" \
      "$W/cocitation_train.npz"; then
    step sweep-bm25 "$W/sweep_val_bm25.json" \
      "$LOCUS" sweep --texts "$W/embed_inputs_val.parquet" --out "$W/sweep_val_bm25.json" \
    && {
      step rank-bm25 "$W/rank_report_bm25.json" \
        "$LOCUS" eval rank --texts "$W/embed_inputs_test.parquet" \
          --sweep "$W/sweep_val_bm25.json" --out "$W/rank_report_bm25.json"
      step swap-bm25 "$W/swap_report_bm25.json" \
        "$LOCUS" eval swap --val-texts "$W/embed_inputs_val.parquet" \
          --test-texts "$W/embed_inputs_test.parquet" \
          --sweep "$W/sweep_val_bm25.json" --tag bm25
    }
  fi
}

stage_controls() {
  local val="$W/embeddings_val.parquet" test="$W/embeddings_test.parquet"
  need controls "$val" "$test" "$W/sweep_val.json" || return 0
  step rank-easy "$W/rank_report_easy.json" \
    "$LOCUS" eval rank --embeddings "$test" --slice easy --out "$W/rank_report_easy.json"
  step controls "$W/controls_specter2.json" \
    "$PYTHON" -m locus.eval.controls --embeddings "$test"
  step hard-easy-contrast "$W/hard_easy_contrast_specter2.json" \
    "$PYTHON" scripts/hard_easy_contrast.py --embeddings "$test" \
      --out "$W/hard_easy_contrast_specter2.json"
  # Robustness of the location inference: one anchor route at a time, and
  # anchors by how close their marker is to the target's.
  local route
  for route in marker clustering; do
    step "rank-route-$route" "$W/rank_report_route_$route.json" \
      "$LOCUS" eval rank --embeddings "$test" --anchor-route "$route" \
        --out "$W/rank_report_route_$route.json"
  done
  step proximity "$W/proximity_specter2.json" \
    "$LOCUS" exp proximity --embeddings "$test" --out "$W/proximity_specter2.json"
  step sweep-count "$W/sweep_val_count.json" \
    "$LOCUS" sweep --embeddings "$val" --assoc count --out "$W/sweep_val_count.json" \
  && step rank-count "$W/rank_report_count.json" \
    "$LOCUS" eval rank --embeddings "$test" --assoc count \
      --sweep "$W/sweep_val_count.json" --out "$W/rank_report_count.json"
}

stage_siblings() {
  local split
  need siblings "$W/index_val.pkl" "$W/index_test.pkl" || return 0
  for split in val test; do
    step "embed-inputs-siblings-$split" "$W/embed_inputs_siblings_$split.parquet" \
      "$LOCUS" build embed-inputs --split "$split" --variant siblings || return 0
    step "embed-siblings-$split" "$W/embeddings_siblings_$split.parquet" \
      "$LOCUS" embed --input "$W/embed_inputs_siblings_$split.parquet" \
        --output "$W/embeddings_siblings_$split.parquet" || return 0
  done
  step sweep-siblings "$W/sweep_val_siblings.json" \
    "$LOCUS" sweep --embeddings "$W/embeddings_siblings_val.parquet" \
      --out "$W/sweep_val_siblings.json" \
  && step rank-siblings "$W/rank_report_siblings.json" \
    "$LOCUS" eval rank --embeddings "$W/embeddings_siblings_test.parquet" \
      --sweep "$W/sweep_val_siblings.json" --out "$W/rank_report_siblings.json"
}

stage_node2vec() {
  local val="$W/embeddings_val.parquet" test="$W/embeddings_test.parquet"
  local vec="$W/node2vec_p1q8"
  need node2vec "$val" "$test" "$W/cocitation_train.npz" || return 0
  step node2vec-train "${vec}_vectors.npy" "$PYTHON" -m locus.scoring.node2vec --p 1 --q 8 \
  && step sweep-node2vec "$W/sweep_val_node2vec.json" \
    "$LOCUS" sweep --embeddings "$val" --assoc node2vec --vectors "$vec" \
      --out "$W/sweep_val_node2vec.json" \
  && {
    step rank-node2vec "$W/rank_report_node2vec.json" \
      "$LOCUS" eval rank --embeddings "$test" --assoc node2vec --vectors "$vec" \
        --sweep "$W/sweep_val_node2vec.json" --out "$W/rank_report_node2vec.json"
    step swap-node2vec "$W/swap_report_node2vec.json" \
      "$LOCUS" eval swap --val-embeddings "$val" --test-embeddings "$test" \
        --assoc node2vec --vectors "$vec" \
        --sweep "$W/sweep_val_node2vec.json" --tag node2vec
  }
  # The rest of the q grid (p, q) is selected over; q = 8 is the paper's row.
  local q
  for q in 0.5 1 2 4; do
    step "node2vec-train-q$q" "$W/node2vec_p1q${q}_vectors.npy" \
      "$PYTHON" -m locus.scoring.node2vec --p 1 --q "$q" \
    && step "sweep-node2vec-q$q" "$W/sweep_val_node2vec_q$q.json" \
      "$LOCUS" sweep --embeddings "$val" --assoc node2vec --vectors "$W/node2vec_p1q$q" \
        --out "$W/sweep_val_node2vec_q$q.json" \
    && step "swap-node2vec-q$q" "$W/swap_report_node2vec_q$q.json" \
      "$LOCUS" eval swap --val-embeddings "$val" --test-embeddings "$test" \
        --assoc node2vec --vectors "$W/node2vec_p1q$q" \
        --sweep "$W/sweep_val_node2vec_q$q.json" --tag "node2vec_q$q"
  done
}

# The corpus-wide vectors, memory-mapped. Shared by completion and corpuswide.
STORE_MSG="download the seven embeddings_test_shard00*.parquet files from https://doi.org/10.5281/zenodo.22434185"
mm_store() {
  if [ ! -e "$W/embeddings_expb_test_mm_index.json" ] && [ ! -d "$W/embeddings_expb_test" ] \
      && [ "$DRY" = 0 ]; then
    echo "--- skip $1: no $W/embeddings_expb_test/ ($STORE_MSG)"
    note skipped "$1 (corpus-wide vectors missing)"
    return 1
  fi
  step mmapstore "$W/embeddings_expb_test_mm_index.json" \
    "$PYTHON" -m locus.scoring.mmapstore --shards "$W/embeddings_expb_test"
}

stage_completion() {
  local mm="$W/embeddings_expb_test_mm"
  need completion "$W/index_test.pkl" "$W/sweep_val.json" || return 0
  mm_store completion || return 0
  step completion-inputs "$W/expb_task_test.jsonl" \
    "$PYTHON" -m locus.experiments.completion_inputs --split test || return 0
  step completion-revealed "$W/expb_report_revealed.json" \
    "$LOCUS" exp completion --embeddings "$mm" --mode revealed --standardize none \
      --out "$W/expb_report_revealed.json"
  step completion-selfseed "$W/expb_report_selfseed.json" \
    "$LOCUS" exp completion --embeddings "$mm" --mode selfseed --standardize none \
      --out "$W/expb_report_selfseed.json"
  need influence "$W/embeddings_test.parquet" || return 0
  step influence "$W/influence_rows_specter2.pkl" \
    "$LOCUS" exp influence --embeddings "$W/embeddings_test.parquet" \
      --sim-store "$mm" --dump "$W/influence_rows_specter2.pkl" \
  && step anchors "$W/anchor_map_specter2.json" "$LOCUS" exp anchors --tag specter2
}

stage_corpuswide() {
  local mm="$W/embeddings_expb_test_mm"
  local task="$W/expb_task_test_corpuswide.jsonl"
  need corpuswide "$W/index_test.pkl" "$W/sweep_val.json" || return 0
  mm_store corpuswide || return 0
  step retrieve "$task" "$PYTHON" -m locus.scoring.retrieve --store "$mm" --k 2000 || return 0
  step completion-corpuswide-selfseed "$W/expb_report_corpuswide.json" \
    "$LOCUS" exp completion --embeddings "$mm" --mode selfseed --standardize none \
      --task "$task" --out "$W/expb_report_corpuswide.json"
  step completion-corpuswide-revealed "$W/expb_report_revealed_corpuswide.json" \
    "$LOCUS" exp completion --embeddings "$mm" --mode revealed --standardize none \
      --task "$task" --out "$W/expb_report_revealed_corpuswide.json"
  # node2vec in place of PPMI (appendix): no gain from a true seed, lower
  # recall in both self-seeded settings.
  need n2v-completion "$W/node2vec_p1q8_vectors.npy" "$W/sweep_val_node2vec.json" || return 0
  local n2v="--assoc node2vec --vectors $W/node2vec_p1q8 --sweep $W/sweep_val_node2vec.json"
  step n2v-completion-revealed "$W/expb_report_n2v_revealed_none.json" \
    "$LOCUS" exp completion --embeddings "$mm" --mode revealed --standardize none $n2v \
      --out "$W/expb_report_n2v_revealed_none.json"
  step n2v-completion-selfseed "$W/expb_report_n2v_selfseed_none.json" \
    "$LOCUS" exp completion --embeddings "$mm" --mode selfseed --standardize none $n2v \
      --out "$W/expb_report_n2v_selfseed_none.json"
  step n2v-completion-corpuswide "$W/expb_report_n2v_corpuswide_none.json" \
    "$LOCUS" exp completion --embeddings "$mm" --mode selfseed --standardize none $n2v \
      --task "$task" --out "$W/expb_report_n2v_corpuswide_none.json"
}

stage_theta() {
  # Everything downstream of the clustering is rebuilt per threshold in its
  # own work directory; vectors are reused from the published ones and only
  # the rows they lack are embedded. A sensitivity analysis, not a way to
  # choose theta.
  local t split dirs=""
  need theta "$W/embeddings_val.parquet" "$W/embeddings_test.parquet" "$W/sweep_val.json" \
    "$W/cocitation_train.npz" || return 0
  if [ -z "${LOCUS_DATA_DIR:-}" ] && [ "$DRY" = 0 ]; then
    echo "--- skip theta: LOCUS_DATA_DIR is not set"
    note skipped "theta (LOCUS_DATA_DIR not set)"
    return 0
  fi
  for t in 0.3 0.4 0.6 0.7; do
    local d="$W/theta_$t"
    mkdir -p "$d"
    step "theta-$t-pipeline-test" "$d/build_report_test.json" \
      env LOCUS_WORK_DIR="$d" "$LOCUS" build pipeline --split test --theta "$t" || continue
    OK_CODES="0 1"
    step "theta-$t-pipeline-val" "$d/build_report_val.json" \
      env LOCUS_WORK_DIR="$d" "$LOCUS" build pipeline --split val --theta "$t" || continue
    step "theta-$t-graph" "$d/cocitation_train.npz" \
      env LOCUS_WORK_DIR="$d" "$LOCUS" build graph --theta "$t" || continue
    for split in val test; do
      step "theta-$t-embed-inputs-$split" "$d/embed_inputs_$split.parquet" \
        env LOCUS_WORK_DIR="$d" "$LOCUS" build embed-inputs --split "$split" || continue 2
      step "theta-$t-embed-$split" "$d/embeddings_$split.parquet" \
        "$LOCUS" embed --input "$d/embed_inputs_$split.parquet" \
          --output "$d/embeddings_$split.parquet" --reuse "$W/embeddings_$split.parquet" || continue 2
    done
    step "theta-$t-sweep" "$d/sweep_val.json" \
      env LOCUS_WORK_DIR="$d" "$LOCUS" sweep --embeddings "$d/embeddings_val.parquet" \
        --graph "$d/cocitation_train.npz" --out "$d/sweep_val.json" || continue
    dirs="$dirs --dir $t=$d"
  done
  [ -n "$dirs" ] || return 0
  # shellcheck disable=SC2086
  step theta-sensitivity "$W/theta_sensitivity.json" \
    "$LOCUS" exp theta $dirs --dir "0.5=$W" \
      --fixed-graph "$W/cocitation_train.npz" --fixed-sweep "$W/sweep_val.json" \
      --out "$W/theta_sensitivity.json"
}

stage_export() {
  local split
  for split in test val; do
    need "export-$split" "$W/index_$split.pkl" "$W/rank_items_$split.pkl" || continue
    step "export-$split" "$W/export/locus_manifest_$split.json" \
      "$LOCUS" build export --split "$split" || continue
    # --verify checks an existing export; it does not write one.
    step "export-verify-$split" "$W/logs/export-verify-$split.ok" \
      "$LOCUS" build export --split "$split" --verify \
      && [ "$DRY" = 0 ] && touch "$W/logs/export-verify-$split.ok"
  done
}

echo "work directory: $W"
echo "corpus:         ${LOCUS_DATA_DIR:-<LOCUS_DATA_DIR not set>}"
[ "$DRY" = 0 ] && "$LOCUS" doctor

for stage in $STAGES; do
  echo
  echo "##### $stage"
  "stage_$stage"
done

echo
echo "##### summary"
printf '%s' "$SUMMARY"
echo "reports: $W/*.json    logs: $W/logs/"
exit "$FAILED"
