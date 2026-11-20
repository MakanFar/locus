# Annotation protocol: are two citations at the same citation site?

Version 0.4 (2026-11-10). Two annotators. Sections 1 to 8 are the protocol
as frozen before the main round; section 9 reports the run. The sample the
annotators label is drawn by `locus exp annotation-sample`
(`docs/reproduce.md`, "The annotation sample"); its cell populations are in
`reports/annotation_manifest.json`, and `locus exp annotation-score` turns
the labels into section 6.

## 1. What this study is for

LOCUS infers, for every citation in a paper, which other citations sit at the
same *site*. It does this without ground truth, by two routes:

- **clustering**: two context windows from the same paper whose 8-word
  shingles overlap with Jaccard ≥ 0.5 are treated as one location;
- **markers**: any other citation marker inside a citation's ~450-character
  window that resolves to a paper is treated as co-located.

Reviewers asked how often these inferences are right. This study answers that
with human labels: precision and recall of each route, and of their union,
against two definitions of "same site".

It does **not** test whether the cited papers are relevant, correct, or well
chosen. Annotators judge only where citations sit relative to each other in
the text.

## 2. Definitions

Three nested tiers. Every item gets exactly one label.

**G — same citation group** (primary definition of a citation site).
Two citations are in the same group when they are offered *jointly for the
same span of text*, with nothing between their markers except punctuation,
conjunctions, a range sign, or other citation markers.

- `... angular margin losses [9, 10, 11] ...` → 9, 10, 11 are one group.
- `... dominant OTHERCIT TARGETCIT OTHERCIT , fundamentally ...` → one group.
- `(Smith, 2020; Jones, 2021)` → one group. `[3]–[7]` → one group.
- `Smith et al. [3] and Jones et al. [4] showed that ...` → one group: the two
  markers share one predicate and only the cited works' own names and "and"
  intervene.

**S — same sentence, different group.**
Both markers are in one sentence but attach to different spans of it.

- `... in electronics [8], optoelectronics [9, 10], optics [11, 12] and
  integrated on chip [17] ...` → 9 and 10 are G; 9 and 11 are S.
- `X improves accuracy [1, 2], while Y degrades it [3].` → 1 and 3 are S.

**W — different sentences, same window.**
The markers are in different sentences. This includes adjacent sentences on
the same topic.

**N — not co-located / cannot be determined.**
The second marker is not visible in the text shown, or the two windows shown
do not overlap, or the text is too damaged to judge. Use the comment column to
say which.

### Decision rules

1. Find both highlighted markers. If either is missing, label N.
2. Read the text strictly between them. If it contains only punctuation,
   "and", "or", "&", a dash or range sign, other citation markers, or the
   author names that belong to those markers, label G.
3. Otherwise, if no sentence ends between them, label S.
4. Otherwise label W.

Notes for the boundary cases:

- A sentence ends at a full stop, question mark or exclamation mark followed
  by a new sentence. Abbreviations (`et al.`, `Fig.`, `Eq.`, `i.e.`, `e.g.`,
  `cf.`, `vs.`), decimal points and equation punctuation do not end a
  sentence.
- A semicolon or colon does not end a sentence.
- Displayed equations and stripped mathematics are common in this corpus. If
  the sentence boundary cannot be recovered, label by your best reading and
  tick the *uncertain* box.
- A marker that opens a sentence (`[12] showed that ...`) belongs to that
  sentence, not the previous one.
- If the same paper is cited twice in the text shown, judge the nearest
  occurrence to the target marker.
- Do not use your knowledge of the papers. Two works on the same topic in
  adjacent sentences are W, however obviously related.

## 3. What an item looks like

One row per (target citation, candidate citation) pair from one citing paper.

| Column | Content |
|---|---|
| `item_id` | opaque id (`L001` … `L400`; pilot `P01` … `P40`); the order is randomised per annotator |
| `text` | the target's window. The target marker is shown as `⟦T⟧`, the candidate marker as `⟦C⟧` when it can be located in this window, and every other citation marker as `[cit]`. Reference numbers the corpus did not mask stay as printed |
| `text_2` | only when `⟦C⟧` could not be located in `text`: the candidate's own window, with `⟦C⟧` marked (and `⟦T⟧` when it can be located there) |
| `label` | G / S / W / N |
| `uncertain` | tick if you would not be surprised to be overruled |
| `comment` | free text |

When an item has a `text_2`, the two windows usually share a passage. Find the
shared passage, work out which `[cit]` in `text` is the candidate (or which in
`text_2` is the target), and then label as usual. If the windows share no
passage, label N.

*Added in version 0.4, after the main round (§9):* `⟦T⟧` in `text_2` marks
every citation of the target paper, and a paper cited more than once may
show a different occurrence there from the one in `text`. The item is the
occurrence in `text`. Judge only that one: if `text_2`'s `⟦T⟧` is another
occurrence, label the pair by where the candidate sits relative to the
target's own marker, W when it is in the target's window in another
sentence and N when the two windows share no passage.

Annotators do **not** see: which route proposed the pair, the Jaccard
similarity, paper titles, or the other annotator's labels.

## 4. Sample

The sample has to cover the pipeline's likely merges *and* its likely splits,
not only pairs near the threshold. The unit is an ordered pair: a context and
another paper its citing paper cites. The test split holds 1,722,838 such
pairs, and every one falls in exactly one stratum.

| # | Stratum | What it tests | pairs in split | n |
|---|---|---|---|---|
| 1 | Anchor by both routes | precision where the routes agree | 186,936 | 60 |
| 2 | Anchor by markers only (marker resolves in the window, different cluster) | marker-route precision; what the clustering splits off | 41,322 | 80 |
| 3 | Anchor by clustering only; the window aligned but the candidate's marker does not resolve in it | clustering precision where the markers disagree | 2,803 | 40 |
| 4 | Anchor by clustering only; the window failed alignment, so clustering is the only evidence | clustering precision on its unverifiable cases | 18,200 | 60 |
| 5 | Same cluster only transitively (the two windows are below the threshold and joined through a third) | over-merging by union-find chaining | 20,883 | 60 |
| 6 | Not an anchor; windows overlap with Jaccard in [0.2, 0.5) | recall: near-miss splits | 19,367 | 60 |
| 7 | Not an anchor; Jaccard in (0, 0.2) | recall further from the threshold | 43,597 | 20 |
| 8 | Not an anchor; windows share nothing | sanity: should be all N | 1,388,346 | 20 |

Total 400 items, at most two per citing paper, drawn with a fixed seed.
Stratum 5 takes precedence over 1, 3 and 4. Strata 1 and 2 are split by the
automatic proximity tier (see §7) so that both likely-G and likely-W pairs are
present: 20 / 20 / 20 over same group / same sentence / other sentence in
stratum 1, and 10 / 35 / 35 in stratum 2, where a marker-only anchor inside
the target's own group is an anomaly (231 pairs in the split). 1,384 pairs
are in no cell (both-route or marker-only anchors in a window whose target
did not validate) and are reported as unsampled.

Changes from version 0.2, made when the strata were counted: the old stratum
"clustering only, marker visible in the window" is empty by construction (a
marker that resolves in the window *is* a marker-route anchor), so strata 3
and 4 now split clustering-only anchors by whether the window aligned; and
non-anchors with Jaccard in (0, 0.2), which the old design did not sample,
have their own stratum.

Population estimates are post-stratified: each cell's rate is weighted by the
cell's population, which the manifest records, with bootstrap intervals
resampling citing papers.

The sample is generated by

    locus exp annotation-sample --out work/annotation

which writes `sheet_A.csv` and `sheet_B.csv` (the same 400 items in different
orders), `pilot_sheet.csv` (40 items, disjoint from the main sample),
`key.csv` and `pilot_key.csv` (stratum, routes, Jaccard, automatic tier and
cell population per item; not for annotators), and `manifest.json`. The draw
is deterministic. The sheets contain corpus text and stay out of the
repository.

## 5. Procedure

1. **Pilot (40 items, discarded).** Both annotators label the same 40 items
   from a separate draw, compare, and discuss every disagreement. The guide in
   §2 is revised once on the basis of the pilot and then frozen. Record what
   changed.
2. **Main round (400 items).** Each annotator labels all items independently,
   in a different random order, without discussion.
3. **Agreement** is computed on the independent labels, before any
   reconciliation.
4. **Adjudication.** The annotators meet, look only at the items they
   disagreed on, and agree a consensus label. If they cannot, a third reader
   decides. Record how many items needed the third reader.
5. **Scoring** uses the consensus labels.

Expected effort: about 15 seconds an item, so under two hours each for the
main round, plus an hour for pilot and adjudication.

## 6. What is reported

- **Agreement**: raw agreement and Cohen's κ on the four-way label, and on the
  two binary collapses (G vs rest; G-or-S vs rest), with the confusion matrix.
- **Pipeline against consensus**, under each definition (site = group; site =
  sentence): precision, recall and F1 for the clustering route, the marker
  route, and their union, post-stratified, with 95% intervals.
- **Error anatomy**: how many false-positive anchors each route adds and how
  many true ones it misses, per 100 contexts.
- **Effect on the result**: base and rescored MRR on the pools touched by
  annotated false positives and by annotated false negatives. This is the
  direct test of whether clustering error inflates or deflates the reported
  gain; it replaces any inference from Δ across θ.
- **Share labelled uncertain or N**, per stratum.
- The annotator instructions (this document, as frozen after the pilot), the
  item files, both annotators' raw labels and the consensus labels are
  released with the benchmark. The item files contain corpus text and are
  released only where the upstream licence permits; otherwise ids and labels.

## 7. What we already know without annotators

An automatic proxy for these tiers (regular expressions over the masked
window, no human judgement; `locus exp proximity`, report
`proximity_specter2.json`) gives, for the anchors the pipeline currently uses
on the test split:

| Anchor instances | same group | same sentence | different sentence | not in window |
|---|---|---|---|---|
| marker route (244,345) | 44.4% | 30.7% | 24.4% | 0.6% |
| clustering route (229,952) | 47.0% | 28.8% | 11.9% | 12.3% |

and the reranker's gain on the headline pools when it is given the anchors of
one tier only (SPECTER2, λ = 3, means over citing papers; paper Table 7):

| Anchors used | pools | Δ MRR | 95% CI |
|---|---|---|---|
| same group only | 22,218 | +0.190 | [+0.180, +0.199] |
| same sentence only | 14,432 | +0.079 | [+0.069, +0.091] |
| different sentence only | 14,724 | +0.010 | [+0.000, +0.019] |
| not in window only | 1,954 | +0.086 | [+0.061, +0.114] |

On the 22,218 pools that have a same-group anchor, adding the looser tiers
lowers the gain: +0.190 (group) → +0.179 (+ sentence) → +0.165 (+ different
sentence).

The annotation study validates this proxy as well: the tier the regular
expressions assign is stored with each item (hidden from annotators) and its
agreement with the consensus label is reported. Label W corresponds to the
proxy's "different sentence" tier and label N to "not in window".

## 8. Open choices for the authors

- Who the two annotators are, and who the third reader is.
- The scoring is `locus exp annotation-score` (`docs/reproduce.md`,
  "Scoring the labels").
- Whether a subsample of 50 items is additionally checked against the citing
  paper's PDF, to verify that the pipeline resolved the candidate marker to
  the right paper. This is the only way to measure marker-resolution error and
  costs about a minute an item.
- Whether narrative citations joined by "and" (the `Smith [3] and Jones [4]`
  case) count as one group. The guide says yes; if the authors prefer a
  strictly typographic definition (same bracket only), rule 2 changes and the
  pilot should be rerun.

## 9. Results (2026-11-10)

Two authors labelled, in a shared web page that showed one item at a time
and nothing else, saved each label as it was made, and kept the other
reader's labels hidden until both had finished. Committed with the key as
`reports/annotation_labels.csv` and `reports/annotation_pilot_labels.csv`
(ids and labels only); scored into `reports/annotation_score.json`.

**Pilot.** 40 items, raw agreement 0.75, κ 0.62. Ten disagreements, of three
kinds: enumerated lists where a topic word sits between the two markers (one
reader G, the guide says S); sentence boundaries in damaged text; and items
whose second window shares no passage with the first (one reader W, the guide
says N). They were discussed and the guide text in §2 was not changed.

**Main round.** 400 items, no uncertain ticks, no comments.

| | raw agreement | κ |
|---|---|---|
| four-way label | 0.860 | 0.792 |
| G vs rest | 0.963 | 0.892 |
| G-or-S vs rest | 0.930 | 0.860 |

56 items disagreed and all were settled by the two readers; none went to a
third reader. The largest disagreement classes were N against W (15 items,
nearly all in the disjoint stratum), S against W (19) and G against S (9).
Four of the settled items were then re-labelled under the occurrence rule
added to §3 (see below); the committed consensus is the result.

Consensus labels by stratum:

| Stratum | G | S | W | N |
|---|---|---|---|---|
| both routes | 23 | 18 | 19 | 0 |
| marker only | 10 | 34 | 36 | 0 |
| clustering only, aligned | 26 | 10 | 3 | 1 |
| clustering only, unaligned | 31 | 19 | 10 | 0 |
| transitive | 1 | 27 | 32 | 0 |
| near miss | 0 | 8 | 52 | 0 |
| weak overlap | 0 | 1 | 18 | 1 |
| disjoint | 0 | 0 | 2 | 18 |

**Pipeline against consensus**, post-stratified, 95% paper-level bootstrap
intervals:

| Site | Route | Precision | Recall | F1 |
|---|---|---|---|---|
| group | clustering | 0.553 [0.521, 0.593] | 0.998 [0.998, 0.998] | 0.712 |
| group | marker | 0.472 [0.444, 0.510] | 0.909 [0.890, 0.927] | 0.621 |
| group | union | 0.469 [0.442, 0.503] | 1.000 [1.000, 1.000] | 0.639 |
| sentence | clustering | 0.852 [0.833, 0.870] | 0.935 [0.909, 0.950] | 0.891 |
| sentence | marker | 0.759 [0.745, 0.772] | 0.889 [0.863, 0.906] | 0.819 |
| sentence | union | 0.754 [0.739, 0.770] | 0.977 [0.951, 0.993] | 0.851 |

The pipeline's errors are over-inclusion, not misses, and the over-inclusion
is almost entirely "same sentence, different group": under the sentence
definition three anchors in four are right, under the group definition about
half. Per 100 contexts (104,401 in the split) the union adds 137 false
anchors and misses none under the group definition; 64 and 4.6 under the
sentence definition. The marker route alone misses 11 and 22, which the
clustering route recovers.

**The occurrence rule.** As first adjudicated, four non-anchor items were
labelled G or S: one in the disjoint stratum (windows share no shingle;
1,388,346 pairs, 20 sampled, so the item stood for 69,417 pairs), two in
weak-overlap and one in near-miss. In each, the target paper is cited more
than once in the citing paper and the second window showed a *different*
occurrence of it next to the candidate; the target's own occurrence is in
another sentence or another passage, and the other occurrence is its own
context, for which the candidate is an anchor. The protocol had not said
which occurrence to judge. The authors adopted the rule now in §3 (the
target's own occurrence only), under which the four items are N, W, W, W.
Precision is unchanged by this; recall under the sentence definition, which
had been 0.690 [0.440, 0.937] for clustering, 0.656 [0.420, 0.891] for
markers and 0.721 [0.459, 0.980] for the union with the one disjoint item
carrying the interval, is as tabled above.

**Proxy.** The regular-expression tier of §7 matches the consensus on 182 of
the 186 items where it assigns a tier (group 30 of 30, sentence 78 of 81,
different sentence 74 of 75). The `not_in_window` tier is not a judgement:
on those 214 items the readers used the second window and found the pair in
the same group 58 times, the same sentence 39, a different sentence 97, and
not co-located 20.

**Effect on the result.** 181 of the 400 target contexts are in the headline
pools. On the 102 pools an annotated false positive touches (site = group),
the reranker's gain is +0.155 with the pipeline's anchors and +0.158 with
that anchor removed; on the 50 pools under the sentence definition, +0.047
and +0.048. Annotated false negatives touch no pool under the group
definition and three under the sentence definition, where adding the pair
changes nothing. Removing the anchors the readers reject does not lower the
gain.

**Open.** §8's question on "and"-joined narrative citations, which the pilot
raised (one item) and the guide still answers with G.
