import hashlib
import json
import pickle

from locus.core.pools import SwapPair, build_rank_items
from locus.core.types import ContextRec
from locus.data.export import export, load_rank
from locus.eval.verify import verify_export


def _rec(cid, refid, loc, markers=(), ok=True):
    return ContextRec(
        context_id=f"{cid}_{refid}_0", citing_id=cid, refid=refid,
        raw="some window text", masked="some window text",
        location=loc, alignment_ok=ok, marker_anchors=frozenset(markers),
    )


def _index(n_locations=6, per_location=2):
    recs, r = [], 0
    for loc in range(n_locations):
        for _ in range(per_location):
            recs.append(_rec("P", str(r), loc))
            r += 1
    return {"P": recs}


def _write_pickles(work, index, pool_size=5):
    items, _ = build_rank_items(index, pool_size=pool_size, seed=0)
    pairs = [SwapPair("P", "P_0_0", "P_2_0", "0", "2")]
    hard = sorted({it.context_id for it in items[:3]})
    headline = hard[:2]
    for name, obj in (("rank_items", items), ("swap_pairs", pairs),
                      ("hard_ids", hard), ("headline_ids", headline),
                      ("index", index)):
        (work / f"{name}_test.pkl").write_bytes(pickle.dumps(obj, protocol=5))
    return items, pairs, set(hard), set(headline)


class TestVerifyExport:
    def test_export_then_verify_passes(self, tmp_path):
        work, out = tmp_path / "work", tmp_path / "out"
        work.mkdir()
        _write_pickles(work, _index())
        m = export("test", work, out)
        assert m["counts"]["citing_papers"] == 1
        assert verify_export("test", work, out) == []

    def test_verify_catches_a_corrupted_file(self, tmp_path):
        work, out = tmp_path / "work", tmp_path / "out"
        work.mkdir()
        _write_pickles(work, _index())
        export("test", work, out)
        p = out / "locus_rank_test.jsonl"
        rows = p.read_text().splitlines()
        d = json.loads(rows[0])
        # Substitute a candidate rather than drop one: a size change is caught
        # earlier by the floor check, which would leave the checksum and
        # round-trip paths untested.
        d["candidates"][0] = "not-a-real-paper"
        p.write_text("\n".join([json.dumps(d), *rows[1:]]) + "\n")
        failures = verify_export("test", work, out)
        assert any("sha256" in f for f in failures)
        assert any("round-trip" in f for f in failures)

    def test_verify_reports_a_truncated_pool_instead_of_raising(self, tmp_path):
        work, out = tmp_path / "work", tmp_path / "out"
        work.mkdir()
        _write_pickles(work, _index())
        export("test", work, out)
        p = out / "locus_rank_test.jsonl"
        rows = p.read_text().splitlines()
        d = json.loads(rows[0])
        d["candidates"] = d["candidates"][:-1]
        p.write_text("\n".join([json.dumps(d), *rows[1:]]) + "\n")
        assert any("mixed sizes" in f for f in verify_export("test", work, out))

    def test_a_pool_off_the_floor_fails_but_a_rounded_mean_does_not(self, tmp_path):
        # The fixture's 12 pools each sit exactly on H_5/5 while their mean
        # does not, because the aggregator's final division rounds. That must
        # not be reported as a benchmark defect -- only a pool that is itself
        # off the floor is.
        work, out = tmp_path / "work", tmp_path / "out"
        work.mkdir()
        _write_pickles(work, _index(), pool_size=5)
        export("test", work, out)
        assert verify_export("test", work, out) == []
        floor = sum(1.0 / i for i in range(1, 6)) / 5
        from locus.scoring.baselines import constant_scorer, rank_mrr
        mrr, values, _ = rank_mrr(
            load_rank(out / "locus_rank_test.jsonl"), constant_scorer(1.0))
        assert set(values) == {floor} and mrr != floor

    def test_verify_catches_a_dropped_anchor(self, tmp_path):
        work, out = tmp_path / "work", tmp_path / "out"
        work.mkdir()
        _write_pickles(work, _index())
        export("test", work, out)
        p = out / "locus_anchors_test.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        rows[0]["anchors"] = []
        m = json.loads((out / "locus_manifest_test.json").read_text())
        with open(p, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        # Re-stamp the checksum so the anchor check is what fails, not the hash.
        m["files"]["locus_anchors_test.jsonl"]["sha256"] = hashlib.sha256(
            p.read_bytes()).hexdigest()
        (out / "locus_manifest_test.json").write_text(json.dumps(m, indent=2))
        failures = verify_export("test", work, out)
        assert any("anchor sets differ on 1" in f for f in failures)

    def test_verify_catches_a_floor_that_does_not_match_the_pools(self, tmp_path):
        work, out = tmp_path / "work", tmp_path / "out"
        work.mkdir()
        _write_pickles(work, _index(), pool_size=5)
        export("test", work, out)
        mp = out / "locus_manifest_test.json"
        m = json.loads(mp.read_text())
        m["floors"]["rank_constant_mrr"] = repr(
            sum(1.0 / i for i in range(1, 11)) / 10
        )
        mp.write_text(json.dumps(m, indent=2))
        assert any("is not the floor" in f for f in verify_export("test", work, out))
