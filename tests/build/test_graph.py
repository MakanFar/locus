import json

import pytest

from locus import config
from locus.build import graph as build_graph
from locus.core.types import ContextRec


def test_main_fails_fast_on_an_empty_corpus(monkeypatch, tmp_path):
    # No real dataset access: iter_papers is monkeypatched to yield nothing,
    # so nothing is streamed from contexts.json.
    monkeypatch.setattr(build_graph, "iter_papers", lambda *a, **k: iter(()))
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    with pytest.raises(SystemExit):
        build_graph.main([])


def _rec(cid, refid, loc):
    return ContextRec(
        context_id=f"{cid}_{refid}_{loc}", citing_id=cid, refid=refid,
        raw="", masked="", location=loc,
    )


def _train_papers():
    return [
        ("T1", [_rec("T1", "A", 0), _rec("T1", "B", 0)]),
        ("T2", [_rec("T2", "A", 0), _rec("T2", "C", 0)]),
    ]


def _held_out_papers(tag):
    return [(f"{tag}1", [_rec(f"{tag}1", "X", 0), _rec(f"{tag}1", "Y", 0)])]


def _fake_iter_papers(train, val, test):
    data = {"train": train, "val": val, "test": test}

    def _f(split, keep_text=True, **kwargs):
        yield from data[split]

    return _f


def _passthrough_child_digest(monkeypatch):
    """Make the gate-5 'child process' step reuse the (monkeypatched)
    iter_papers in-process instead of spawning a real subprocess, so tests
    that don't care about the determinism gate specifically are fast and
    are not sensitive to whatever tiny fixture data they use."""
    monkeypatch.setattr(
        build_graph,
        "_run_second_val_build",
        lambda seed, corpus=None, theta=None: build_graph._digest_of_graph(
            build_graph.count_sites(build_graph.iter_papers("val", keep_text=False))
        ),
    )


class TestNpzWriteGating:
    def test_npz_is_not_written_when_a_gate_fails(self, monkeypatch, tmp_path):
        # A tiny fixture like this always misses the 1.6M-6.4M pair-instance
        # sanity band, so failures is always non-empty here -- exactly the
        # case where the old code still wrote the artefact.
        monkeypatch.setattr(config, "WORK_DIR", tmp_path)
        monkeypatch.setattr(
            build_graph, "iter_papers",
            _fake_iter_papers(_train_papers(), _held_out_papers("V"), _held_out_papers("S")),
        )
        _passthrough_child_digest(monkeypatch)

        rc = build_graph.main([])

        assert rc == 1
        report = json.loads((tmp_path / "graph_report.json").read_text())
        assert report["failures"]
        assert not (tmp_path / "cocitation_train.npz").exists()


class TestLeakageGateOnEmptyHeldOut:
    def test_empty_held_out_is_reported_as_unevaluated_not_as_a_pass(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "WORK_DIR", tmp_path)
        monkeypatch.setattr(
            build_graph, "iter_papers",
            _fake_iter_papers(_train_papers(), [], []),
        )
        _passthrough_child_digest(monkeypatch)

        build_graph.main([])

        report = json.loads((tmp_path / "graph_report.json").read_text())
        assert report["held_out_papers"] == 0
        assert any(
            "could not be evaluated" in f and "leakage" in f.lower()
            for f in report["failures"]
        )
        assert not any(f.startswith("LEAKAGE:") for f in report["failures"])


class TestDeterminismGateAcrossProcesses:
    def test_digest_mismatch_between_parent_and_child_fails_the_gate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "WORK_DIR", tmp_path)
        monkeypatch.setattr(
            build_graph, "iter_papers",
            _fake_iter_papers(_train_papers(), _held_out_papers("V"), _held_out_papers("S")),
        )
        monkeypatch.setattr(build_graph, "_run_second_val_build", lambda seed, corpus=None, theta=None: "0" * 64)

        rc = build_graph.main([])

        assert rc == 1
        report = json.loads((tmp_path / "graph_report.json").read_text())
        assert report["val_determinism"] is False
        assert any("not deterministic" in f for f in report["failures"])
        assert not (tmp_path / "cocitation_train.npz").exists()

    def test_matching_digests_pass_the_gate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "WORK_DIR", tmp_path)
        monkeypatch.setattr(
            build_graph, "iter_papers",
            _fake_iter_papers(_train_papers(), _held_out_papers("V"), _held_out_papers("S")),
        )
        _passthrough_child_digest(monkeypatch)

        build_graph.main([])

        report = json.loads((tmp_path / "graph_report.json").read_text())
        assert report["val_determinism"] is True


class TestDigestOfGraph:
    def test_same_content_gives_the_same_digest(self):
        g1 = build_graph.count_sites(_train_papers())
        g2 = build_graph.count_sites(_train_papers())
        assert build_graph._digest_of_graph(g1) == build_graph._digest_of_graph(g2)

    def test_different_content_gives_a_different_digest(self):
        g1 = build_graph.count_sites(_train_papers())
        g2 = build_graph.count_sites(_held_out_papers("V"))
        assert build_graph._digest_of_graph(g1) != build_graph._digest_of_graph(g2)


class TestRunSecondValBuildSubprocess:
    """Exercises the real subprocess wiring (sys.executable, env,
    PYTHONHASHSEED) end to end, against a tiny on-disk fixture -- not the
    real ~3GB dataset -- so the plumbing itself is covered, not just the
    comparison logic."""

    def _write_fixture(self, data_dir, val_refids):
        (data_dir / "contexts.json").write_text(json.dumps({
            f"c{i}": {"citing_id": "V1", "refid": r, "raw": "x", "masked_text": "x"}
            for i, r in enumerate(val_refids)
        }))
        (data_dir / "val.json").write_text(
            json.dumps([{"context_id": f"c{i}"} for i in range(len(val_refids))])
        )
        (data_dir / "train.json").write_text(json.dumps([]))
        (data_dir / "test.json").write_text(json.dumps([]))

    def test_spawns_a_real_subprocess_and_returns_a_stable_hex_digest(self, monkeypatch, tmp_path):
        self._write_fixture(tmp_path, ["A", "B"])
        monkeypatch.setenv("LOCUS_DATA_DIR", str(tmp_path))

        got1 = build_graph._run_second_val_build("111")
        got2 = build_graph._run_second_val_build("222")

        assert got1 == got2  # same data, different child hashseed -> same digest
        assert len(got1) == 64
        int(got1, 16)  # valid hex

    def test_different_val_data_gives_a_different_digest(self, monkeypatch, tmp_path):
        self._write_fixture(tmp_path, ["A", "B"])
        monkeypatch.setenv("LOCUS_DATA_DIR", str(tmp_path))
        got1 = build_graph._run_second_val_build("111")

        self._write_fixture(tmp_path, ["A", "C"])
        got2 = build_graph._run_second_val_build("111")

        assert got1 != got2


class TestPeakRssGb:
    def test_darwin_treats_ru_maxrss_as_bytes(self, monkeypatch):
        monkeypatch.setattr(build_graph.sys, "platform", "darwin")
        monkeypatch.setattr(
            build_graph.resource, "getrusage",
            lambda who: type("R", (), {"ru_maxrss": 2**30})(),
        )
        assert build_graph._peak_rss_gb() == pytest.approx(1.0)

    def test_linux_treats_ru_maxrss_as_kilobytes(self, monkeypatch):
        monkeypatch.setattr(build_graph.sys, "platform", "linux")
        monkeypatch.setattr(
            build_graph.resource, "getrusage",
            lambda who: type("R", (), {"ru_maxrss": 2**20})(),
        )
        assert build_graph._peak_rss_gb() == pytest.approx(1.0)

    def test_unknown_platform_raises_systemexit_rather_than_guessing(self, monkeypatch):
        monkeypatch.setattr(build_graph.sys, "platform", "freebsd13")
        monkeypatch.setattr(
            build_graph.resource, "getrusage",
            lambda who: type("R", (), {"ru_maxrss": 2**20})(),
        )
        with pytest.raises(SystemExit):
            build_graph._peak_rss_gb()


def test_a_build_that_counts_a_held_out_paper_fails_the_leakage_gate(monkeypatch, tmp_path):
    # The graph is train-only by construction; if the train stream ever
    # carries a held-out paper, the gate has to fire and no .npz is written.
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    leaky_train = _train_papers() + _held_out_papers("V")
    monkeypatch.setattr(
        build_graph, "iter_papers",
        _fake_iter_papers(leaky_train, _held_out_papers("V"), _held_out_papers("S")),
    )
    _passthrough_child_digest(monkeypatch)
    rc = build_graph.main([])
    assert rc == 1
    report = json.loads((tmp_path / "graph_report.json").read_text())
    assert any(f.startswith("LEAKAGE:") for f in report["failures"])
    assert not (tmp_path / "cocitation_train.npz").exists()


def test_theta_flag_reaches_the_train_stream(monkeypatch, tmp_path):
    # The graph is built from train LOCATIONS, so a theta sensitivity analysis
    # that rebuilt only the test index would compare test locations at one
    # threshold against a graph of train locations at another.
    seen = []

    def spy(split, **kw):
        seen.append((split, kw.get("theta")))
        return iter(())

    monkeypatch.setattr(build_graph, "iter_papers", spy)
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    with pytest.raises(SystemExit, match="0 papers streamed"):
        build_graph.main(["--theta", "0.3"])
    assert seen == [("train", 0.3)]
