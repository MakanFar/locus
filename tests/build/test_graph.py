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
        lambda seed, corpus=None: build_graph._digest_of_graph(
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
        monkeypatch.setattr(build_graph, "_run_second_val_build", lambda seed, corpus=None: "0" * 64)

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


class TestLeakyControlBuild:
    """--splits train,val,test builds the train-only-vs-full leakage control."""

    def _setup(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "WORK_DIR", tmp_path)
        monkeypatch.setattr(
            build_graph, "iter_papers",
            _fake_iter_papers(_train_papers(), _held_out_papers("V"),
                              _held_out_papers("S")),
        )
        _passthrough_child_digest(monkeypatch)

    def test_leakage_is_expected_and_not_a_failure(self, monkeypatch, tmp_path):
        # The whole point of this build is to include held-out papers, so the
        # leakage gate must report rather than fail -- otherwise the control
        # can never be produced.
        self._setup(monkeypatch, tmp_path)
        build_graph.main(["--splits", "train,val,test"])
        report = json.loads((tmp_path / "graph_report_full.json").read_text())
        assert report["leaky_by_design"] is True
        assert report["leakage_overlap"] > 0
        assert not any(f.startswith("LEAKAGE:") for f in report["failures"])

    def test_leaky_build_never_claims_the_canonical_filename(self, monkeypatch, tmp_path):
        # cocitation_train.npz is what every scorer loads by default. A leaky
        # graph landing there would silently poison every later result.
        self._setup(monkeypatch, tmp_path)
        build_graph.main(["--splits", "train,val,test"])
        assert not (tmp_path / "cocitation_train.npz").exists()

    def test_train_only_build_still_fails_on_leakage(self, monkeypatch, tmp_path):
        # The default path must keep its teeth: same fixture, but train now
        # includes a held-out paper, and the gate has to fire.
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
