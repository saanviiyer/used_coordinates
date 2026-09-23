"""The p-values in Section 3 come from here, so they are checked here."""
import re

import numpy as np
import pytest

from make_numbers import _horizon_para, plane_stats, sign_p


def test_sign_p_matches_hand_computed_exact_values():
    assert sign_p(8, 8) == pytest.approx(2 / 2 ** 8)      # 0.0078125
    assert sign_p(7, 8) == pytest.approx(2 * 9 / 2 ** 8)  # 0.0703125
    assert sign_p(0, 8) == pytest.approx(2 / 2 ** 8)      # symmetric


def test_sign_p_is_symmetric_and_bounded():
    for k in range(9):
        assert sign_p(k, 8) == pytest.approx(sign_p(8 - k, 8))
        assert 0 < sign_p(k, 8) <= 1.0
    assert sign_p(4, 8) == 1.0


def _row(resid, null_median, null_min, kappa=0.5):
    return {"planes": [{"residual": resid,
                        "null_residual_median": null_median,
                        "null_residual_min": null_min,
                        "best_kappa": kappa,
                        "best_basis": [1, 0]}]}


def test_strict_count_is_stricter_than_beats():
    # Below the null's median but above its minimum: counted once, not twice.
    rows = [_row(0.5, 0.6, 0.4) for _ in range(4)]
    st = plane_stats(rows, 0)
    assert st["beats"] == 4
    assert st["strict"] == 0
    assert st["gap"] == pytest.approx(0.1)


def test_gap_and_p_track_the_rows():
    rows = [_row(0.1, 0.9, 0.8) for _ in range(8)]
    st = plane_stats(rows, 0)
    assert st["beats"] == st["strict"] == 8
    assert st["gap"] == pytest.approx(0.8)
    assert st["p"] == pytest.approx(sign_p(8, 8))


def test_horizon_para_is_silent_until_a_horizon_is_complete():
    # A partial sweep must not produce a sentence describing fewer seeds.
    rows = [{"untrained": False, "horizon": 4, "model_kind": "gru",
             "tag": f"s{i}",
             "planes": [{"residual": 0.1, "null_residual_median": 0.9},
                        {"residual": 0.8, "null_residual_median": 0.9}]}
            for i in range(3)]
    assert _horizon_para({"rows": rows}) == ""
    assert _horizon_para(None) == ""


def test_horizon_para_reports_a_complete_horizon():
    rows = [{"untrained": False, "horizon": 4, "model_kind": "gru",
             "tag": f"s{i}",
             "planes": [{"residual": 0.1, "null_residual_median": 0.9},
                        {"residual": 0.8, "null_residual_median": 0.9}]}
            for i in range(8)]
    out = _horizon_para({"rows": rows})
    assert "horizon" in out.lower()
    assert "0.10" in out and "0.80" in out


def _hrow(seed, horizon, kind, r1=0.1, r2=0.8):
    return {"untrained": False, "horizon": horizon, "model_kind": kind,
            "tag": f"arm_{kind}_dark_s{seed}",
            "planes": [{"residual": r1, "null_residual_median": 0.9},
                       {"residual": r2, "null_residual_median": 0.9}]}


def test_horizon_para_ignores_the_ungated_architecture():
    # Only the GRU cleared the blackout gate. A complete modular horizon must
    # not make an incomplete GRU horizon look reportable, and modular numbers
    # must never enter the average.
    rows = ([_hrow(i, 4, "modular", r1=0.5, r2=0.5) for i in range(8)]
            + [_hrow(i, 4, "gru") for i in range(3)])
    assert _horizon_para({"rows": rows}) == ""

    rows += [_hrow(i, 4, "gru") for i in range(3, 8)]
    out = _horizon_para({"rows": rows})
    assert "0.10" in out and "0.80" in out
    # 0.50 would be a modular value quoted directly; 0.30 would be the mean of
    # the two architectures. Neither may appear.
    assert "0.50" not in out and "0.30" not in out


def test_horizon_para_lists_only_the_horizons_it_reports():
    # Horizon 8 is half-finished, so it must not appear in the prose list.
    rows = ([_hrow(i, 4, "gru") for i in range(8)]
            + [_hrow(i, 8, "gru") for i in range(4)])
    out = _horizon_para({"rows": rows})
    # Assert on the horizon list the sentence names, however it is phrased.
    listed = re.search(r"At ([0-9, ]+) steps", out).group(1)
    assert listed == "4", f"half-finished horizon 8 was listed: {listed!r}"
