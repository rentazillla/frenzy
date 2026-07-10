from __future__ import annotations

from frenzy.properties import compute, compute_many


def test_compute_returns_values():
    r = compute("CCO")
    assert r.MW > 0
    assert r.LogP is not None
    assert r.HBD >= 0
    assert 0.0 <= r.QED <= 1.0


def test_compute_many():
    results = compute_many(["CCO", "c1ccccc1"])
    assert len(results) == 2
    assert "MW" in results[0]
    assert "QED" in results[1]