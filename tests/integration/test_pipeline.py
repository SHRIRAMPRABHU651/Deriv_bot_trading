"""research.pipeline / charts / download_universe on synthetic data and the mock broker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.deriv.websocket import DerivWebSocket
from research import charts, download_ticks, download_universe, pipeline
from tests.mocks.deriv_mock import MockDeriv
from tests.unit.test_backtest import series


def write_csv(path: Path, phi: float, seed: int, n: int = 12_000) -> None:
    epochs, prices = series(n, phi, seed)
    np.savetxt(
        path,
        np.c_[epochs, prices],
        delimiter=",",
        header="epoch,quote",
        comments="",
        fmt=["%d", "%.6f"],
    )


def test_pipeline_writes_reports_and_never_approves_a_random_walk(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "reports"
    data.mkdir()
    write_csv(data / "RANDOM_300s.csv", 0.0, 3)
    write_csv(data / "TRENDY_300s.csv", 0.8, 4)
    rc = pipeline.main(
        ["--skip-download", "--data-dir", str(data), "--out", str(out), "--capital", "20"]
    )
    assert rc == 0
    index = (out / "index.html").read_text()
    assert "RANDOM_300s" in index and "TRENDY_300s" in index
    page = (out / "RANDOM_300s.html").read_text()
    assert "<polyline" in page and "consistent with a random walk" in page
    assert "REJECTED" in page and "APPROVED" not in page
    assert "APPROVED" in (out / "TRENDY_300s.html").read_text()
    assert "<script" not in page  # static page, no external code


def test_charts_escape_market_names(tmp_path: Path) -> None:
    data, out = tmp_path / "d", tmp_path / "o"
    data.mkdir()
    write_csv(data / "a<b>.csv", 0.0, 1, n=500)
    charts.build_all(data, out)
    assert "<b>" not in (out / "index.html").read_text().replace("<b>", "", 0).split("<td>")[1]


def written(path: Path) -> list[str]:
    return sorted(f.name for f in path.glob("*.csv"))


async def test_universe_download_continues_after_a_failing_symbol(
    mock: MockDeriv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = download_ticks.download

    async def flaky(
        ws: DerivWebSocket,
        symbol: str,
        total: int,
        chunk: int = 5000,
        granularity: int | None = None,
    ) -> list[tuple[int, float]]:
        if symbol == "BAD":
            raise RuntimeError("market closed")
        return await real(ws, symbol, total, chunk, granularity)

    monkeypatch.setattr(download_universe, "download", flaky)
    rc = await download_universe.run(
        ["BAD", "frxEURUSD"], months=0.002, granularity=60, out_dir=tmp_path, ws_url=mock.url
    )
    assert rc == 0  # one symbol worked
    assert written(tmp_path) == ["frxEURUSD_60s.csv"]
