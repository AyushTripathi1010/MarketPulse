"""
Synthetic seed corpus for the hybrid retriever.

This is a small set of plausible historical (ticker, date, regime, outcome,
news_summary) tuples used for:
  - End-to-end tests of the retriever without depending on real S3 data.
  - Local development demos before the indexer has been run.

Each entry is HAND-WRITTEN to exercise different retrieval modes:
  - Some have exact ticker symbols (BM25-friendly).
  - Some use paraphrased event language (semantic-friendly).
  - Cover all three regimes (bull / bear / sideways).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeedRow:
    """A row in the synthetic corpus (mirror of IndexedDoc fields)."""

    ticker: str
    occurred_at: str  # ISO datetime
    regime_then: str  # 'bull' | 'bear' | 'sideways'
    outcome_24h_return: float
    news_summary: str


# Twenty hand-curated synthetic setups across our supported tickers.
# Picked to span regimes and to mix ticker-heavy vs prose-heavy text
# so the retriever's BM25 and semantic sides have plausible work to do.
SEED_CORPUS: tuple[SeedRow, ...] = (
    SeedRow("AAPL", "2024-03-12T09:30:00", "bull", 0.018,
            "AAPL Q1 earnings beat — services revenue up 14% YoY; analyst targets raised across the board."),
    SeedRow("AAPL", "2023-11-02T09:30:00", "sideways", -0.003,
            "AAPL guides cautiously for the holiday quarter despite strong iPhone 15 demand; supply chain remains a concern."),
    SeedRow("AAPL", "2024-08-05T09:30:00", "bear", -0.029,
            "Broad tech selloff hits AAPL; macroeconomic fears around rate cuts pressure mega-caps."),
    SeedRow("AAPL", "2025-01-15T09:30:00", "bull", 0.024,
            "AAPL announces new Vision Pro variants; pre-orders exceed internal forecasts."),

    SeedRow("MSFT", "2024-04-30T09:30:00", "bull", 0.027,
            "MSFT Azure growth accelerates to 31% YoY; AI workloads cited as primary driver."),
    SeedRow("MSFT", "2024-10-30T09:30:00", "sideways", 0.004,
            "MSFT meets but doesn't beat consensus; cloud margin expansion stalls quarter-over-quarter."),
    SeedRow("MSFT", "2023-09-12T09:30:00", "bear", -0.021,
            "Broad market repricing on inflation surprise; MSFT drops with the QQQ basket."),

    SeedRow("NVDA", "2024-02-21T09:30:00", "bull", 0.071,
            "NVDA blowout earnings; data center revenue triples; CEO Jensen Huang on AI compute demand."),
    SeedRow("NVDA", "2024-08-28T09:30:00", "bull", 0.034,
            "NVDA H200 production update; cloud customers commit multi-billion-dollar orders."),
    SeedRow("NVDA", "2025-02-25T09:30:00", "sideways", -0.006,
            "NVDA earnings in line but China revenue softness offsets US strength; mixed outlook."),

    SeedRow("GOOGL", "2024-04-25T09:30:00", "bull", 0.022,
            "GOOGL ad revenue rebounds; first-ever dividend announcement surprises the market."),
    SeedRow("GOOGL", "2023-07-25T09:30:00", "sideways", 0.002,
            "GOOGL search market share defended; Gemini progress disclosed but financial impact deferred."),

    SeedRow("TSLA", "2024-10-23T09:30:00", "bear", -0.058,
            "TSLA misses delivery target; price cuts in China weigh on margin; CEO commentary cautious."),
    SeedRow("TSLA", "2025-03-19T09:30:00", "bull", 0.041,
            "TSLA robotaxi unveiling exceeds expectations; investor day pushes valuation higher."),

    SeedRow("BTC-USD", "2024-01-11T09:30:00", "bull", 0.045,
            "Bitcoin spot ETF approval drives institutional flows; major asset managers launch products."),
    SeedRow("BTC-USD", "2024-08-05T09:30:00", "bear", -0.082,
            "Crypto cascade liquidations as macro risk-off triggers leveraged unwinds; BTC drops sharply."),
    SeedRow("BTC-USD", "2025-04-21T09:30:00", "sideways", -0.005,
            "Bitcoin range-bound around prior consolidation as ETF flows balance against profit-taking."),

    SeedRow("ETH-USD", "2024-05-22T09:30:00", "bull", 0.062,
            "Ethereum spot ETF approval signaled imminent; gas fee trends favorable as L2 adoption climbs."),
    SeedRow("ETH-USD", "2024-09-05T09:30:00", "bear", -0.041,
            "ETH/BTC ratio falls amid Solana rotation; staking yields compress."),
    SeedRow("ETH-USD", "2025-02-03T09:30:00", "sideways", 0.012,
            "Ethereum quiet on macro; developer activity steady, no major protocol upgrades pending."),
)


def build_embedding_text(row: SeedRow) -> str:
    """Compose the text we embed + tokenize for a corpus row.

    Including the ticker explicitly here matters — BM25 needs the exact
    symbol present, and embeddings benefit from the additional signal.
    """
    return f"{row.ticker} ({row.regime_then}): {row.news_summary}"
