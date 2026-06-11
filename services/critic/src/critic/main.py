"""
critic — FastAPI entry point.

Phase 3: /critique endpoint that grades a Forecast's confidence using Groq.
Phase 4: optional Phi-3 regime classifier (HF Inference API).
Phase 5: hybrid RAG retrieval of historical analogues — Qdrant semantic +
         BM25 lexical + RRF fusion + optional cross-encoder rerank. The
         retrieved analogues are fed BOTH to the Groq judge prompt and
         returned to the orchestrator on the Critique payload.

Every external dep is OPTIONAL:
  - No GROQ_API_KEY → stub Critique with confidence=UNKNOWN
  - No HF_TOKEN     → no Phi-3 classifier, no semantic retrieval, no rerank;
                      retriever degrades to BM25-only on the seed corpus
  - No Qdrant       → use in-memory mode (:memory:), still works
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from marketplus_shared.models import HealthResponse, HistoricalAnalogue, Regime

from critic.classifier.phi3_regime import (
    MarketSnapshot,
    RegimeClassifierError,
    classify as classify_regime,
)
from critic.config import settings
from critic.judge.groq_critic import CriticLLMError, judge, stub_critique
from critic.rag.bm25_index import BM25Index
from critic.rag.hybrid_retriever import HybridRetriever
from critic.rag.indexer import populate_from_seed
from critic.rag.qdrant_store import QdrantStore
from critic.schemas import CritiqueRequest, CritiqueResponse

logger = logging.getLogger(__name__)

# Module-level state holders populated by lifespan().
# Keeping them here (vs a class) matches FastAPI convention and lets the
# tests monkey-patch them surgically.
_qdrant: QdrantStore | None = None
_bm25: BM25Index | None = None
_retriever: HybridRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the RAG index once at startup; tear nothing down.

    Cost on startup:
      - Qdrant init: ~10 ms (in-memory mode) or one HTTP round-trip.
      - BM25 build:  ~few ms for the 20-row seed corpus.
      - Embedding:   ~N HF API calls when HF_TOKEN is set. Skipped otherwise.

    So a no-key startup is essentially instant. A keyed startup pays the
    embedding cost once and amortizes it across every subsequent /critique.
    """
    global _qdrant, _bm25, _retriever

    logger.info(
        "critic starting. groq=%s phi3=%s rag_qdrant=%s reranker=%s",
        bool(settings.groq_api_key),
        bool(settings.hf_token),
        settings.qdrant_url,
        settings.use_reranker,
    )
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY not set; /critique returns stub responses.")
    if not settings.hf_token:
        logger.warning(
            "HF_TOKEN not set; Phi-3 classifier disabled AND RAG runs BM25-only."
        )

    # Build the RAG index.
    _qdrant = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    _bm25 = BM25Index()
    if settings.seed_index_on_startup:
        populate_from_seed(
            _qdrant,
            _bm25,
            hf_api_key=settings.hf_token or None,
            embed_model_id=settings.rag_embed_model_id,
        )

    _retriever = HybridRetriever(
        _qdrant,
        _bm25,
        hf_api_key=settings.hf_token,
        embed_model_id=settings.rag_embed_model_id,
        reranker_model_id=settings.rag_reranker_model_id,
        use_reranker=settings.use_reranker,
    )

    yield


app = FastAPI(
    title="MarketPulse — Critic",
    version="0.1.0",
    description="Critic Agent: regime classifier + hybrid RAG + Groq judgment.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        service="critic",
        extras={
            "environment": settings.environment,
            "llm_configured": bool(settings.groq_api_key),
            "phi3_classifier_configured": bool(settings.hf_token),
            "rag_corpus_size": len(_bm25) if _bm25 else 0,
            "rag_qdrant_count": _qdrant.count() if _qdrant else 0,
        },
    )


def _maybe_classify_regime(snapshot_dict: dict | None) -> Regime | None:
    """Run the Phi-3 classifier when both creds and snapshot are present."""
    if not settings.hf_token or snapshot_dict is None:
        return None

    try:
        snapshot = MarketSnapshot(
            ticker=str(snapshot_dict.get("ticker", "")),
            return_5d=float(snapshot_dict.get("return_5d", 0.0)),
            return_20d=float(snapshot_dict.get("return_20d", 0.0)),
            vol_20d=float(snapshot_dict.get("vol_20d", 0.0)),
            rsi_14=float(snapshot_dict.get("rsi_14", 50.0)),
            sentiment_24h=float(snapshot_dict.get("sentiment_24h", 0.0)),
        )
    except (TypeError, ValueError) as e:
        logger.warning("Phi-3: bad snapshot fields, skipping classifier: %s", e)
        return None

    try:
        return classify_regime(
            snapshot,
            hf_api_key=settings.hf_token,
            model_id=settings.phi3_model_id,
        )
    except RegimeClassifierError as e:
        logger.warning("Phi-3 classifier unavailable, continuing without: %s", e)
        return None


def _build_retrieval_query(
    forecast_ticker: str,
    snapshot_dict: dict | None,
    news: list[str],
) -> str:
    """Compose the natural-language query the retriever uses for analogues.

    Strategy: ticker first (BM25 anchor), then the snapshot's key signals
    in plain English (semantic anchor), then a few news headlines.

    We include the ticker so BM25 has an exact-match anchor — the dominant
    signal for "find me similar setups for THIS ticker." We include the
    snapshot numbers in human-readable form (e.g. "5d return: +2.5%") so
    the embedding captures the directional + volatility flavor of the
    setup. News headlines add event context.
    """
    parts: list[str] = [forecast_ticker]
    if snapshot_dict:
        r5 = float(snapshot_dict.get("return_5d", 0.0))
        r20 = float(snapshot_dict.get("return_20d", 0.0))
        v20 = float(snapshot_dict.get("vol_20d", 0.0))
        rsi = float(snapshot_dict.get("rsi_14", 50.0))
        parts.append(
            f"5d return {r5:+.2%}, 20d return {r20:+.2%}, "
            f"20d vol {v20:.3f}, RSI {rsi:.0f}"
        )
    if news:
        # Cap at 5 headlines to keep the query embedding tight.
        parts.extend(news[:5])
    return ". ".join(parts)


def _retrieve_analogues(
    forecast_ticker: str,
    snapshot_dict: dict | None,
    news: list[str],
) -> list[HistoricalAnalogue]:
    """Run hybrid RAG. Returns [] on any failure — never raises."""
    if _retriever is None:
        return []

    query = _build_retrieval_query(forecast_ticker, snapshot_dict, news)
    try:
        return _retriever.retrieve(
            query,
            top_k_each=settings.rag_top_k_each,
            top_k_final=settings.rag_top_k_final,
            ticker_filter=forecast_ticker,  # narrow semantic search to this ticker
        )
    except Exception as e:  # noqa: BLE001  -- retriever is best-effort
        logger.warning("RAG retrieval failed; continuing without analogues: %s", e)
        return []


@app.post("/critique", response_model=CritiqueResponse)
def critique(req: CritiqueRequest) -> CritiqueResponse:
    """Grade a Forecast end-to-end (Phase 5 flow).

    Pipeline:
      1. (Phase 4) Phi-3 regime classification.
      2. (Phase 5) Hybrid RAG retrieval of historical analogues.
      3. (Phase 3) Groq judge sees forecast + news + regime hint + analogues
         and produces HIGH/MEDIUM/LOW confidence + plain-English reasoning.
      4. Critique returned to orchestrator (analogues included so the
         frontend can display them).
    """
    if not settings.groq_api_key:
        # Stub mode — but we STILL include analogues if RAG is configured,
        # so the frontend can show "here's what the model would have used"
        # even in degraded mode.
        stub = stub_critique(req.forecast)
        analogues = _retrieve_analogues(req.forecast.ticker, req.snapshot, req.recent_news)
        return stub.model_copy(update={"analogues": analogues})

    regime_hint = _maybe_classify_regime(req.snapshot)
    analogues = _retrieve_analogues(req.forecast.ticker, req.snapshot, req.recent_news)

    try:
        critique_result = judge(
            req.forecast,
            req.recent_news,
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            regime_hint=regime_hint,
            analogues=analogues,
        )
    except CriticLLMError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Critic LLM error: {e}",
        ) from e

    # The judge currently returns Critique with analogues=[] (Phase 3 stub).
    # Overlay the analogues we retrieved so the response is complete.
    return critique_result.model_copy(update={"analogues": analogues})
