import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

from data.loader import (load_data,load_multi_timeframe_data)
from models.trainer import(train_model,ensemble_predict)
from news.api import fetch_news
from news.sentiment import analyze_overall_sentiment
from utils.helpers import (
    prepare_data, run_backtest,
    show_chart, show_metrics, show_prediction, show_candlestick_chart,
)
from utils.stock_search import load_stock_data
from utils.decision_engine import generate_signal
from features.engineer import get_trend_signal
from utils.regime import detect_regime
from utils.risk import calculate_risk
from utils.explainability import build_recommendation_explanation, build_card_summary
from scanner.cache import load_category_cache, cache_age_minutes, any_cache_exists
from scanner.background import (
    start_background_scan, is_scan_running, scan_progress, needs_scan
)
from storage.tracker import (
    save_signal, get_recent_signals, get_accuracy_stats,
    save_recommendation,
)
from utils.logger import (
    get_logger, configure_logging,
    read_last_log_lines, clear_log_file, log_file_size_kb,
)
from config import ENABLE_DEBUG_LOGS

# Initialise logging once — safe to call on every Streamlit rerun
configure_logging(debug=ENABLE_DEBUG_LOGS)
_app_logger = get_logger(__name__)
_app_logger.info("App started / reloaded")
from storage.recommendation_validation import (
    validate_old_recommendations,
    load_validated_recommendations,
    migrate_schema as _migrate_validation_schema,
)
_migrate_validation_schema()   # ensure validation columns exist on every start

from storage.performance_analytics import (
    load_validated_df,
    summary_metrics,
    signal_performance,
    confidence_performance,
    confluence_performance,
    sentiment_performance,
    monthly_trend,
    top_winners,
    top_losers,
    generate_insights,
)

from config import CATEGORIES

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockAI Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #ffffff; }
.stApp [data-testid="stMarkdownContainer"] * { color: #ffffff; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 60%, #0f3460 100%);
    border: 1px solid #30363d; border-radius: 16px;
    padding: 1.6rem 2rem 1.4rem; margin-bottom: 0.8rem;
}
.hero-title { font-size: 2rem; font-weight: 800; color: #f0f6fc; margin: 0 0 .2rem; letter-spacing: -.5px; }
.hero-sub   { color: #ffffff; font-size: .9rem; margin: 0; }
.hero-badge { display:inline-block; margin-top:.7rem; background:#1c2128; border:1px solid #ffa028; color:#ffa028; font-size:.68rem; padding:.16rem .5rem; border-radius:20px; }

.scan-badge-running { background:#0d2b1e; border:1px solid #238636; color:#3fb950; display:inline-block; padding:.2rem .7rem; border-radius:20px; font-size:.75rem; font-weight:600; }
.scan-badge-stale   { background:#2b1d00; border:1px solid #bb8009; color:#d29922; display:inline-block; padding:.2rem .7rem; border-radius:20px; font-size:.75rem; font-weight:600; }
.scan-badge-fresh   { background:#0d1b2e; border:1px solid #1f6feb; color:#58a6ff; display:inline-block; padding:.2rem .7rem; border-radius:20px; font-size:.75rem; font-weight:600; }

.sec-title { font-size:.8rem; font-weight:700; color:#ffffff; text-transform:uppercase; letter-spacing:1px; margin:1.1rem 0 .65rem; padding-bottom:.3rem; border-bottom:1px solid #21262d; }

.cap-header { font-size:1rem; font-weight:700; color:#f0f6fc; margin:.5rem 0 .8rem; padding:.5rem .9rem; background:#161b22; border-radius:8px; border-left:3px solid #58a6ff; }
.cap-header-mid   { border-left-color: #a371f7; }
.cap-header-small { border-left-color: #3fb950; }

.pick-card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:.9rem .8rem; text-align:center; transition:border-color .2s,transform .15s; }
.pick-card:hover { border-color:#58a6ff; transform:translateY(-2px); }
.pick-rank   { color:#ffffff; font-size:.65rem; font-weight:600; letter-spacing:.5px; }
.pick-name   { color:#ffffff; font-size:.83rem; font-weight:700; margin:.28rem 0 .06rem; line-height:1.2; }
.pick-symbol { color:#ffffff; font-size:.67rem; }
.pick-badge-strong-buy { display:inline-block; margin-top:.4rem; background:#0a2e1a; color:#4ade80; border:2px solid #22c55e; padding:.11rem .7rem; border-radius:20px; font-size:.75rem; font-weight:800; }
.pick-badge-buy  { display:inline-block; margin-top:.4rem; background:#0d2b1e; color:#3fb950; border:1px solid #238636; padding:.11rem .6rem; border-radius:20px; font-size:.75rem; font-weight:700; }
.pick-badge-sell { display:inline-block; margin-top:.4rem; background:#2d0c0c; color:#f85149; border:1px solid #da3633; padding:.11rem .6rem; border-radius:20px; font-size:.75rem; font-weight:700; }
.pick-badge-hold { display:inline-block; margin-top:.4rem; background:#2b1d00; color:#d29922; border:1px solid #bb8009; padding:.11rem .6rem; border-radius:20px; font-size:.75rem; font-weight:700; }
.pick-meta { color:#ffffff; font-size:.67rem; margin-top:.4rem; line-height:1.7; }

div[data-testid="metric-container"] { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:.65rem .9rem; }
div[data-testid="stExpander"] { border:1px solid #30363d !important; border-radius:8px !important; background:#161b22 !important; }
hr { border-color:#21262d; }

button[data-testid="baseButton-secondary"],
button[kind="secondary"] {
    background-color: #000000 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}
button[data-testid="baseButton-secondary"]:hover,
button[kind="secondary"]:hover {
    background-color: #1a3a5c !important;
    border-color: #58a6ff !important;
    color: #58a6ff !important;
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD STOCK DATABASE ──────────────────────────────────────────────────────
try:
    stocks_df = load_stock_data()
    stocks_df.columns = stocks_df.columns.str.strip()
except Exception as e:
    st.error(f"Failed to load stock database: {e}")
    st.stop()

if stocks_df.empty:
    st.error("No stocks found in database.")
    st.stop()

company_map: dict = dict(zip(
    stocks_df["Symbol"].str.strip(),
    stocks_df["Company"].str.strip(),
))

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">📈 StockAI Pro</div>
  <div class="hero-sub">ML + FinBERT sentiment · NIFTY Large / Mid / Small Cap universe · NSE</div>
  <span class="hero-badge">⚠️ Experimental model &nbsp;·&nbsp; Not financial advice</span>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_analyse, tab_home, tab_tracker, tab_performance = st.tabs([
    "🔍  Analyse Stock",
    "🏆  Top Picks",
    "📋  My Tracker",
    "📊  Performance",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:

    scanning = is_scan_running()

    # ── Auto-refresh every 20 s while a scan is running or cache is empty ─────
    if scanning or not any_cache_exists():
        st_autorefresh(interval=20_000, limit=90, key="scan_autorefresh")

    # ── Scan status banner ────────────────────────────────────────────────────
    progress = scan_progress()
    status_col, refresh_col = st.columns([5, 1])

    with status_col:
        _err = progress.get("category", "")
        if isinstance(_err, str) and _err.startswith("error:"):
            st.markdown(
                f'<span class="scan-badge-stale">⚠️ Scan failed: {_err[6:].strip()} — click Refresh to retry</span>',
                unsafe_allow_html=True,
            )
        elif scanning:
            cat   = progress.get("category", "stocks")
            done  = progress.get("done", 0)
            total = progress.get("total", 0)
            pct   = f"{done}/{total}" if total else "starting…"
            st.markdown(
                f'<span class="scan-badge-running">'
                f'🔄 Scanning {cat} — {pct} &nbsp;·&nbsp; page refreshes automatically'
                f'</span>',
                unsafe_allow_html=True,
            )
        elif not any_cache_exists():
            st.markdown(
                '<span class="scan-badge-stale">⏳ No scan data yet — starting first scan…</span>',
                unsafe_allow_html=True,
            )
        else:
            ages   = [cache_age_minutes(c) for c in CATEGORIES]
            ages   = [a for a in ages if a is not None]
            oldest = max(ages) if ages else None
            label  = f"Last scanned {oldest:.0f} min ago" if oldest else "Cache loaded"
            st.markdown(
                f'<span class="scan-badge-fresh">✅ {label} · auto-refreshes hourly</span>',
                unsafe_allow_html=True,
            )

    with refresh_col:
        force_refresh = st.button("🔄 Refresh", width="stretch")

    # ── Start background scan if needed ───────────────────────────────────────
    if force_refresh or needs_scan():
        start_background_scan(company_map)
        if force_refresh:
            st.toast("Scan started — dashboard updates every 20 s automatically.", icon="🔄")

    # ── Overall KPI row ───────────────────────────────────────────────────────
    all_recs: list[dict] = []
    for cat in CATEGORIES:
        recs = load_category_cache(cat) or []
        for r in recs:
            r["category"] = cat
        all_recs.extend(recs)

    st.markdown('<div class="sec-title">📊 Market overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Stocks analysed", len(all_recs))
    k2.metric("📈 BUY",  sum(1 for r in all_recs if r["signal"] in ("STRONG BUY", "BUY")))
    k3.metric("📉 SELL", sum(1 for r in all_recs if r["signal"] in ("STRONG SELL", "SELL")))
    k4.metric("⏸️ HOLD", sum(1 for r in all_recs if r["signal"] == "HOLD"))

    buy_candidate = [
        r for r in all_recs
        if r["signal"] in  ("STRONG BUY","BUY")
    ]

    for r in buy_candidate:
        r["rank_score"]=(
            r["score"]
              * (r["confidence"] / 100)
              * (r["accuracy"] / 100)
        )
      

    best = max(
        buy_candidate,
        key=lambda r: r.get("rank_score",r["score"]),
         default=None
    )

    if best:
        k5.metric("Top pick", best["symbol"])
    else:
        k5.metric("Top pick","No BUY")

    # ── Per-category sections ─────────────────────────────────────────────────
    CAP_COLORS  = {"Large Cap": "#58a6ff", "Mid Cap": "#a371f7", "Small Cap": "#3fb950"}
    CAP_ICONS   = {"Large Cap": "🏦", "Mid Cap": "🏢", "Small Cap": "🌱"}
    CAP_HEADERS = {
        "Large Cap": "cap-header",
        "Mid Cap":   "cap-header cap-header-mid",
        "Small Cap": "cap-header cap-header-small",
    }

    for category in CATEGORIES:
        cat_recs = load_category_cache(category) or []
        buy_recs = [r for r in cat_recs if r["signal"] in ("STRONG BUY", "BUY")][:5]

        icon = CAP_ICONS[category]
        css  = CAP_HEADERS[category]
        age  = cache_age_minutes(category)
        age_str = f"· {age:.0f} min ago" if age else "· no data yet"

        st.markdown(
            f'<div class="{css}">{icon} {category} &nbsp;<span style="color:#6e7681;font-size:.75rem;font-weight:400">'
            f'{len(cat_recs)} scanned &nbsp;{age_str}</span></div>',
            unsafe_allow_html=True,
        )

        if not cat_recs:
            st.caption("⏳ Scan running in background — check back in a few minutes.")
            continue

        if buy_recs:
            cols = st.columns(min(len(buy_recs), 5))
            for i, (col, rec) in enumerate(zip(cols, buy_recs)):
                with col:
                    sig = rec['signal']
                    if sig == "STRONG BUY":
                        badge = '<span class="pick-badge-strong-buy">🚀 STRONG BUY</span>'
                    else:
                        badge = '<span class="pick-badge-buy">📈 BUY</span>'
                    regime_tag = f'<br/>Regime <b>{rec.get("regime","—")}</b>' if rec.get("regime") else ""
                    st.markdown(f"""
                        <div class="pick-card">
                        <div class="pick-rank">#{i+1} TOP PICK</div>
                        <div class="pick-name">{rec['stock']}</div>
                        <div class="pick-symbol">{rec['symbol']}</div>
                        <div>{badge}</div>
                        <div class="pick-meta">
                            Score <b>{round(rec['score']*100,0):.0f}/100</b><br/>
                            Conf <b>{rec['confidence']}%</b><br/>
                            Acc <b>{rec['accuracy']}%</b><br/>
                            Weekely <b>{rec.get('weekly_trend','-')}<br/>
                            Daily <b>{rec.get('daily_trend','-')}<br/>
                            ₹ <b>{rec.get('close','—')}</b>{regime_tag}
                        </div>
                        </div>""", unsafe_allow_html=True)

                    with st.expander("🧠 Why?", expanded=False):
                        try:
                            card_exp = build_card_summary(rec)
                            st.caption(card_exp["signal_explanation"])
                            if card_exp.get("reason"):
                                st.markdown(f"**{card_exp['reason']}**")
                            factors_list = card_exp.get("factors", [])
                            if factors_list:
                                for f in factors_list[:5]:
                                    st.markdown(f"- {f}")
                            st.caption(card_exp.get("risk_summary", ""))
                            for wp in card_exp.get("watch_points", []):
                                st.caption(f"👀 {wp}")
                        except Exception:
                            st.caption("Explanation unavailable for this pick.")
        else:
            st.caption("No BUY signals passed quality filters in this category.")

        # Collapsible full table for this category
        if cat_recs:
            with st.expander(f"All {category} results ({len(cat_recs)} stocks)", expanded=False):
                df = pd.DataFrame(cat_recs)[[
                    "stock", "symbol", "signal", "score",
                    "confidence", "accuracy", "model", "close",
                ]]
                df.columns = ["Company", "Symbol", "Signal", "Score",
                              "Conf %", "Acc %", "Model", "Close ₹"]
                icons = {"BUY": "📈 BUY", "SELL": "📉 SELL", "HOLD": "⏸️ HOLD"}
                df["Signal"] = df["Signal"].map(icons).fillna(df["Signal"])
                df = df.sort_values("Score", ascending=False).reset_index(drop=True)
                df.index += 1
                st.dataframe(df, width="stretch")

    if not any_cache_exists():
        st.info(
            "First-time setup: background scan has been started. "
            "The dashboard will populate automatically — this takes 3–5 minutes. "
            "You can use the **🔍 Analyse Stock** tab in the meantime."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSE STOCK
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyse:

    st.markdown('<div class="sec-title">🔍 Deep-dive any stock</div>', unsafe_allow_html=True)

    search_col, btn_col = st.columns([5, 1])
    with search_col:
        selected_company = st.selectbox(
            "Stock",
            stocks_df["Company"].dropna().unique(),
            index=0,
            label_visibility="collapsed",
        )
    with btn_col:
        predict_clicked = st.button("🚀 Analyse", type="primary", width="stretch")

    stock_symbol = stocks_df.loc[
        stocks_df["Company"] == selected_company, "Symbol"
    ].iloc[0]

    st.caption(f"Symbol: **{stock_symbol}**")

    # ── Clear cached results if user selects a different stock ────────────────
    if st.session_state.get("_analysed_symbol") != stock_symbol:
        for key in ["_analysis_result", "_analysed_symbol"]:
            st.session_state.pop(key, None)

    # ── Run analysis only when button is clicked ──────────────────────────────
    if predict_clicked:
        st.divider()
        result = {}

        with st.spinner(f"Loading {stock_symbol}…"):
            try:
                data = load_data(stock_symbol)
            except Exception as e:
                st.error(f"Data load failed: {e}")
                st.stop()

        if data.empty:
            st.error("❌ No price data found.")
            st.stop()

        try:
            data, X, y, _, _, y_train, _ = prepare_data(data)
        except Exception as e:
            st.error(f"Feature engineering failed: {e}")
            st.stop()

        _model_key = f"_model_{stock_symbol}"
        if _model_key in st.session_state:
            models, acc = st.session_state[_model_key]
        else:
            with st.spinner("Training model (walk-forward validation)…"):
                try:
                    models, acc = train_model(X, y)
                    st.session_state[_model_key] = (models, acc)
                except Exception as e:
                    st.error(f"Model training failed: {e}")
                    st.stop()

        model_name = "Ensemble"

        try:
            pred, confidence, prob = ensemble_predict(models, X.tail(1))
        except AttributeError:
            confidence = 0.0

        try:
            data = run_backtest(data, models["Random Forest"], X)
        except Exception as e:
            st.warning(f"Backtest skipped: {e}")

        try:
            headlines = fetch_news(stock_symbol)
        except Exception:
            headlines = []

        try:
            overall_sentiment, overall_score, headline_results, sentiment_counts = analyze_overall_sentiment(headlines)
        except Exception:
            overall_sentiment, overall_score, headline_results = "Neutral", 0.0, []
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

        try:
            regime_info = detect_regime(data)
        except Exception:
            regime_info = None

        try:
            multi_tf_data = load_multi_timeframe_data(stock_symbol)
            weekly_trend = get_trend_signal(multi_tf_data["weekly"])
            daily_trend = get_trend_signal(multi_tf_data["daily"])
            raw_tf_score = (weekly_trend["score"] + daily_trend["score"])
            timeframe_score = raw_tf_score / 2
        except Exception:
            weekly_trend = {"trend": "UNKNOWN", "score": 0}
            daily_trend = {"trend": "UNKNOWN", "score": 0}
            timeframe_score = 0

        try:
            final_signal, final_score, reason, factors = generate_signal(
                prediction=int(pred[0]) if hasattr(pred, "__len__") else int(pred),
                confidence=confidence,
                news_score=overall_score,
                timeframe_score=timeframe_score,
                data=data,
                regime_info=regime_info,
            )
        except Exception as e:
            st.error(f"Signal generation failed:{e}")
            final_signal, final_score, reason, factors = "HOLD", 0.0, "Error", []

        try:
            risk = calculate_risk(data, final_signal)
        except Exception:
            risk = None

        close_price = float(data["Close"].iloc[-1])

        # ── Build the explanation panel (purely additive — never breaks the page) ──
        try:
            explanation = build_recommendation_explanation(
                symbol=stock_symbol,
                stock_name=selected_company,
                signal=final_signal,
                score=final_score,
                confidence=confidence,
                accuracy=acc,
                prediction=int(pred[0]) if hasattr(pred, "__len__") else int(pred),
                news_score=overall_score,
                timeframe_score=timeframe_score,
                regime_info=regime_info,
                factors=factors,
                risk=risk,
                data=data,
            )
        except Exception as e:
            st.warning(f"Explanation unavailable: {e}")
            explanation = None

        try:
            save_signal(stock_symbol, selected_company, final_signal,
                        final_score, confidence, acc, close_price)
        except Exception:
            pass

        try:
            save_recommendation(
                symbol           = stock_symbol,
                stock            = selected_company,
                signal           = final_signal,
                cmp              = close_price,
                confluence_score = final_score,
                ml_confidence    = confidence,
                news_score       = overall_score,
                accuracy         = acc,
                target           = risk.get("target")    if risk else None,
                stop_loss        = risk.get("stop_loss") if risk else None,
            )
        except Exception:
            pass

        # ── Persist everything to session state ───────────────────────────────
        st.session_state["_analysed_symbol"] = stock_symbol
        st.session_state["_analysis_result"] = dict(
            data=data, X=X, models=models, acc=acc,
            model_name=model_name, pred=pred, confidence=confidence,
            overall_sentiment=overall_sentiment, overall_score=overall_score,
            headline_results=headline_results, headlines=headlines,
            sentiment_counts=sentiment_counts,
            regime_info=regime_info, weekly_trend=weekly_trend,
            daily_trend=daily_trend, timeframe_score=timeframe_score,
            final_signal=final_signal, final_score=final_score,
            reason=reason, factors=factors, risk=risk,
            close_price=close_price, explanation=explanation,
        )

    # ── Render results from session state (survives autorefresh reruns) ───────
    if "_analysis_result" in st.session_state:
        r = st.session_state["_analysis_result"]

        # Unpack
        data            = r["data"]
        X               = r["X"]
        models          = r["models"]
        acc             = r["acc"]
        model_name      = r["model_name"]
        confidence      = r["confidence"]
        overall_sentiment  = r["overall_sentiment"]
        overall_score      = r["overall_score"]
        headline_results   = r["headline_results"]
        headlines          = r["headlines"]
        sentiment_counts   = r.get("sentiment_counts", {"positive": 0, "neutral": 0, "negative": 0})
        weekly_trend    = r["weekly_trend"]
        daily_trend     = r["daily_trend"]
        timeframe_score = r["timeframe_score"]
        final_signal    = r["final_signal"]
        final_score     = r["final_score"]
        reason          = r["reason"]
        factors         = r["factors"]
        risk            = r["risk"]
        close_price     = r["close_price"]
        explanation     = r.get("explanation")

        st.divider()
        chart_col, signal_col = st.columns([3, 2])

        with chart_col:
            try:
                show_candlestick_chart(data)
                show_chart(data)
                show_metrics(data)
            except Exception as e:
                st.error(f"Chart error: {e}")
            m1, m2 = st.columns(2)
            trade_count = (
                int((data["Strategy_Return"] != 0).sum())
                if "Strategy_Return" in data.columns else 0
            )
            m1.metric("Trades executed", trade_count)
            m2.metric("Latest close",    f"₹{round(close_price, 2)}")

        with signal_col:
            try:
                show_prediction(confidence, acc, model_name,
                                final_signal, final_score, reason,
                                factors=factors, risk=risk)
            except Exception as e:
                st.error(f"Signal display error: {e}")

            # ── Explainability panel ──────────────────────────────────────────
            if explanation:
                with st.expander("🧠 Why this recommendation?", expanded=False):
                    st.markdown(f"**{explanation.get('final_interpretation', '')}**")
                    st.caption(explanation.get("signal_explanation", ""))
                    if explanation.get("confidence_note"):
                        st.caption(explanation["confidence_note"])

                    st.markdown("---")

                    sw_col1, sw_col2 = st.columns(2)
                    with sw_col1:
                        st.markdown("**✅ Strengths**")
                        strengths = explanation.get("strengths", [])
                        if strengths:
                            for s in strengths:
                                st.markdown(f"- {s}")
                        else:
                            st.caption("No standout strengths identified.")
                    with sw_col2:
                        st.markdown("**⚠️ Weaknesses**")
                        weaknesses = explanation.get("weaknesses", [])
                        if weaknesses:
                            for w in weaknesses:
                                st.markdown(f"- {w}")
                        else:
                            st.caption("No notable weaknesses identified.")

                    st.markdown("---")
                    st.markdown("**📊 Pillar Contribution**")
                    pillar_rows = explanation.get("pillar_breakdown", [])
                    if pillar_rows:
                        pillar_df = pd.DataFrame(pillar_rows)
                        pillar_df = pillar_df.rename(columns={
                            "pillar": "Pillar",
                            "score": "Raw Score",
                            "impact": "Impact",
                            "weight": "Weight",
                            "weighted_contribution": "Weighted Contribution",
                            "explanation": "Explanation",
                        })

                        def _impact_colour(val):
                            colours = {
                                "positive": "color: #3fb950",
                                "negative": "color: #f85149",
                                "neutral":  "color: #d29922",
                            }
                            return colours.get(val, "")

                        try:
                            styled = pillar_df.style.applymap(
                                _impact_colour, subset=["Impact"]
                            )
                            st.dataframe(styled, width="stretch", hide_index=True)
                        except Exception:
                            st.dataframe(pillar_df, width="stretch", hide_index=True)
                    else:
                        st.caption("Pillar breakdown not available.")

                    st.markdown("---")
                    st.markdown("**💰 Risk Summary**")
                    st.caption(explanation.get("risk_summary", "Risk data unavailable."))

                    watch_points = explanation.get("watch_points", [])
                    if watch_points:
                        st.markdown("---")
                        st.markdown("**👀 Watch Points**")
                        for wp in watch_points:
                            st.markdown(f"- {wp}")

            st.markdown(
                '<div class="sec-title">⏱️ Multi-timeframe analysis</div>',
                unsafe_allow_html=True
            )
            t1, t2, t3 = st.columns(3)
            t1.metric("Weekly Trend",      weekly_trend["trend"])
            t2.metric("Daily Trend",       daily_trend["trend"])
            t3.metric("Confluence Score",  timeframe_score)

            st.markdown('<div class="sec-title">📰 Market sentiment</div>', unsafe_allow_html=True)

            s1, s2 = st.columns(2)
            s1.metric("Mood",      overall_sentiment)
            s2.metric("Avg score", round(overall_score, 2))

            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Positive News", sentiment_counts.get("positive", 0))
            c2.metric("🟡 Neutral News",  sentiment_counts.get("neutral",  0))
            c3.metric("🔴 Negative News", sentiment_counts.get("negative", 0))

            if headline_results:
                with st.expander("Latest news", expanded=True):
                    for item in headline_results:
                        s   = item.get("sentiment", "Neutral")
                        ico = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}.get(s, "⚪")
                        st.markdown(f"**{ico} {item.get('headline', '')}**")
                        st.caption(f"{s} · {round(item.get('score', 0), 2)}")
                        st.divider()
            elif not headlines:
                st.info("No recent news found.")

        st.success("Signal saved — view it in the **📋 My Tracker** tab.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MY TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with tab_tracker:

    st.markdown('<div class="sec-title">📋 Saved prediction signals</div>', unsafe_allow_html=True)

    correct, total = get_accuracy_stats()
    if total:
        t1, t2, t3 = st.columns(3)
        t1.metric("Predictions tracked", total)
        t2.metric("Outcomes validated",  correct)
        t3.metric("Validated accuracy",  f"{round(correct/total*100,1)}%")
    else:
        st.info("No signals saved yet. Run an analysis to start tracking.")

    signals = get_recent_signals(limit=30)
    if signals:
        df = pd.DataFrame(signals)
        icons = {"BUY": "📈 BUY", "SELL": "📉 SELL", "HOLD": "⏸️ HOLD"}
        df["Signal"]  = df["Signal"].map(icons).fillna(df["Signal"])
        df["Correct"] = df["Correct"].map(
            lambda v: "✅" if v == 1 else ("❌" if v == 0 else "—")
        )
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.caption("No signals saved yet.")

    st.divider()

    # ── Recommendation Validation ─────────────────────────────────────────────
    st.markdown('<div class="sec-title">🎯 Recommendation Validation (5-Day)</div>', unsafe_allow_html=True)
    st.caption("Validates recommendations that are at least 5 trading days old.")

    if st.button("🔄 Validate Old Recommendations", use_container_width=True):
        with st.spinner("Fetching prices and validating recommendations..."):
            try:
                count = validate_old_recommendations()
                if count > 0:
                    st.success(f"✅ Validated {count} recommendation{'s' if count != 1 else ''}.")
                else:
                    st.info("No recommendations ready for validation yet (need 5 trading days).")
            except Exception as _ve:
                st.error(f"Validation error: {_ve}")

    validated_rows = load_validated_recommendations(limit=50)
    if validated_rows:
        st.markdown("**Recent validated recommendations**")
        vdf = pd.DataFrame(validated_rows)

        # Format columns for display
        vdf["Success"] = vdf["Success"].map(
            lambda v: "✅ Success" if v == 1 else ("❌ Failed" if v == 0 else "—")
        )
        vdf["Return %"] = vdf["Return %"].apply(
            lambda v: f"{v:+.2f}%" if v is not None else "—"
        )
        signal_icons = {
            "STRONG BUY": "🚀 STRONG BUY", "BUY": "📈 BUY",
            "HOLD": "⏸️ HOLD",
            "SELL": "📉 SELL", "STRONG SELL": "🔥 STRONG SELL",
        }
        vdf["Signal"] = vdf["Signal"].map(signal_icons).fillna(vdf["Signal"])

        st.dataframe(vdf, use_container_width=True, hide_index=True)

        # Summary metrics
        total_v  = len(validated_rows)
        success_v = sum(1 for r in validated_rows if r.get("Success") == 1)
        v1, v2, v3 = st.columns(3)
        v1.metric("Total Validated",  total_v)
        v2.metric("Successful",       success_v)
        v3.metric("Success Rate",     f"{round(success_v / total_v * 100, 1)}%" if total_v else "—")
    else:
        st.caption("No validated recommendations yet.")

    st.divider()

    # ── Application Logs ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🪵 Application Logs</div>', unsafe_allow_html=True)

    log_c1, log_c2 = st.columns([3, 1])
    with log_c1:
        debug_enabled = st.checkbox(
            "Enable Debug Logs",
            value=ENABLE_DEBUG_LOGS,
            help="Write per-stock pillar diagnostics. Requires app restart to take effect.",
        )
        if debug_enabled != ENABLE_DEBUG_LOGS:
            st.info(
                "Update `ENABLE_DEBUG_LOGS` in `config.py` and restart the app "
                "to change the log level."
            )
    with log_c2:
        if st.button("🗑️ Clear Logs", use_container_width=True):
            if clear_log_file():
                st.success("Log file cleared.")
                _app_logger.info("Log file cleared by user")
            else:
                st.error("Could not clear log file.")

    log_size = log_file_size_kb()
    st.caption(f"Log file size: **{log_size} KB** · Max 5 MB · 5 rotating backups")

    log_lines = read_last_log_lines(100)
    if log_lines:
        # Colour-code by level
        coloured = []
        for line in log_lines:
            if "| ERROR   |" in line or "| CRITICAL|" in line:
                coloured.append(f'<span style="color:#f85149">{line}</span>')
            elif "| WARNING |" in line:
                coloured.append(f'<span style="color:#d29922">{line}</span>')
            elif "| DEBUG   |" in line:
                coloured.append(f'<span style="color:#8b949e">{line}</span>')
            else:
                coloured.append(f'<span style="color:#c9d1d9">{line}</span>')

        log_html = "<br>".join(coloured)
        st.markdown(
            f'''<div style="background:#0d1117;border:1px solid #30363d;
            border-radius:8px;padding:1rem;font-family:monospace;
            font-size:.72rem;line-height:1.5;max-height:420px;
            overflow-y:auto;white-space:pre-wrap;">{log_html}</div>''',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No log entries yet.")



# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_performance:

    st.markdown('<div class="sec-title">📊 Recommendation Performance Dashboard</div>', unsafe_allow_html=True)
    st.caption("Read-only analytics over all validated recommendations (5-day outcomes).")

    # ── Load data once; cache for this render cycle ───────────────────────────
    @st.cache_data(ttl=300, show_spinner=False)
    def _load_analytics():
        df      = load_validated_df()
        sig_df  = signal_performance(df)
        conf_df = confidence_performance(df)
        confl_df= confluence_performance(df)
        sent_df = sentiment_performance(df)
        mon_df  = monthly_trend(df)
        return df, sig_df, conf_df, confl_df, sent_df, mon_df

    _df, _sig, _conf, _confl, _sent, _mon = _load_analytics()

    if _df.empty:
        st.info(
            "No validated recommendations yet. "
            "Recommendations are validated automatically after 5 trading days. "
            "Use the **📋 My Tracker** tab to trigger validation."
        )

    else:
        _metrics = summary_metrics(_df)

        # ══════════════════════════════════════════════════════════════════════════
        # SECTION 1 — Summary Metrics
        # ══════════════════════════════════════════════════════════════════════════
        st.markdown("### 📈 Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Recommendations", _metrics["total"])
        m2.metric("Successful",            _metrics["successful"])
        m3.metric("Failed",                _metrics["failed"])
        m4.metric("Overall Success Rate",  f"{_metrics['success_rate']}%")

        m5, m6, m7 = st.columns(3)
        m5.metric("Average Return",  f"{_metrics['avg_return']:+.2f}%")
        m6.metric("Best Return",     f"{_metrics['best_return']:+.2f}%")
        m7.metric("Worst Return",    f"{_metrics['worst_return']:+.2f}%")

        st.divider()

        # ══════════════════════════════════════════════════════════════════════════
        # SECTION 2 — Signal Performance
        # ══════════════════════════════════════════════════════════════════════════
        st.markdown("### 🎯 Signal Performance")
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("**By Signal Type**")
            if not _sig.empty:
                st.dataframe(_sig, use_container_width=True, hide_index=True)
            else:
                st.caption("No data.")

        with col_b:
            if not _sig.empty:
                import plotly.graph_objects as go

                # Success Rate by Signal
                fig_sig = go.Figure()
                colours = {
                    "STRONG BUY": "#22c55e", "BUY": "#4ade80",
                    "HOLD": "#f59e0b",
                    "SELL": "#f87171", "STRONG SELL": "#ef4444",
                }
                bar_colours = [colours.get(s, "#8b949e") for s in _sig["Signal"]]
                fig_sig.add_trace(go.Bar(
                    x=_sig["Signal"], y=_sig["Success Rate %"],
                    marker_color=bar_colours,
                    text=_sig["Success Rate %"].astype(str) + "%",
                    textposition="outside",
                    name="Success Rate %",
                ))
                fig_sig.update_layout(
                    title="Success Rate by Signal",
                    yaxis_title="Success Rate %",
                    yaxis=dict(range=[0, 110]),
                    paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"),
                    height=320,
                    showlegend=False,
                )
                st.plotly_chart(fig_sig, use_container_width=True)

        # Average Return by Signal
        if not _sig.empty:
            fig_ret = go.Figure()
            ret_colours = ["#22c55e" if v >= 0 else "#ef4444" for v in _sig["Avg Return %"]]
            fig_ret.add_trace(go.Bar(
                x=_sig["Signal"], y=_sig["Avg Return %"],
                marker_color=ret_colours,
                text=_sig["Avg Return %"].apply(lambda v: f"{v:+.2f}%"),
                textposition="outside",
            ))
            fig_ret.update_layout(
                title="Average Return % by Signal",
                yaxis_title="Avg Return %",
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9"),
                height=300, showlegend=False,
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════════════════════════════
        # SECTION 3 — Confidence & Confluence Analysis
        # ══════════════════════════════════════════════════════════════════════════
        st.markdown("### 🧠 Confidence & Confluence Analysis")
        cc1, cc2 = st.columns(2)

        with cc1:
            st.markdown("**ML Confidence Bands**")
            st.caption("Does higher model confidence produce better results?")
            if not _conf.empty:
                st.dataframe(_conf, use_container_width=True, hide_index=True)
                fig_conf = go.Figure()
                fig_conf.add_trace(go.Scatter(
                    x=_conf["Confidence Band"], y=_conf["Success Rate %"],
                    mode="lines+markers+text",
                    text=_conf["Success Rate %"].astype(str) + "%",
                    textposition="top center",
                    line=dict(color="#4ade80", width=2),
                    marker=dict(size=8, color="#22c55e"),
                ))
                fig_conf.update_layout(
                    title="Confidence vs Success Rate",
                    yaxis_title="Success Rate %",
                    yaxis=dict(range=[0, 110]),
                    paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"), height=280,
                )
                st.plotly_chart(fig_conf, use_container_width=True)
            else:
                st.caption("No data.")

        with cc2:
            st.markdown("**Confluence Score Bands**")
            st.caption("Is the confluence score meaningful?")
            if not _confl.empty:
                st.dataframe(_confl, use_container_width=True, hide_index=True)
                fig_confl = go.Figure()
                fig_confl.add_trace(go.Scatter(
                    x=_confl["Confluence Band"], y=_confl["Success Rate %"],
                    mode="lines+markers+text",
                    text=_confl["Success Rate %"].astype(str) + "%",
                    textposition="top center",
                    line=dict(color="#f59e0b", width=2),
                    marker=dict(size=8, color="#d97706"),
                ))
                fig_confl.update_layout(
                    title="Confluence Score vs Success Rate",
                    yaxis_title="Success Rate %",
                    yaxis=dict(range=[0, 110]),
                    paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"), height=280,
                )
                st.plotly_chart(fig_confl, use_container_width=True)
            else:
                st.caption("No data.")

        st.divider()

        # ══════════════════════════════════════════════════════════════════════════
        # SECTION 4 — News Sentiment Analysis
        # ══════════════════════════════════════════════════════════════════════════
        st.markdown("### 📰 News Sentiment Analysis")
        st.caption("Does news sentiment help or hurt predictions?")
        ns1, ns2 = st.columns([1, 1])

        with ns1:
            if not _sent.empty:
                st.dataframe(_sent, use_container_width=True, hide_index=True)
            else:
                st.caption("No data.")

        with ns2:
            if not _sent.empty:
                sent_colours = {"🟢 Positive": "#22c55e", "🟡 Neutral": "#f59e0b", "🔴 Negative": "#ef4444"}
                fig_sent = go.Figure(go.Bar(
                    x=_sent["Sentiment"], y=_sent["Success Rate %"],
                    marker_color=[sent_colours.get(s, "#8b949e") for s in _sent["Sentiment"]],
                    text=_sent["Success Rate %"].astype(str) + "%",
                    textposition="outside",
                ))
                fig_sent.update_layout(
                    title="Sentiment vs Success Rate",
                    yaxis_title="Success Rate %",
                    yaxis=dict(range=[0, 110]),
                    paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"), height=300, showlegend=False,
                )
                st.plotly_chart(fig_sent, use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════════════════════════════
        # SECTION 5 — Monthly Trend
        # ══════════════════════════════════════════════════════════════════════════
        st.markdown("### 📅 Monthly Success Rate Trend")
        if not _mon.empty and len(_mon) > 1:
            fig_mon = go.Figure()
            fig_mon.add_trace(go.Scatter(
                x=_mon["Month"], y=_mon["Success Rate %"],
                mode="lines+markers+text",
                text=_mon["Success Rate %"].astype(str) + "%",
                textposition="top center",
                fill="tozeroy",
                line=dict(color="#4ade80", width=2),
                marker=dict(size=7),
                fillcolor="rgba(34,197,94,0.15)",
            ))
            fig_mon.update_layout(
                title="Monthly Success Rate Trend",
                yaxis_title="Success Rate %",
                yaxis=dict(range=[0, 110]),
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9"), height=320,
            )
            st.plotly_chart(fig_mon, use_container_width=True)
        elif not _mon.empty:
            st.info("Need at least 2 months of data to show the trend.")
        else:
            st.caption("No monthly data yet.")

        st.divider()

        # ══════════════════════════════════════════════════════════════════════════
        # SECTION 6 — Top Winners & Losers
        # ══════════════════════════════════════════════════════════════════════════
        st.markdown("### 🏅 Top Winners & Losers")
        w_col, l_col = st.columns(2)

        _winners = top_winners(_df, n=10)
        _losers  = top_losers(_df,  n=10)

        with w_col:
            st.markdown("**🏆 Top 10 Winners**")
            if not _winners.empty:
                _winners["Return %"] = _winners["Return %"].apply(lambda v: f"{v:+.2f}%")
                st.dataframe(_winners, use_container_width=True, hide_index=True)
            else:
                st.caption("No data.")

        with l_col:
            st.markdown("**📉 Top 10 Losers**")
            if not _losers.empty:
                _losers["Return %"] = _losers["Return %"].apply(lambda v: f"{v:+.2f}%")
                st.dataframe(_losers, use_container_width=True, hide_index=True)
            else:
                st.caption("No data.")

        st.divider()

        # ══════════════════════════════════════════════════════════════════════════
        # SECTION 7 — Auto Insights
        # ══════════════════════════════════════════════════════════════════════════
        st.markdown("### 💡 Automated Insights")
        _insights = generate_insights(_df, _sig, _conf, _confl, _sent)

        if _insights:
            for i, insight in enumerate(_insights, 1):
                st.markdown(
                    f'<div style="background:#161b22;border-left:3px solid #4ade80;'                f'padding:.6rem 1rem;margin:.4rem 0;border-radius:4px;'                f'color:#c9d1d9;font-size:.88rem;">'                f'<b style="color:#4ade80">{i}.</b> {insight}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Insights will appear once enough recommendations are validated.")