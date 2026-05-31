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
from scanner.cache import load_category_cache, cache_age_minutes, any_cache_exists
from scanner.background import (
    start_background_scan, is_scan_running, scan_progress, needs_scan
)
from storage.tracker import save_signal, get_recent_signals, get_accuracy_stats
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
tab_analyse, tab_home, tab_tracker = st.tabs([
    "🔍  Analyse Stock",
    "🏆  Top Picks",
    "📋  My Tracker",
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
        key=lambda r: r["score"],
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

    if predict_clicked:
        st.divider()

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
            headlines = fetch_news(selected_company)
        except Exception:
            headlines = []

        try:
            overall_sentiment, overall_score, headline_results = analyze_overall_sentiment(headlines)
        except Exception:
            overall_sentiment, overall_score, headline_results = "Neutral", 0.0, []

        try:
            regime_info = detect_regime(data)
        except Exception:
            regime_info = None

        # ── Multi-timeframe analysis ─────────────────────────────

        try:

            multi_tf_data = load_multi_timeframe_data(
                stock_symbol
            )

            weekly_trend = get_trend_signal(
                multi_tf_data["weekly"]
            )

            daily_trend = get_trend_signal(
                multi_tf_data["daily"]
            )

            raw_tf_score = (
                weekly_trend["score"] +
                daily_trend["score"]
            )

            timeframe_score = raw_tf_score / 2

        except Exception:

            weekly_trend = {
                "trend": "UNKNOWN",
                "score": 0
            }

            daily_trend = {
                "trend": "UNKNOWN",
                "score": 0
            }

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

        # Auto-save to tracker
        try:
            save_signal(stock_symbol, selected_company, final_signal,
                        final_score, confidence, acc, close_price)
        except Exception:
            pass

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

            st.markdown(
                '<div class="sec-title">⏱️ Multi-timeframe analysis</div>',
                unsafe_allow_html=True
            )

            t1, t2, t3 = st.columns(3)

            t1.metric(
                "Weekly Trend",
                weekly_trend["trend"]
            )

            t2.metric(
                "Daily Trend",
                daily_trend["trend"]
            )

            t3.metric(
                "Confluence Score",
                timeframe_score
            )

            st.markdown('<div class="sec-title">📰 Market sentiment</div>', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            s1.metric("Mood",      overall_sentiment)
            s2.metric("Avg score", round(overall_score, 2))

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
