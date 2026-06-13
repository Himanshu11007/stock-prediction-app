import streamlit as st
from data.loader import (load_data,load_multi_timeframe_data)
from utils.helpers import prepare_data
from models.trainer import (train_model,ensemble_predict)
from news.api import fetch_news
from news.sentiment import analyze_overall_sentiment
from utils.decision_engine import generate_signal
from features.engineer import get_trend_signal


def get_top_recommendations(stock_list, stocks_df, _progress_bar=None):
    """
    Generate AI stock recommendations with quality filters.
    
    Args:
        stock_list (list): List of stock symbols (e.g., ["RELIANCE.NS", "TCS.NS"])
        stocks_df (pd.DataFrame): Stock database DataFrame with 'Symbol' and 'Company' columns
        progress_bar (st.progress, optional): Streamlit progress bar to update
        
    Returns:
        list: List of recommendation dictionaries with keys:
            - stock (company name), symbol, signal, score, confidence, accuracy, reason
    """
    recommendations = []
    
    for idx, stock in enumerate(stock_list):
        try:
            # Update progress bar if provided
            if _progress_bar is not None:
                _progress_bar.progress(
                    (idx + 1) / len(stock_list),
                    text=f"Processing {stock}..."
                )
            
            # Load stock data
            data = load_data(stock)
            if data.empty:
                continue
            
            # Prepare ML data
            data, X, y, _, _, y_train, _ = prepare_data(data)
            
            # Skip if target has only one class (can't train classifier)
            if len(set(y_train)) < 2:
                continue
            
            # Train model (walk-forward validation; final fit on full history)
            models, acc = train_model(X, y)
            
            # Filter weak models (accuracy threshold)
            if acc < 0.50:
                continue
            
            # Latest prediction
            latest_data = X.iloc[-1:]
            
            # Confidence (handle models without predict_proba)
            try:
                pred, confidence ,proba = ensemble_predict(models,latest_data)
            except AttributeError:
                confidence = 50.0  # Neutral fallback
            
            # News sentiment
            headlines = fetch_news(stock)
            overall_sentiment, overall_score, headline_results, _ = analyze_overall_sentiment(headlines)
            
            multi_tf_data = load_multi_timeframe_data(stock)

            weekly_trend = get_trend_signal(
                multi_tf_data["weekly"]
            )

            daily_trend = get_trend_signal(
                multi_tf_data["daily"]
            )

            timeframe_score = (weekly_trend["score"] + daily_trend["score"]) / 2

            # Final AI signal
            final_signal, final_score, reason,factors = generate_signal(
                prediction = pred,
                confidence = confidence,
                news_score = overall_score,
                timeframe_score=timeframe_score,
                data=data,
                regime_info=None
            )
        

            # Get company name from stock database
            company_row = stocks_df[stocks_df["Symbol"] == stock]
            company_name = (
                company_row.iloc[0]["Company"] 
                if not company_row.empty 
                else stock
            )
            
            if final_signal not in("BUY","STRONG BUY"):
                continue

            # Save result
            recommendations.append({
                "stock": company_name,  # Display name
                "symbol": stock,        # Keep symbol for reference
                "signal": final_signal,
                "score": round(final_score, 2),
                "confidence": round(confidence, 2),
                "accuracy": round(acc * 100, 2),
                "reason": reason,
                "weekly_trend":weekly_trend["trend"],
                "daily_trend":daily_trend["trend"],
                "timeframe_score":timeframe_score,
                "factors":factors
            })
            
        except Exception as e:
            st.warning(f"⚠️ Failed to process {stock}: {str(e)}")
            continue
    
    return recommendations