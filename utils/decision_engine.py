def generate_signal(prediction,confidence,news_score):
    """Generate final AI recommendation"""

    ml_score = confidence / 100
    
    #Convert ML prediction direction

    if prediction == 0:
        ml_score = -ml_score
    final_score =(
        (ml_score * 0.7) + (news_score * 0.3)
    )
    
    #Final recommendation
    if final_score > 0.25:
        return "BUY",round(final_score,2),"Strong bullish alignment between ML Model and news sentiment"
    elif final_score < -0.25:
        return "SELL",round(final_score,2),"Negative momentum and berish news sentiment detected"
    return "HOLD",round(final_score,2),"Signals are mixed or confidence is weak"
