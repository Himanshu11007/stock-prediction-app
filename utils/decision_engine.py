def generate_signal(prediction,confidence,sentiment_score):
    """Generate final AI recommendation"""

    #Convert ML Prediction
    ml_score = 1 if prediction == 1 else -1
    
    #Weighted final score
    final_score=(
        (ml_score * 0.7) + (sentiment_score * 0.3)
    )
    #Weak confidence protection
    if confidence < 60:
        return "HOLD",final_score
    
    #Final recommendation
    if final_score > 0.5:
        return "BUY",final_score
    elif final_score < -0.5:
        return "SELL",final_score
    return "HOLD",final_score
