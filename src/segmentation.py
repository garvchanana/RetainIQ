def segment_from_rfm(rfm_score: float) -> str:
    rfm = int(round(rfm_score))

    if rfm >= 10:
        return "High Value"
    elif rfm >= 7:
        return "Potential Loyalists"
    elif rfm >= 5:
        return "At Risk"
    else:
        return "Low Value"