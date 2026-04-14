def classify_pattern(avg_sim, nli_score, final_score):
    """
    Classifies the type of hallucination based on similarity and NLI metrics.
    Returns:
        pattern (str), explanation (str)
    """
    if final_score < 0.52:
        return "None", "No significant hallucination detected."

    if nli_score <= 0.3 and avg_sim > 0.6:
        # High document similarity but fails NLI (contradicts)
        return "intrinsic", "Intrinsic hallucination: The response contradicts the retrieved evidence despite high lexical similarity."
    elif avg_sim <= 0.4 and nli_score <= 0.3:
        # Low document similarity and fails NLI
        return "extrinsic", "Extrinsic hallucination: The response contains unverifiable information not present in the retrieved evidence."
    else:
        return "semantic_drift", "Semantic drift: The response shifts context beyond the source material, leading to reduced factual alignment."
