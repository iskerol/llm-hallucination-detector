def classify_pattern(avg_sim, nli_score, final_score):
    if nli_score < 0.3:
        return "contradiction", "Response contradicts retrieved knowledge."
    elif avg_sim < 0.3:
        return "fabrication", "Response contains unsupported information."
    elif avg_sim > 0.6 and nli_score < 0.5:
        return "inference", "Response misinterprets retrieved context."
    else:
        return "specificity", "Response may be too vague or overly specific."
