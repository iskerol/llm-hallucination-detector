import numpy as np
import spacy

from utils.nli import get_nli_score

nlp = spacy.load("en_core_web_sm")

def selfcheck_nli(response, sampled_responses):
    """
    Implements SelfCheckGPT-NLI sentence-level scoring natively.
    Returns:
        avg_score: Document-level score (average of sentence scores)
        sentence_scores: List of hallucination scores for each sentence
    """
    doc = nlp(response)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    sentence_scores = []

    for sent in sentences:
        sample_scores = []
        for sample in sampled_responses:
            # SelfCheckGPT-NLI checks if the sample (premise) entails the sentence (hypothesis)
            label, score = get_nli_score(sample, sent)

            # proxy for Equation 10: evaluate contradiction probability
            # The hallucination sentence score S(i) is based on the average contradiction probability
            # across the samples.
            if label == "CONTRADICTION":
                p_contradict = score
            elif label == "ENTAILMENT":
                p_contradict = 1.0 - score
            else:
                p_contradict = 0.5

            sample_scores.append(p_contradict)

        sentence_scores.append(float(np.mean(sample_scores)) if sample_scores else 0.0)

    avg_score = float(np.mean(sentence_scores)) if sentence_scores else 0.0
    return avg_score, sentence_scores
