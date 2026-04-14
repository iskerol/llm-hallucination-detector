import os
import sys

def run_tests():
    passed = 0
    total = 5

    print("--- STARTING INTEGRATION TESTS ---\n")

    # Test 1: Utils imports
    print("Test 1: Utils imports & get_nli_score")
    try:
        from utils.nli import get_nli_score
        from utils.selfcheck import selfcheck_nli
        from utils.taxonomy import classify_pattern
        
        label1, _ = get_nli_score("The sky is blue.", "The sky is blue.")
        assert label1 == "ENTAILMENT", f"Expected ENTAILMENT, got {label1}"
        
        label2, _ = get_nli_score("The sky is blue.", "The sky is green.")
        assert label2 == "CONTRADICTION", f"Expected CONTRADICTION, got {label2}"
        print("[PASS] Test 1\n")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 1: {e}\n")

    # Test 2: batch_nli_scores
    print("Test 2: batch_nli_scores")
    try:
        from utils.nli import batch_nli_scores
        results = batch_nli_scores([
            ("Sky is blue", "Sky is blue"),
            ("Sky is blue", "Sky is red")
        ], batch_size=16)
        assert len(results) == 2, "Expected 2 results"
        assert results[0][0] == "ENTAILMENT", f"Expected ENTAILMENT, got {results[0][0]}"
        print("[PASS] Test 2\n")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 2: {e}\n")

    # Test 3: taxonomy classify_pattern
    print("Test 3: taxonomy classify_pattern")
    try:
        from utils.taxonomy import classify_pattern
        # Extrinsic
        p1, _ = classify_pattern(avg_sim=0.3, nli_score=0.2, final_score=0.8)
        assert p1 == "extrinsic", f"Expected extrinsic, got {p1}"
        
        # Intrinsic
        p2, _ = classify_pattern(avg_sim=0.8, nli_score=0.2, final_score=0.6)
        assert p2 == "intrinsic", f"Expected intrinsic, got {p2}"
        
        # Semantic Drift
        p3, _ = classify_pattern(avg_sim=0.8, nli_score=0.8, final_score=0.6)
        assert p3 == "semantic_drift", f"Expected semantic_drift, got {p3}"
        
        # None
        p4, _ = classify_pattern(avg_sim=0.8, nli_score=0.8, final_score=0.2)
        assert p4 == "None", f"Expected None, got {p4}"
        
        print("[PASS] Test 3\n")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 3: {e}\n")

    # Test 4: selfcheck_nli
    print("Test 4: selfcheck_nli")
    try:
        from utils.selfcheck import selfcheck_nli
        response = "Paris is the capital of France."
        sampled = [
            "Paris is the capital of France.",
            "Paris is the capital of France.",
            "London is the capital of France."
        ]
        avg, _ = selfcheck_nli(response, sampled)
        assert 0.0 <= avg <= 1.0, f"Average {avg} out of bounds"
        print("[PASS] Test 4\n")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 4: {e}\n")

    # Test 5: Full pipeline
    print("Test 5: Full pipeline")
    try:
        if not os.path.exists("models/faiss.index"):
            print("[SKIP] Test 5: models/faiss.index not found. Run index build first.\n")
            total -= 1
        else:
            from pipeline import run_pipeline
            result = run_pipeline("Who is the PM of India?", "The PM of India is Elon Musk.")
            keys = {"score", "label", "explanation", "pattern", "spans", "components"}
            assert set(result.keys()) == keys, "Missing keys in result"
            assert 0.0 <= result["score"] <= 1.0, "Score out of bounds"
            assert result["label"] is True, "Expected hallucination label=True"
            print("[PASS] Test 5\n")
            passed += 1
    except Exception as e:
        print(f"[FAIL] Test 5: {e}\n")

    print(f"--- SUMMARY: {passed}/{total} tests passed ---")

if __name__ == "__main__":
    run_tests()
