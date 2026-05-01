import os
import gradio as gr  # type: ignore
import requests  # type: ignore

API_URL = os.environ.get("SPACE_API_URL", "http://localhost:8000")

def get_color_map():
    # Pre-compute all potential confidence string labels to proper gradio color bounds
    cmap = {}
    for i in range(101):
        conf = i / 100.0
        label = f"Score: {conf:.2f}"
        if conf >= 0.8:
            cmap[label] = "red"
        elif conf >= 0.6:
            cmap[label] = "orange"
        else:
            cmap[label] = "yellow"
    return cmap


def detect_hallucination(prompt, response, sampled_responses_text):
    try:
        payload = {
            "query": prompt,
            "response": response
        }

        # Parse sampled responses
        if sampled_responses_text and sampled_responses_text.strip():
            lines = [line.strip() for line in sampled_responses_text.split("\n") if line.strip()]
            if lines:
                payload["sampled_responses"] = lines

        res = requests.post(f"{API_URL}/detect", json=payload)

        if res.status_code != 200:
            raise Exception(f"API Error: {res.text}")

        data = res.json()

        # Build highlighted spans from backend response
        spans = data.get("spans", [])
        highlighted = []
        current_idx = 0

        for s in spans:
            start, end = s.get("start", 0), s.get("end", 0)
            conf = s.get("confidence", 0.0)

            if start > current_idx:
                highlighted.append((response[current_idx:start], None))

            if end > current_idx:
                actual_start = max(start, current_idx)
                label = f"Score: {conf:.2f}"
                highlighted.append((response[actual_start:end], label))
                current_idx = end

        if current_idx < len(response):
            highlighted.append((response[current_idx:], None))

        if not highlighted:
            highlighted = [(response, None)]

        # Map backend 'signals' to UI output fields
        signals = data.get("signals", {})
        is_hal = data.get("is_hallucinated", False)
        explanation = f"Hallucination detected (score: {data.get('hallucination_score', 0):.3f})" if is_hal else "No significant hallucination detected."

        return (
            highlighted,
            data.get("hallucination_score", 0.0),
            explanation,
            signals.get("retrieval_similarity", 0.0),
            signals.get("nli_entailment", 0.0),
            signals.get("semantic_entropy", 0.0),
            data
        )

    except requests.exceptions.ConnectionError:
        gr.Warning("API server not running. Start with: uvicorn api:app --port 8000")
        return ([(response, None)], 0.0, "Connection Failed", 0.0, 0.0, 0.0, {})
    except Exception as e:
        return ([(response, None)], 0.0, f"Error: {str(e)}", 0.0, 0.0, 0.0, {})


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 RUC-Detect")
    gr.Markdown("Real-time hallucination detection system")

    with gr.Row():
        with gr.Column(scale=1):
            prompt_in = gr.Textbox(label="Prompt", lines=3)
            response_in = gr.Textbox(label="LLM Response", lines=4)
            sampled_in = gr.Textbox(
                label="Sampled Responses (optional, one per line, min 3)", 
                lines=5, 
                placeholder="Paste 3+ alternative responses to enable SelfCheckGPT..."
            )
            btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            span_out = gr.HighlightedText(
                label="Flagged Sentences",
                color_map=get_color_map()
            )
            
            with gr.Row():
                score_out = gr.Number(label="Overall Hallucination Score")

            with gr.Row():
                ret_sim_out = gr.Number(label="Retrieval Similarity")
                nli_out = gr.Number(label="NLI Score")
                self_out = gr.Number(label="SelfCheck Score")

            explanation_out = gr.Textbox(label="Explanation", lines=3)
            
            with gr.Accordion("Raw API Response", open=False):
                raw_json_out = gr.JSON()

    btn.click(
        fn=detect_hallucination,
        inputs=[prompt_in, response_in, sampled_in],
        outputs=[
            span_out, 
            score_out, 
            explanation_out,
            ret_sim_out,
            nli_out,
            self_out,
            raw_json_out
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
