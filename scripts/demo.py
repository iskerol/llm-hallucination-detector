import os
import sys
import argparse
from rich.console import Console

def main():
    parser = argparse.ArgumentParser(description="RAID CLI Demo Toolkit")
    parser.add_argument("--query", required=True, help="Input textual query maps.")
    parser.add_argument("--response", required=True, help="Generative LLM payload verifying logic bounds.")
    args = parser.parse_args()
    
    console = Console()
    console.print("[cyan]Initializing local memory dependencies...[/cyan]")
    
    from config import config
    from knowledge_base.embedder import SentenceEmbedder
    from knowledge_base.builder import KnowledgeBaseBuilder
    from detection.retriever import FAISSRetriever
    from detection.scorer import RetrievalSimilarityScorer, NLIEntailmentScorer, SemanticEntropyScorer
    from detection.span_detector import SpanLevelDetector
    from detection.ensemble import HallucinationDetector
    
    embedder = SentenceEmbedder(getattr(config, "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    builder = KnowledgeBaseBuilder(embedder)
    
    # Boot logic 
    kb_path = getattr(config, "FAISS_INDEX_PATH", "./data/index")
    if os.path.exists(kb_path) and os.path.exists(os.path.join(kb_path, "faiss_index.bin")):
        chunks, _, index = builder.load(kb_path)
    else:
        console.print("[yellow]Index bounds missing. Sideloading localized dummy arrays...[/yellow]")
        docs = [{"id": "1", "title": "KB", "text": "Alexander Graham Bell invented the telephone in 1876."}]
        builder.build_from_scratch(docs, "IndexFlatIP", kb_path)
        chunks, index = builder.chunks, builder.index

    nli_mod = getattr(config, "NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
    retriever = FAISSRetriever(chunks, index, embedder, top_k=3)
    sim_scorer = RetrievalSimilarityScorer(embedder)
    nli_scorer = NLIEntailmentScorer(nli_mod)
    detector = HallucinationDetector(retriever, sim_scorer, nli_scorer, SemanticEntropyScorer(embedder), SpanLevelDetector(sim_scorer, nli_scorer))
    
    console.print("[cyan]Tracing...[/cyan]")
    res = detector.detect(args.query, args.response)
    
    c = "red" if res["is_hallucinated"] else "line"
    v = "HALLUCINATED" if res["is_hallucinated"] else "GROUNDED"
    
    console.print(f"\n[bold]Verdict:[/bold] [{c}]{v}[/{c}] (Confidence/Score: {res['hallucination_score']:.3f})")
    console.print(f"[bold]Latency:[/bold] {res['latency_ms']:.1f}ms\n")
    
    console.print("[bold]Top Retrievers:[/bold]")
    for p in res.get("supporting_passages", [])[:3]:
        console.print(f"  - \[{p.get('score', 0):.2f}] {p.get('text', '')}")
        
    console.print(f"\n[bold]Highlighted Span Markup:[/bold]\n{res.get('highlighted_html', '')}\n")

if __name__ == "__main__":
    main()
