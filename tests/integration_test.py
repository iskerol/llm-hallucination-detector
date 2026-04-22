import os
import sys
import time
import numpy as np
from rich.console import Console
from rich.table import Table

# Import core modules cleanly assuming structural definitions exist natively
from knowledge_base.embedder import SentenceEmbedder
from knowledge_base.faiss_index import FAISSIndexManager
from detection.retriever import FAISSRetriever
from detection.scorer import RetrievalSimilarityScorer, NLIEntailmentScorer, SemanticEntropyScorer
from detection.span_detector import SpanLevelDetector
from detection.ensemble import HallucinationDetector

KNOWLEDGE_BASE = [
    "Alexander Graham Bell invented the telephone in 1876.",
    "The telephone was the first device in history that enabled people to talk directly across long distances.",
    "Thomas Edison invented the practical incandescent light bulb in 1879.",
    "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
    "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics.",
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "Jupiter is the fifth planet from the Sun and the largest in the Solar System.",
    "Water is a chemical substance composed of two hydrogen atoms and one oxygen atom.",
    "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states.",
    "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas.",
    "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer.",
    "The human genome contains roughly 3 billion DNA base pairs.",
    "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
    "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York.",
    "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969.",
    "Neil Armstrong and Buzz Aldrin were the first two humans to walk on the lunar surface.",
    "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.",
    "The Amazon River in South America is the largest river by discharge volume.",
    "Python is a high-level programming language created by Guido van Rossum.",
    "Mitochondria are membrane-bound organelles that generate biological energy."
]

def run_integration():
    console = Console()
    console.print("\n[bold cyan]=== Initializing RAID End-To-End Memory Tests ===[/bold cyan]")
    
    t0 = time.time()
    embedder = SentenceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    docs = [{"id": str(i), "title": f"Doc {i}", "text": text} for i, text in enumerate(KNOWLEDGE_BASE)]
    
    console.print("[cyan]Encoding 20 FAISS arrays...[/cyan]")
    embeddings = embedder.encode(KNOWLEDGE_BASE, show_progress=False)
    index = FAISSIndexManager().build(embeddings, "IndexFlatIP")
    
    console.print("[cyan]Initializing Cross-Encoder and Tri-State Ensemble Logic...[/cyan]")
    retriever = FAISSRetriever(docs, index, embedder, top_k=3)
    sim_scorer = RetrievalSimilarityScorer(embedder)
    
    # Running base deBERTa v3. May take 2-4 seconds bounding inference memory
    nli_scorer = NLIEntailmentScorer(model_name="cross-encoder/nli-deberta-v3-base")
    ent_scorer = SemanticEntropyScorer(embedder)
    span_detector = SpanLevelDetector(sim_scorer, nli_scorer)
    
    detector = HallucinationDetector(retriever, sim_scorer, nli_scorer, ent_scorer, span_detector)
    console.print(f"[green]Framework Hotloaded in {time.time()-t0:.2f}s locally.[/green]\n")
    
    test_cases = [
        # Grounded validation sets
        {"query": "Who invented the telephone?", "response": "Alexander Graham Bell invented the telephone in 1876.", "max_score": 0.4},
        {"query": "Where is the Eiffel Tower?", "response": "The Eiffel Tower is in Paris, France.", "max_score": 0.4},
        # Defect validation structures
        {"query": "Who invented the telephone?", "response": "Thomas Edison invented the telephone in 1912.", "min_score": 0.6},
        {"query": "What is the speed of light?", "response": "The speed of light is 100 miles per hour.", "min_score": 0.6},
        # Mixed Ambiguity (Triggers NLI Neutral bounds)
        {"query": "Tell me about Jupiter.", "response": "Jupiter is the fifth planet and has water rings.", "max_score": 0.7, "min_score": 0.3}
    ]
    
    table = Table(title="RAID Component Matrix Verifier")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Query Preview", style="magenta")
    table.add_column("Verdict", justify="center")
    table.add_column("Score", justify="right", style="blue")
    table.add_column("Latency", justify="right")
    table.add_column("Pass?", justify="center")
    
    all_passed = True
    for i, t in enumerate(test_cases):
        res = detector.detect(t["query"], t["response"])
        score = res["hallucination_score"]
        lat = res["latency_ms"]
        
        passed = True
        if "max_score" in t and score > t["max_score"]: passed = False
        if "min_score" in t and score < t["min_score"]: passed = False
        if lat > 5000: passed = False  # Latency boundary constraints
        
        vdict = "[red]🚨 HAL[/red]" if res["is_hallucinated"] else "[green]✅ GND[/green]"
        icon = "[bold green]✓[/bold green]" if passed else "[bold red]✗[/bold red]"
        if not passed: all_passed = False
            
        table.add_row(str(i+1), t["response"][:35] + "...", vdict, f"{score:.3f}", f"{int(lat)}ms", icon)
        
    console.print(table)
    
    if all_passed:
        console.print("\n[bold green]SUCCESS: All logical assertions validated correctly (Return Code 0).[/bold green]\n")
        sys.exit(0)
    else:
        console.print("\n[bold red]FAILURE: Trace violation bounds broken dynamically (Return Code 1).[/bold red]\n")
        sys.exit(1)

if __name__ == "__main__":
    run_integration()
