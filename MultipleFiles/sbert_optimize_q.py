import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# SBERT is optional; fall back to bag-of-words if not available
USE_SBERT = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    USE_SBERT = False

def build_raw_q(concepts_per_item_path: str, concept_universe_path: str, out_path: str):
    """
    Build raw Q-matrix from concepts per item and concept universe.

    Args:
        concepts_per_item_path (str): Path to concepts_per_item.json
        concept_universe_path (str): Path to concept_universe.json
        out_path (str): Path to save raw Q-matrix CSV

    Returns:
        tuple: (out_path, concepts list, items list, Q numpy array)
    """
    with open(concepts_per_item_path, "r") as f:
        cpi = json.load(f)
    with open(concept_universe_path, "r") as f:
        universe = json.load(f)

    concepts = [c for c in universe if c]
    items = sorted(cpi.keys())

    Q = np.zeros((len(items), len(concepts)), dtype=int)
    for i, qid in enumerate(items):
        for c in cpi[qid]:
            if c in concepts:
                j = concepts.index(c)
                Q[i, j] = 1

    df = pd.DataFrame(Q, columns=concepts)
    df.insert(0, "item_id", items)
    df.to_csv(out_path, index=False)
    return out_path, concepts, items, Q

def optimize_with_sbert(items_csv: str, concepts: list, raw_q_path: str, out_path: str, threshold: float = 0.38):
    """
    Optimize Q-matrix using SBERT similarity or fallback keyword overlap.

    Args:
        items_csv (str): Path to items.csv containing question_text
        concepts (list): List of concept names
        raw_q_path (str): Path to raw Q-matrix CSV
        out_path (str): Path to save optimized Q-matrix CSV
        threshold (float): Similarity threshold for concept assignment

    Returns:
        str: Path to optimized Q-matrix CSV
    """
    items_df = pd.read_csv(items_csv)
    raw_q = pd.read_csv(raw_q_path)

    item_texts = items_df.set_index("item_id")["question_text"].to_dict()
    concept_names = concepts

    if USE_SBERT:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        item_vecs = model.encode([item_texts[iid] for iid in raw_q["item_id"].tolist()], normalize_embeddings=True)
        concept_vecs = model.encode(concept_names, normalize_embeddings=True)
        sims = cosine_similarity(item_vecs, concept_vecs)
    else:
        # fallback: simple Jaccard-like keyword overlap
        def tokenize(s): 
            return set(str(s).lower().replace("(", " ").replace(")", " ").replace(",", " ").split())
        item_toks = [tokenize(item_texts[iid]) for iid in raw_q["item_id"].tolist()]
        concept_toks = [tokenize(c) for c in concept_names]
        sims = np.zeros((len(item_toks), len(concept_toks)))
        for i, it in enumerate(item_toks):
            for j, ct in enumerate(concept_toks):
                inter = len(it & ct)
                union = len(it | ct) or 1
                sims[i, j] = inter / union

    refined = raw_q.copy()
    refined.iloc[:, 1:] = (sims >= threshold).astype(int)  # overwrite with similarity rule

    refined.to_csv(out_path, index=False)
    return out_path

def run(items_csv: str, reports_dir: str = "reports"):
    """
    Run the full pipeline to build and optimize Q-matrix.

    Args:
        items_csv (str): Path to items.csv
        reports_dir (str): Directory containing concept files and to save outputs

    Returns:
        str: Path to optimized Q-matrix CSV
    """
    os.makedirs(reports_dir, exist_ok=True)
    concepts_per_item = os.path.join(reports_dir, "concepts_per_item.json")
    concept_universe = os.path.join(reports_dir, "concept_universe.json")
    raw_q_path = os.path.join(reports_dir, "q_matrix_raw.csv")
    opt_q_path = os.path.join(reports_dir, "q_matrix_optimized.csv")

    raw_q_path, concepts, items, Q = build_raw_q(concepts_per_item, concept_universe, raw_q_path)
    optimize_with_sbert(items_csv, concepts, raw_q_path, opt_q_path)

    print(f"✅ Raw Q saved to {raw_q_path}")
    print(f"✅ Optimized Q saved to {opt_q_path}")
    return opt_q_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and optimize Q-matrix using SBERT.")
    parser.add_argument("items_csv", help="Path to items.csv")
    parser.add_argument("--reports_dir", default="reports", help="Directory for concept files and outputs")

    args = parser.parse_args()
    run(args.items_csv, args.reports_dir)