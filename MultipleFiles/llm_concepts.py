import os, json, re
import pandas as pd
import ollama

USE_OLLAMA = True
try:
    import ollama  # requires Ollama app running
except Exception as e:
    USE_OLLAMA = False
    print("Warning: Ollama not available. LLM concept extraction will not work.")

BLOOM_LEVELS = ["Remember","Understand","Apply","Analyze","Evaluate","Create"]

SYSTEM_PROMPT = (
    "You are an expert in educational assessment. "
    "Given a question, extract 2–4 short concept tags and 1 Bloom level. "
    "Return STRICT JSON with fields: concepts (list of strings), bloom (one of Remember, Understand, Apply, Analyze, Evaluate, Create)."
)

def ask_llama(question: str):
    prompt = (
        SYSTEM_PROMPT + "\n"
        + "Question: " + question + "\n"
        + "Respond ONLY JSON. Example: {\"concepts\":[\"Stack\",\"LIFO\"],\"bloom\":\"Understand\"}"
    )
    if USE_OLLAMA:
        try:
            resp = ollama.chat(model=os.getenv("LLM_MODEL","llama3:3b"), messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ])
            txt = resp["message"]["content"]
            j = json.loads(txt)
            if "concepts" in j and "bloom" in j:
                return j
            else:
                raise ValueError(f"LLM response missing 'concepts' or 'bloom' field: {txt}")
        except Exception as e:
            raise RuntimeError(f"Error calling Ollama or parsing response: {e}. Ensure Ollama is running and model is pulled.")
    else:
        raise RuntimeError("Ollama is not available. Cannot perform LLM concept extraction without it.")

def run(items_csv: str, out_dir: str = "reports"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(items_csv)
    out_rows = []
    all_concepts = set()

    print(f"Processing {len(df)} questions from {items_csv}...")
    for idx, row in df.iterrows():
        qid = row["item_id"]
        text = row["question_text"]
        try:
            ans = ask_llama(text)
            concepts = ans.get("concepts", [])
            bloom = ans.get("bloom", "Understand")
            all_concepts.update(concepts)
            out_rows.append({"item_id": qid, "question_text": text, "concepts": ";".join(concepts), "bloom": bloom})
            print(f"  Processed {qid}: Concepts={concepts}, Bloom={bloom}")
        except Exception as e:
            print(f"  Error processing {qid} ('{text}'): {e}. Skipping this item.")
            # Optionally, you can append a row with empty concepts/default bloom here
            out_rows.append({"item_id": qid, "question_text": text, "concepts": "", "bloom": "Unknown"})


    result_df = pd.DataFrame(out_rows)
    result_df.to_csv(os.path.join(out_dir, "questions_bloom.csv"), index=False)

    # also save raw concept list per item
    concepts_map = {r["item_id"]: r["concepts"].split(";") if r["concepts"] else [] for r in out_rows}
    with open(os.path.join(out_dir, "concepts_per_item.json"), "w") as f:
        json.dump(concepts_map, f, indent=2)

    # save concept universe
    with open(os.path.join(out_dir, "concept_universe.json"), "w") as f:
        json.dump(sorted(c for c in all_concepts if c), f, indent=2)

    print(f"✅ Concepts + Bloom saved to {out_dir}/questions_bloom.csv and concepts_per_item.json")
    return os.path.join(out_dir, "questions_bloom.csv")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python llm_concepts.py <items_csv>")
        print("Example: python llm_concepts.py data/items.csv")
        sys.exit(1)

    items_csv = sys.argv[1]
    run(items_csv)
