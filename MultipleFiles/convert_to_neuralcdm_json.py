import os
import json
import pandas as pd

def run(input_csv: str, output_dir: str):
    """
    Convert responses CSV to NeuralCDM JSON format.

    Args:
        input_csv (str): Path to the input CSV file containing responses.
        output_dir (str): Directory to save the output JSON file.

    Returns:
        str: Path to the generated JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    out = []
    for _, row in df.iterrows():
        out.append({
            "user_id": str(row["student_id"]),
            "item_id": str(row["item_id"]),
            "is_correct": int(row["correct"]),
        })

    out_path = os.path.join(output_dir, "responses.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✅ Converted {len(out)} records to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert responses CSV to NeuralCDM JSON format.")
    parser.add_argument("input_csv", help="Path to responses.csv")
    parser.add_argument("output_dir", help="Output directory for responses.json")

    args = parser.parse_args()
    run(args.input_csv, args.output_dir)