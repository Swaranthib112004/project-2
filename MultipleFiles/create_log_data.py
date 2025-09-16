import pandas as pd
import json
import os

def create_log_data():
    responses_csv = "data/responses.csv"
    concepts_per_item_json = "reports/concepts_per_item.json"
    output_json = "data/log_data.json"

    # Check input files exist
    if not os.path.exists(responses_csv):
        print(f"Error: {responses_csv} not found. Please provide your responses.csv file.")
        return
    if not os.path.exists(concepts_per_item_json):
        print(f"Error: {concepts_per_item_json} not found. Please run llm_concepts.py first to generate it.")
        return

    # Load responses.csv
    responses_df = pd.read_csv(responses_csv)

    # Load concepts_per_item.json
    with open(concepts_per_item_json, 'r', encoding='utf8') as f:
        concepts_per_item = json.load(f)

    # Map concept names to integer codes (1-indexed)
    all_concepts = sorted(list(set(c for sublist in concepts_per_item.values() for c in sublist if c)))
    concept_to_code = {concept: i + 1 for i, concept in enumerate(all_concepts)}

    log_data = []
    unique_users = responses_df['student_id'].unique()

    max_exer_id = 0
    for user_id_str in unique_users:
        user_logs = responses_df[responses_df['student_id'] == user_id_str]
        logs_list = []
        for _, row in user_logs.iterrows():
            item_id = row['item_id']
            is_correct = row['correct']

            # Extract numeric part of item_id (e.g., Q1 -> 1)
            try:
                exer_id_num = int(item_id.replace('Q', ''))
            except ValueError:
                print(f"Warning: Could not parse numeric exer_id from {item_id}. Skipping this log.")
                continue
            max_exer_id = max(max_exer_id, exer_id_num)

            # Get knowledge codes for the item
            knowledge_codes = []
            if item_id in concepts_per_item:
                for concept_name in concepts_per_item[item_id]:
                    if concept_name and concept_name in concept_to_code:
                        knowledge_codes.append(concept_to_code[concept_name])

            logs_list.append({
                'exer_id': exer_id_num,
                'score': is_correct,
                'knowledge_code': knowledge_codes
            })

        log_data.append({
            'user_id': int(user_id_str.replace('S', '')),  # Assuming student IDs like S1, S2
            'log_num': len(logs_list),
            'logs': logs_list
        })

    # Save log_data.json
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf8') as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    print(f"Generated {output_json} with {len(log_data)} students.")
    print(f"Total unique concepts (knowledge_n): {len(all_concepts)}")
    print(f"Max exercise ID (exer_n): {max_exer_id}")
    print(f"Total unique students (student_n): {len(unique_users)}")

    print("\nPlease create or update your data/config.txt with the following content:")
    print("student_n,exer_n,knowledge_n")
    print(f"{len(unique_users)},{max_exer_id},{len(all_concepts)}")

if __name__ == "__main__":
    create_log_data()