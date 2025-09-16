import os, json
import subprocess # To run external Python scripts
import sys # To pass arguments to external scripts

# Import functions from your existing modules
from MultipleFiles.llm_concepts import run as concepts_run
from MultipleFiles.sbert_optimize_q import run as sbert_q_run
# from MultipleFiles.convert_to_neuralcdm_json import run as json_run # Not needed with new NCDM

def main():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("model", exist_ok=True) # For NCDM model snapshots
    os.makedirs("result", exist_ok=True) # For NCDM result files

    print("--- Starting Pipeline ---")

    # Step 0: Prepare data for NCDM (run dividedata.py)
    print("0) Preparing data for NCDM (running dividedata.py)...")
    try:
        # Ensure log_data.json and config.txt are in data/
        # You might need a script to generate log_data.json from your responses.csv and concepts_per_item.json
        # (See create_log_data.py example above)
        subprocess.run([sys.executable, 'ncdm_src/dividedata.py'], check=True)
        print("✅ Data preparation complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during data preparation: {e}")
        print("Please ensure 'data/log_data.json' and 'data/config.txt' are correctly set up.")
        sys.exit(1)


    print("1) Extracting concepts + Bloom with LLaMA (Ollama)...")
    try:
        concepts_run("data/items.csv", "reports")
    except Exception as e:
        print(f"❌ Error during LLM concept extraction: {e}")
        print("Please ensure Ollama is running and the specified LLM model is pulled.")
        sys.exit(1)

    print("2) Building & optimizing Q-matrix with SBERT...")
    try:
        opt_q_path = sbert_q_run("data/items.csv", "reports")
        print(f"✅ Optimized Q-matrix saved to {opt_q_path}")
    except Exception as e:
        print(f"❌ Error during SBERT Q-matrix optimization: {e}")
        print("Please ensure sentence-transformers is installed. Fallback to keyword overlap is used if SBERT fails.")
        sys.exit(1)

    # Step 3: Train the NCDM model (run train.py)
    print("3) Training NCDM model...")
    try:
        # Read student_n, exer_n, knowledge_n from config.txt for train.py
        with open('data/config.txt') as f:
            f.readline() # Skip header
            student_n, exer_n, knowledge_n = map(int, f.readline().split(','))
        
        # Determine device (cpu or cuda)
        device_arg = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Run train.py
        # We'll use a fixed number of epochs for the pipeline, e.g., 10
        ncdm_epochs = 10 
        subprocess.run([sys.executable, 'ncdm_src/train.py', device_arg, str(ncdm_epochs)], check=True)
        print(f"✅ NCDM training complete for {ncdm_epochs} epochs.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during NCDM training: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during NCDM training setup: {e}")
        sys.exit(1)

    # Step 4: Evaluate the NCDM model and get mastery/difficulty (run predict.py)
    print("4) Evaluating NCDM model and extracting mastery/difficulty...")
    try:
        # Use the last trained epoch for prediction
        subprocess.run([sys.executable, 'ncdm_src/predict.py', str(ncdm_epochs)], check=True)
        print("✅ NCDM evaluation and data extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during NCDM prediction/evaluation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during NCDM prediction setup: {e}")
        sys.exit(1)

    print("--- Pipeline Finished ---")

    # Display final metrics from reports/metrics.json
    metrics_path = "reports/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print("✅ Final Metrics:")
        print(f"   RMSE={metrics.get('RMSE', float('nan')):.3f}")
        print(f"   AUC={metrics.get('AUC', float('nan')):.3f}")
        print(f"   Accuracy={metrics.get('Accuracy', float('nan')):.3f}")
        print(f"   F1={metrics.get('F1', float('nan')):.3f}  Precision={metrics.get('Precision', float('nan')):.3f}  Recall={metrics.get('Recall', float('nan')):.3f}")
    else:
        print("❌ Metrics file not found. Something went wrong during NCDM evaluation.")


if __name__ == "__main__":
    # Add ncdm_src to Python path so imports work
    sys.path.append(os.path.join(os.path.dirname(__file__), 'ncdm_src'))
    main()