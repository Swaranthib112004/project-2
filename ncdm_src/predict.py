import torch
import numpy as np
import json
import sys
import os # Added for path handling
from sklearn.metrics import roc_auc_score
from ncdm_src.data_loader import ValTestDataLoader # Adjusted import path
from ncdm_src.model import Net # Adjusted import path
import pandas as pd # Added for saving results

# These will be read from config.txt
exer_n = 0
knowledge_n = 0
student_n = 0


def test(epoch):
    data_loader = ValTestDataLoader('test')
    net = Net(student_n, exer_n, knowledge_n)
    device = torch.device('cpu') # Default to CPU for prediction
    print('testing model...')
    data_loader.reset()
    
    model_path = 'model/model_epoch' + str(epoch)
    if not os.path.exists(model_path):
        print(f"Error: Model snapshot not found at {model_path}")
        return

    load_snapshot(net, model_path)
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        if input_stu_ids is None: # Handle end of data
            break
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        out_put = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
        out_put = out_put.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and out_put[i] > 0.5) or (labels[i] == 0 and out_put[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += out_put.tolist()
        label_all += labels.tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))
    
    # Save metrics to reports/metrics.json
    metrics = {"RMSE": rmse, "AUC": auc, "Accuracy": accuracy, 
               "F1": float("nan"), "Precision": float("nan"), "Recall": float("nan")} # F1, Precision, Recall not computed here
    
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open('result/model_test.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch, accuracy, rmse, auc))
    
    return rmse, auc, accuracy


def load_snapshot(model, filename):
    # Ensure the file exists before loading
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model snapshot not found at {filename}")
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def get_status(epoch):
    '''
    An example of getting student's knowledge status
    :return:
    '''
    net = Net(student_n, exer_n, knowledge_n)
    model_path = 'model/model_epoch' + str(epoch)
    load_snapshot(net, model_path)       # load model
    net.eval()
    
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    # Get concept names for student_mastery.csv header
    config_file = 'data/config.txt'
    with open(config_file) as i_f:
        i_f.readline()
        _, _, knowledge_n_str = i_f.readline().split(',')
        knowledge_dim = int(knowledge_n_str)

    # Load concept universe to get concept names
    concept_universe_path = 'MultipleFiles/concept_universe.json'
    concept_names = []
    if os.path.exists(concept_universe_path):
        with open(concept_universe_path, 'r') as f:
            concept_names = json.load(f)
        # Ensure concept_names matches knowledge_dim
        if len(concept_names) != knowledge_dim:
            print(f"Warning: Mismatch between knowledge_dim ({knowledge_dim}) and concept_universe size ({len(concept_names)}). Using generic concept names.")
            concept_names = [f"Concept_{i+1}" for i in range(knowledge_dim)]
    else:
        print("Warning: MultipleFiles/concept_universe.json not found. Using generic concept names.")
        concept_names = [f"Concept_{i+1}" for i in range(knowledge_dim)]


    student_mastery_data = []
    for stu_id_idx in range(student_n): # stu_id_idx is 0-indexed
        # get knowledge status of student with stu_id (index)
        status = net.get_knowledge_status(torch.LongTensor([stu_id_idx])).tolist()[0]
        row = {'student_id': f"S{stu_id_idx + 1:02d}"} # Convert back to S01, S02 format
        for i, concept_mastery in enumerate(status):
            row[concept_names[i]] = concept_mastery
        student_mastery_data.append(row)
    
    student_mastery_df = pd.DataFrame(student_mastery_data)
    student_mastery_df.to_csv(os.path.join(reports_dir, "student_mastery.csv"), index=False)
    print(f"✅ Student mastery saved to {reports_dir}/student_mastery.csv")


def get_exer_params(epoch):
    '''
    An example of getting exercise's parameters (knowledge difficulty and exercise discrimination)
    :return:
    '''
    net = Net(student_n, exer_n, knowledge_n)
    model_path = 'model/model_epoch' + str(epoch)
    load_snapshot(net, model_path)    # load model
    net.eval()
    
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    item_difficulty_data = []
    for exer_id_idx in range(exer_n): # exer_id_idx is 0-indexed
        # get knowledge difficulty and exercise discrimination of exercise with exer_id (index)
        k_difficulty, e_discrimination = net.get_exer_params(torch.LongTensor([exer_id_idx]))
        
        # For simplicity, we'll use the mean of k_difficulty as item difficulty
        # and e_discrimination as is.
        # The original NCDM `item_difficulty.csv` only has 'difficulty'.
        # We'll use the mean of k_difficulty for 'difficulty'.
        item_difficulty_data.append({
            'item_id': f"Q{exer_id_idx + 1}", # Convert back to Q1, Q2 format
            'difficulty': k_difficulty.mean().item() # Use mean of knowledge difficulties
            # 'discrimination': e_discrimination.item() # If you want to save discrimination too
        })
    
    item_difficulty_df = pd.DataFrame(item_difficulty_data)
    item_difficulty_df.to_csv(os.path.join(reports_dir, "item_difficulty.csv"), index=False)
    print(f"✅ Item difficulty saved to {reports_dir}/item_difficulty.csv")


if __name__ == '__main__':
    if (len(sys.argv) != 2) or (not sys.argv[1].isdigit()):
        print('command:\n\tpython predict.py {epoch}\nexample:\n\tpython predict.py 70')
        sys.exit(1)

    # global student_n, exer_n, knowledge_n
    config_file = 'data/config.txt'
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found. Please create it with student_n,exer_n,knowledge_n.")
        sys.exit(1)

    with open(config_file) as i_f:
        i_f.readline() # Skip header
        student_n, exer_n, knowledge_n = list(map(int, i_f.readline().split(','))) # Use int for conversion

    epoch_to_test = int(sys.argv[1])
    rmse, auc, accuracy = test(epoch_to_test)
    get_status(epoch_to_test)
    get_exer_params(epoch_to_test)