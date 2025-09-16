import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
import os # Added for path handling
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, precision_score, recall_score # Added for full metrics
from ncdm_src.data_loader import TrainDataLoader, ValTestDataLoader # Adjusted import path
from ncdm_src.model import Net # Adjusted import path
import pandas as pd # Added for saving training metrics

os.makedirs('result', exist_ok=True)
# These will be read from config.txt
exer_n = 0
knowledge_n = 0
student_n = 0
# can be changed according to command parameter
device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
epoch_n = 5


def train():
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    print('training model...')

    loss_function = nn.NLLLoss()
    
    # For saving training metrics
    training_losses = []
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            if input_stu_ids is None: # Handle end of data
                break
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0
        
        avg_epoch_loss = running_loss / (batch_count % 200) if (batch_count % 200) != 0 else 0.0
        if batch_count % 200 == 0 and batch_count > 0: # If last batch was a full 200
             avg_epoch_loss = running_loss / 200
        training_losses.append(avg_epoch_loss)

        # validate and save current model every epoch
        rmse, auc, acc, f1, prec, rec = validate(net, epoch)
        save_snapshot(net, 'model/model_epoch' + str(epoch + 1))
    
    # Save training metrics to reports/training_metrics.csv
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    pd.DataFrame({"epoch": list(range(1,epoch_n+1)), "loss": training_losses}).to_csv(os.path.join(reports_dir, "training_metrics.csv"), index=False)
    print(f"✅ Training metrics saved to {reports_dir}/training_metrics.csv")


def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    # No need to create a new Net instance here, use the passed model
    net = model # Use the model passed from train()
    print('validating model...')
    data_loader.reset()
    
    net = net.to(device)
    net.eval() # Set to evaluation mode

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        if input_stu_ids is None: # Handle end of data
            break
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        
        with torch.no_grad(): # Disable gradient calculation for validation
            output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(mean_squared_error(label_all, pred_all)) # Using sklearn's RMSE
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    # compute F1, Precision, Recall
    y_pred_binary = [1 if p >= 0.5 else 0 for p in pred_all]
    f1 = f1_score(label_all, y_pred_binary, zero_division=0)
    prec = precision_score(label_all, y_pred_binary, zero_division=0)
    rec = recall_score(label_all, y_pred_binary, zero_division=0)

    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc, accuracy, f1, prec, rec


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        sys.exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])

    # global student_n, exer_n, knowledge_n, device
    config_file = 'data/config.txt'
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found. Please create it with student_n,exer_n,knowledge_n.")
        sys.exit(1)

    with open(config_file) as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(int, i_f.readline().split(',')))

    train()