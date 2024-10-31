import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import time
import random
import matplotlib.pyplot as plt

class ComplexCNN(nn.Module):
  def __init__(self, args):
    super(ComplexCNN, self).__init__()
    self.args = args
    
    # First convolutional block
    self.conv1 = nn.Conv1d(in_channels =   5, out_channels = 16, kernel_size = 3, padding = "same")
    self.bn1 = nn.BatchNorm1d(16)

    # Fully connected layers
    self.pool = nn.MaxPool1d(2, 2)
    self.fc1 = nn.Linear(16 * 64, 256)
    self.fc2 = nn.Linear(256, 128)
    
    self.db1 = nn.Linear(128, 64)
    self.db2 = nn.Linear(64, 64)
    self.db3 = nn.Linear(64, 1)
    
    self.sb1 = nn.Linear(128, 64)
    self.sb2 = nn.Linear(64, 64)
    self.sb3 = nn.Linear(64, 1)

    self.do1 = nn.Dropout(0.20)
    self.do2 = nn.Dropout(0.10)

  def forward(self, x):
    # First convolutional block
    x = self.pool(torch.relu(self.bn1(self.conv1(x))))
    x = x.view(x.size(0), -1)

    # Fully connected layers with dropout
    x = torch.relu(self.fc1(x))
    x = self.do1(x)
    x = torch.relu(self.fc2(x))
    x = self.do2(x)
    
    d = torch.relu(self.db1(x))
    d = torch.relu(self.db2(d))
    d = self.db3(d).reshape(-1, 1)
    
    s = torch.relu(self.sb1(x))
    s = torch.relu(self.sb2(s))
    s = self.sb3(s).reshape(-1, 1)
    
    x = torch.cat((d, s), dim = 1)
    return x

  def set_seed(self, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False 
      
  def count_parameters(self):
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(f'Total Trainable Parameters: {total_params}')
    return total_params
            
  @staticmethod
  def rmse_per_label(y_true, y_pred):
    rmse_label_1 = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_label_2 = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    return rmse_label_1, rmse_label_2

  def train_model(self, sub, ph, seed, train_loader, validation_loader, test_loader, num_epochs=10, learning_rate=1e-4, device='cpu'):
    self.set_seed(seed)
    self.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    valSetRatio = self.args["mapping"]["phyAware"]["valSetRatio"]

    best_loss = np.Inf
    for epoch in range(num_epochs):
        start_time = time.time()
        self.train()
        running_loss = 0.0
        total_batches = len(train_loader)  # Get total number of batches

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs, labels

            optimizer.zero_grad()
            outputs = self(inputs)
            # loss_dbp = criterion(outputs[:, 0], labels[:, 0])
            # loss_sbp = criterion(outputs[:, 1], labels[:, 1])
            # loss = loss_sbp + loss_dbp
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print batch progress
            # print(f'Batch [{batch_idx + 1}/{total_batches}] processed', flush=True, end = " " * 20 + "\r")

        print(f'Time Taken By Epoch: {time.time() - start_time}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / total_batches}', flush=True)

        error = self.evaluate_model(sub, ph, train_loader, validation_loader, test_loader, saveReport = False, writeReport = True, device = device)
        if error[0] * valSetRatio + error[1] < best_loss:
            best_loss = error[0] * valSetRatio + error[1]
            self.evaluate_model(sub, ph, train_loader, validation_loader, test_loader, saveReport = True, writeReport = False, device = device)
            torch.save(self.state_dict(), f'/esplabdata/yaleBiozData/analyticDataBase/models/model_{sub}_{ph[0]}.pth')
            print(f"Model saved as model_{sub}_{ph[0]}.pth", flush = True)

        print()
    return self

  def plot_pearson_and_bland_altman(self, sub, ph, name, y_true, y_pred, ax, row_idx):
      for label_num in range(2):
          tag = "dia" if label_num == 0 else "sys"
          pearson_corr_label, _ = pearsonr(y_true[:, label_num], y_pred[:, label_num])

          mean = np.mean([y_true[:, label_num], y_pred[:, label_num]], axis=0)
          diff = y_true[:, label_num] - y_pred[:, label_num]

          # Pearson plot
          ax[row_idx, label_num * 2].scatter(y_true[:, label_num], y_pred[:, label_num], alpha=0.7, edgecolors='k')
          ax[row_idx, label_num * 2].plot([y_true[:, label_num].min(), y_true[:, label_num].max()],
                                           [y_true[:, label_num].min(), y_true[:, label_num].max()],
                                           color='red', linestyle='--', label='y = x')
          ax[row_idx, label_num * 2].set_title(f'{name} Pearson: Label {tag}\nCorr = {pearson_corr_label:.3f}', fontsize=14)
          ax[row_idx, label_num * 2].set_xlabel('True Values', fontsize=12)
          ax[row_idx, label_num * 2].set_ylabel('Predicted Values', fontsize=12)
          ax[row_idx, label_num * 2].grid()
          ax[row_idx, label_num * 2].legend()

          # Bland-Altman plot
          ax[row_idx, label_num * 2 + 1].scatter(mean, diff, alpha=0.7, edgecolors='k')
          ax[row_idx, label_num * 2 + 1].axhline(np.mean(diff), color='gray', linestyle='--', label='Mean Difference')
          ax[row_idx, label_num * 2 + 1].axhline(np.mean(diff) + 1.96 * np.std(diff), color='red', linestyle='--', label='Upper Limit')
          ax[row_idx, label_num * 2 + 1].axhline(np.mean(diff) - 1.96 * np.std(diff), color='red', linestyle='--', label='Lower Limit')
          ax[row_idx, label_num * 2 + 1].set_title(f'{name} Bland-Altman: Label {tag}\nCorr = {pearson_corr_label:.3f}', fontsize=14)
          ax[row_idx, label_num * 2 + 1].set_xlabel('Mean of True and Predicted Values', fontsize=12)
          ax[row_idx, label_num * 2 + 1].set_ylabel('Difference (True - Predicted)', fontsize=12)
          ax[row_idx, label_num * 2 + 1].grid()
          ax[row_idx, label_num * 2 + 1].legend()

  def evaluate_model(self, sub, ph, train_loader, val_loader, test_loader, saveReport = False, writeReport = False, device='cpu'):
      self.eval()

      all_preds = []
      all_labels = []

      # Prepare figure for all plots
      fig, axs = plt.subplots(3, 4, figsize=(24, 18))  # 3 rows, 4 columns

      # Evaluate on train, validation, and test sets
      error = []
      for idx, (loader, name) in enumerate(zip([train_loader, val_loader, test_loader], ['Train', 'Validation', 'Test'])):
          all_preds = []
          all_labels = []
          with torch.no_grad():
              for inputs, labels in loader:
                  inputs, labels = inputs.to(device), labels.to(device)
                  outputs = self(inputs)
                  all_preds.append(outputs.cpu().numpy())
                  all_labels.append(labels.cpu().numpy())

          all_preds = np.vstack(all_preds)
          all_labels = np.vstack(all_labels)

          # RMSE calculations
          rmse_label_1, rmse_label_2 = self.rmse_per_label(all_labels, all_preds)
          if writeReport:
            reportDBP = f"DBP RMSE ({name.lower()[:2]}) -> {rmse_label_1:.3f}"
            reportSBP = f"SBP RMSE ({name.lower()[:2]}) -> {rmse_label_2:.3f}"
            print(reportDBP.ljust(28), "|", reportSBP.ljust(28), flush = True)
            
          error.append(np.power(rmse_label_1, 2) + np.power(rmse_label_2, 2))

          # Call the modified plotting function
          self.plot_pearson_and_bland_altman(sub, ph, name, all_labels, all_preds, axs, idx)
      
      if saveReport:
        plt.tight_layout()
        plt.savefig(f"/esplabdata/yaleBiozData/analyticDataBase/images/pear_bland_all_{sub}_{ph[0]}", facecolor = "white")
        
      plt.close()
      
      return error
