##IMPORT LIBS

import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

from sklearn.model_selection import KFold

from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv

from torch.nn.functional import softmax
from  torch_geometric.nn import GraphNorm
import matplotlib.pyplot as plt
from typing import Optional, List

import time
import os
import random
from rich.console import Console
from torch.utils.data import random_split

from rich.console import Console
from rich.table import Table
from rich.progress import track
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data

from torch.optim.lr_scheduler import StepLR

import logging
import json

import xgboost as xgb

import optuna

from datetime import datetime

from torch_geometric.utils import from_networkx

from rich.console import Console
from rich.table import Table
from rich.columns import Columns

import statistics

##CHECK OS
system = os.name

  ##CHOOSE DATASET DIR PATH
dataset_path = "/home/nzheludkov/phd/wiregnn28nm/dataset/"

##GET BLOCK LIST
block_list = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

class loadBlockfromDataset_v2:

    def __init__(self, name, root=None):

        ##CHECK OS
        self.system = os.name
        self.dataset_path = "/home/nzheludkov/phd/wiregnn28nm/dataset/"

        self.name = name
        self.graph_path = f'{self.dataset_path}{self.name}/netlist_to_graph_net.txt'
        self.net_feat_path = f'{self.dataset_path}{self.name}/net_feat.csv'
        
        ##create graph from netlist
        G = nx.Graph()
        file_path = self.graph_path
        with open(file_path, 'r') as f:
            code = f.read()
        exec(code)
        self.G = G

        ##MAPPING NODE NAMES TO INDEX
        self.node_to_index = {node: i for i, node in enumerate(self.G.nodes())}

        ##Replace node names in edges with integer indices
        self.edges = [(self.node_to_index[u], self.node_to_index[v]) for u, v in self.G.edges()]

        ##Convert edges to a 2xN tensor (edge_index)
        self.edge_index = torch.tensor(self.edges, dtype=torch.long).t()

        ##load feats and labels
        self.df = pd.read_csv(self.net_feat_path, sep=';').fillna(0)
        self.df['length'] = self.df['length'].mask(self.df['length'] == 0, self.df['hpwl'])

        ##CREATE NODE(NET) FEATURE and LABELS(NET LENGTH)
        self.x = torch.zeros(G.number_of_nodes(),9)
        self.y = torch.zeros(G.number_of_nodes(),1)

        x_mean = torch.tensor([ 3.0312,  2.9923,  1.4407,  4.5219, 17.2115, 11.3378, 16.0953, 13.7574, 1.7583])
        x_std =  torch.tensor([ 1.2838,  5.2747,  1.0134, 11.7338, 35.3156, 29.0777, 33.7732, 30.8073, 5.2603])

        for i,j in enumerate(G.nodes()):
            new_value = self.df[self.df['net'].isin([j])].values[0][1:len(self.df.columns)].astype(np.float64)
            x_feat_part1 = torch.tensor(new_value[0:4], dtype=torch.float64)
            x_feat_part2 = torch.tensor(new_value[8:13], dtype=torch.float64)
            self.x[i] =  torch.cat((x_feat_part1, x_feat_part2), dim=0)
            self.y[i] = torch.tensor(new_value[-1], dtype=torch.float64)

        self.x_original = self.x
        self.y_original = self.y
        self.block_area_original = torch.tensor(self.df['urx'].max() * self.df['ury'].max(), dtype=torch.float64)

        self.x = (self.x - x_mean) / x_std

        ##to pytorch geometric
        self.data = from_networkx(G)
        self.data.x = self.x
        self.data.y = self.y
        self.data.name = self.name

class training():
    
    def __init__(self, model_class, dataset, config, run_name=None):
        self.model_class = model_class
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = run_name or datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join("logs", self.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.split_dataset()
        self._setup_logging()
        self._save_config()

    def _setup_logging(self):
        """Initialize logging to write to a fresh log file for each run"""
        log_file = os.path.join(self.log_dir, "train.log")
        
        # Clear any existing handlers
        logging.root.handlers = []
        
        logging.basicConfig(
            filename=log_file,
            filemode='w',  # Overwrite existing log file
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            force=True  # Override any existing config (Python 3.8+)
        )
        logging.info(f"Initialized new training run: {self.run_name}")

    def _save_config(self):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    def split_dataset_random(self):

        # Задаем размеры выборок (в количестве примеров или долях)
        train_ratio = 0.8  # 70% train
        val_ratio = 0.1   # 15% val
        test_ratio = 0.1   # 15% test

        num_total = len(self.dataset)
        num_train = int(train_ratio * num_total)
        num_val = int(val_ratio * num_total)
        num_test = num_total - num_train - num_val

        # Разделяем с фиксированным random_seed для воспроизводимости
        torch.manual_seed(42)  # Фиксируем seed!
        train_dataset, val_dataset, test_dataset = random_split(
        self.dataset, [num_train, num_val, num_test]
        )

        #data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1)
        self.test_loader = DataLoader(test_dataset, batch_size=1)

    def split_dataset(self):

        train_dataset = []
        val_dataset = []
        test_dataset = []

        for block in self.dataset.data_list:
            if block.name in  self.config["block_list_val"]:
                val_dataset.append(block)
            elif block.name in self.config["block_list_test"]:
                test_dataset.append(block)
            else:
                train_dataset.append(block)

        #data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1)
        self.test_loader = DataLoader(test_dataset, batch_size=1)
        

    def compute_metrics(self, pred, target):
        mae = torch.mean(torch.abs(pred - target)).item()
        mape = torch.mean(torch.abs((pred - target) / (target + 1e-8))).item() * 100
        ss_res = torch.sum((pred - target) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        r2 = r2.item()
        
        return mae, mape, r2

    def train(self, model, train_loader, optimizer, criterion):
    
        total_loss = 0
        all_preds, all_targets = [], []
    
        for batch in train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
    
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
    
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            all_preds.append(out.detach())
            all_targets.append(batch.y.detach())
    
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        mae, mape, r2 = self.compute_metrics(preds, targets)
    
        return total_loss / len(train_loader), mae, mape, r2


    def evaluate(self, model, val_loader, criterion):
        total_loss = 0
        all_preds, all_targets = [], []
    
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                total_loss += loss.item()
    
                all_preds.append(out)
                all_targets.append(batch.y)
    
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        mae, mape, r2 = self.compute_metrics(preds, targets)
    
        return total_loss / len(val_loader), mae, mape, r2

    def main_train(self):
        
        train_losses, val_losses = [], []
        train_maes, val_maes = [], []
        train_mapes, val_mapes = [], []

        self.model = self.model_class().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        criterion = torch.nn.MSELoss()
        #scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

        logging.info(f'Model architecture = {self.model}')
        
        best_val_loss = float("inf")
        best_val_mae = float("inf")

        epoch_time_start = time.time()
        
        for epoch in range(1, self.config["epochs"] + 1):
            epoch_time_start = time.time()
            train_loss, train_mae = self.train(self.model, self.train_loader, optimizer, criterion)
            val_loss, val_mae = self.evaluate(self.model, self.val_loader, criterion)
    
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
            train_maes.append(train_mae)
            val_maes.append(val_mae)

            #scheduler.step()
        
            #train_mapes.append(train_mape)
            #val_mapes.append(val_mape)
            epoch_time_end= time.time()
            epoch_time = round(epoch_time_end - epoch_time_start, 0)

            logging.info(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} | "
                  f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}  | Epoch Time: {epoch_time}")
        
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} | "
                  f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}  | Epoch Time: {epoch_time}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "best_model.pth"))   

            if val_mae < best_val_mae:
                best_val_mae = val_mae


        #test
        test_loss, test_mae = self.evaluate(self.model, self.test_loader, criterion)
        logging.info(f"Test Results: " f"Loss: {test_loss:.4f} | MAE: {test_mae:.2f} | Epoch Time: {epoch_time}")
        logging.info(f"Training complete. Best Val Loss = {best_val_loss:.4f}, Test Loss = {test_loss:.4f}\n")
        print(f"Training complete. Test Loss = {test_loss:.4f}")
        print(f"Test Results: " f"Loss: {test_loss:.4f} | MAE: {test_mae:.2f}")

        self.test_mae = test_mae

        metrics = {
            "last_train_loss": round(train_loss,2),
            "last_train_mae" :  round(train_mae,2),
            "last_val_loss":  round(val_loss,2),
            "last_val_mae":  round(val_mae,2),
            "best_val_loss":  round(best_val_loss,2),
            "best_val_mae":  round(best_val_mae,2),
            "test_loss":  round(test_loss,2),
            "test_mae":  round(test_mae,2)
        }
        with open(os.path.join(self.log_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        #pd.DataFrame([metrics]).to_csv(os.path.join(self.log_dir, "metrics.csv"), mode='a', header=True)

        #add attribues for drawing
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_maes = train_maes
        self.val_maes= val_maes

        epochs = range(1, self.config["epochs"]  +1 ) 


        ##draw plots
        plt.figure(figsize=(18, 5))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        
        # MAE
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_maes, label='Train MAE')
        plt.plot(epochs, val_maes, label='Val MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE over Epochs')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_metrics.png"), dpi=300, bbox_inches='tight')
        plt.show()

    def test(self, block_list_test):

        for block in block_list_test:
        
            block = loadBlockfromDataset_v2(block)
            
            y_true = block.data.y
            y_pred = self.model.forward(block.data.x.to(device), block.data.edge_index.to(device)).to(device)
            y_mean = block.data.y.mean()
            y_pred = y_pred.detach().cpu()
        
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            r2 = r2_score(y_true, y_pred)
            
            console = Console(width=80)
            table = Table(title=f"Сравнение точности моделей предсказания длины цепей в блоке {block.name}", show_lines=True)
            table.add_column("Модель")
            table.add_column("MAE, мкм", justify="center")
            table.add_column("MAPE, %", justify="center")
            table.add_column("R2", justify="center")
            table.add_column("MEAN LENGTH, мкм", justify="center")
            table.add_row(f"{self.model_class}", f"{mae:.2f}", f"{mape:.2f}", f"{r2:.2f}", f"{y_mean:.2f}")
            console.print(Columns([table]))

    def cross_validation(self):

        kf = KFold(n_splits=8, shuffle=True, random_state=42)
        val_mae_list = []
        val_mape_list = []
        val_r2_list = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset.data_list)):
            
            logging.info(f"\n--- Fold {fold+1} ---")
            print(f"\n--- Fold {fold+1} ---")
            
            train_data = [self.dataset.data_list[i] for i in train_idx]
            val_data = [self.dataset.data_list[i] for i in val_idx]

            print(f'Validation blocks for Fold {fold+1}: {val_data}')

            train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

            train_losses, val_losses = [], []
            train_maes, val_maes = [], []
            train_mapes, val_mapes = [], []
            train_r2, val_r2 = [], []
    
            self.model = self.model_class().to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
            criterion = torch.nn.MSELoss()
            scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    
            logging.info(f'Model architecture = {self.model}')

        
            best_val_loss = float("inf")
            best_val_mae = float("inf")
            best_val_mape = float("inf")
            
            epoch_time_start = time.time()
            
            for epoch in range(1, self.config["epochs"] + 1):
                epoch_time_start = time.time()
                train_loss, train_mae, train_mape, train_r2 = self.train(self.model, self.train_loader, optimizer, criterion)
                val_loss, val_mae, val_mape, val_r2 = self.evaluate(self.model, self.val_loader, criterion)
        
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            
                train_maes.append(train_mae)
                val_maes.append(val_mae)
    
                scheduler.step()
            
                #train_mapes.append(train_mape)
                #val_mapes.append(val_mape)
                epoch_time_end= time.time()
                epoch_time = round(epoch_time_end - epoch_time_start, 0)
    
                logging.info(f"Epoch {epoch:03d} | "
                      f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} | "
                      f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}  | Epoch Time: {epoch_time}")

                if (epoch % 10 == 0 or epoch == 1):
                    print(f"Epoch {epoch:03d} | "
                          f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} | "
                          f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}  | Epoch Time: {epoch_time}")
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.log_dir, "best_model.pth"))   
    
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    
                if val_mape < best_val_mape:
                    best_val_mape = val_mape

            
            val_mae_list.append(best_val_mae)
            val_mape_list.append(val_mape)
            val_r2_list.append(val_r2)
            
            logging.info(f'Training ok Fold {fold+1} ends. Best VAL MAE: {best_val_mae} Best VAL MAPE: {best_val_mape}')
            print(f'Training ok Fold {fold+1} ends. Best VAL MAE: {best_val_mae}  Best VAL MAPE: {best_val_mape}')

        val_mae_avg = statistics.mean(val_mae_list)
        val_mae_std = statistics.stdev(val_mae_list)

        val_mape_avg = statistics.mean(val_mape_list)
        val_mape_std = statistics.stdev(val_mape_list)

        val_r2_avg = statistics.mean(val_r2_list)
        val_r2_std = statistics.stdev(val_r2_list)
        
        print(f'AVG MAE: {val_mae_avg} STD MAE: {val_mae_std}')
        print(f'AVG MAPE: {val_mape_avg} STD MAPE: {val_mape_std}')
        print(f'AVG R2: {val_r2_avg} STD R2: {val_r2_std}')

##CREATE DATASET CLASS
class graph_dataset(InMemoryDataset):
    
    def __init__(self, root, data_list: Optional[List[Data]] = None, transform=None):
        
        self._data_list = data_list if data_list is not None else []
        super().__init__(root, transform)
        
        if data_list is not None:
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data, self.slices), self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def data_list(self) -> List[Data]:
        """Возвращает список объектов Data"""
        if hasattr(self, '_data_list') and self._data_list:
            return self._data_list
        
        # Если _data_list не существует, восстанавливаем из data и slices
        data_list = []
        for i in range(len(self)):
            data = self.get(i)
            data_list.append(data)
        self._data_list = data_list
        return self._data_list

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Сохраняем то, что передали
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])

    def save(self):
        # Сохраняем данные в processed_paths[0]
        torch.save((self.data, self.slices), self.processed_paths[0])
      
##STORE DATASET

pyg_graphs = []

for block in block_list:
    block = loadBlockfromDataset_v2(block)
    pyg_graphs.append(block.data)
    print(pyg_graphs)

dataset = graph_dataset(root = './pyg_dataset_v3', data_list = pyg_graphs)

class linear_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = Linear(9, 128)
        self.linear2 = Linear(128, 128)
        self.linear3 = Linear(128, 1)

        self.dropout = nn.Dropout(p=0)

    def forward(self, x, edge_index):

        h = self.linear1(x)
        h = torch.nn.functional.relu(h)
        h = self.dropout(h)

        h = self.linear2(h)
        h = torch.nn.functional.relu(h)
        h = self.dropout(h)

        h = self.linear3(h)
        return h

kf = KFold(n_splits=8, shuffle=True, random_state=42)

all_mae = 0
all_mape = 0
all_r2 = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset.data_list)):
    
    train_data = [dataset.data_list[i] for i in train_idx]
    val_data = [dataset.data_list[i] for i in val_idx]
    
    print(f'Validation blocks for Fold {fold+1}: {val_data}')
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_mapes, val_mapes = [], []
    train_r2, val_r2 = [], []

    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 400,
        'max_depth': 20,
        'learning_rate': 0.1,
        #'subsample': 0.9,
        #'colsample_bytree': 0.9,
        #'gamma': 0.1,
        #'reg_alpha': 0.9,
        #'reg_lambda': 0.9
    }

    xgb_model = xgb.XGBRegressor(**params)

    for batch in train_loader:
        xgb_model.fit(batch.x, batch.y)

    mae_sum = 0
    mape_sum = 0
    r2_sum =0
    
    for batch in val_loader:
        y_pred = xgb_model.predict(batch.x)
        y_true = batch.y
        mae_sum = mae_sum + mean_absolute_error(y_true, y_pred)
        mape_sum = mape_sum + mean_absolute_percentage_error(y_true, y_pred)
        r2_sum = r2_sum + r2_score(y_true, y_pred)

    mae_f = mae_sum / len(val_loader)
    mape_f =  mape_sum / len(val_loader)
    r2_f = r2_sum / len(val_loader)

    all_mae = all_mae + mae_f
    all_mape = all_mape + mape_f
    all_r2 = all_r2 + r2_f

    print(mae_sum / len(val_loader), mape_sum / len(val_loader), r2_sum / len(val_loader))

print(all_mae/8, all_mape / 8, all_r2 / 8)


##CREATE GCN MODEL CLASS

class gcn_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = GCNConv(9, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)

        self.linear2 = Linear(25, 128)
        self.linear3 = Linear(128, 128)
        self.linear4 = Linear(128, 1)

        self.dropout = nn.Dropout(p=0)
    
    def forward(self, x, edge_index):
                
        h = self.conv1(x, edge_index)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)
        h1 =h 

        h = self.conv2(h, edge_index)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)
        h2 =h 

        h = self.conv3(h, edge_index)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)

        h = torch.cat((x,h), dim=1)

        h = self.linear2(h)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)

        h = self.linear3(h)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)

        h = self.linear4(h)
        
        return h

##CREATE GAT MODEL CLASS

class gat_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = GATConv(in_channels = 9, out_channels = 8, heads = 4, concat = True)
        self.conv2 = GATConv(in_channels = 32, out_channels = 8, heads = 4, concat = True)
        self.conv3 = GATConv(in_channels = 32, out_channels = 8, heads = 4, concat = True)

        self.linear2 = Linear(41, 128)
        self.linear3 = Linear(128, 128)
        self.linear4 = Linear(128, 1)

        self.dropout = nn.Dropout(p=0)
    
    def forward(self, x, edge_index):
                
        h = self.conv1(x, edge_index)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)
        h1 =h 

        h = self.conv2(h, edge_index)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)
        h2 =h 

        h = self.conv3(h, edge_index)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)

        h = torch.cat((x,h), dim=1)

        h = self.linear2(h)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)

        h = self.linear3(h)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)

        h = self.linear4(h)
        
        return h

##CREATE SAGE MODEL CLASS

class sage_mean_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = SAGEConv(9, 32, aggr='mean')
        self.conv2 = SAGEConv(32, 32, aggr='mean')
        self.conv3 = SAGEConv(32, 32, aggr='mean')

        self.linear2 = Linear(41, 128)
        self.linear3 = Linear(128, 128)
        self.linear4 = Linear(128, 1)

        self.dropout = nn.Dropout(p=0)
    
    def forward(self, x, edge_index):
                
        h = self.conv1(x, edge_index)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)
        h1 =h 

        h = self.conv2(h, edge_index)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)
        h2 =h 

        h = self.conv3(h, edge_index)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)

        h = torch.cat((x,h), dim=1)

        h = self.linear2(h)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)

        h = self.linear3(h)
        h = self.dropout(h)
        h = torch.nn.functional.relu(h)

        h = self.linear4(h)
        
        return h
