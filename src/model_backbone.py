import os
import copy
import math
import json
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from tqdm import tqdm
import time

from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
from qlib.model.base import Model

from types import SimpleNamespace

# vis
import plotly
from qlib.contrib.report.analysis_model.analysis_model_performance import model_performance_graph

from src.models.PDF import PDF
from src.models.PatchTST import PatchTST
from src.models.TimeMixer import TimeMixer
from src.models.TimesNet import TimesNet
from src.models.SegRNN import SegRNN
from src.models.diffusion_stock import DiffStock
from src.models.Crossformer import Crossformer
from src.models.LSTM import LSTM
from src.models.GRU import GRU
from src.models.Transformer import Transformer
# from src.models.Mamba import mamba
from src.models.TCN import TCN
from src.models.GAT import GAT
from src.models.GCN import GCN
from src.models.TimeBridge import TimeBridge

device = "cuda:1" if torch.cuda.is_available() else "cpu"


class RankMSELoss(nn.Module):
    def __init__(self, rank_weight=3.0, mse_weight=1.0):
        super(RankMSELoss, self).__init__()
        self.rank_weight = rank_weight
        self.mse_weight = mse_weight

    def forward(self, pred, label):
        mse_loss = (pred - label).pow(2).mean()

        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        label_diff = label.unsqueeze(1) - label.unsqueeze(0)
    
        rank_loss = F.relu(- (pred_diff * label_diff))
        
        rank_loss = rank_loss.sum(dim=[1])
        rank_loss = rank_loss.mean()

        combined_loss = self.rank_weight * rank_loss + self.mse_weight * mse_loss
        
        return combined_loss
    

class QniverseModel(Model):
    def __init__(
            self,
            model_config,
            model_type="WFTNet",
            lr=1e-3,
            n_epochs=500,
            early_stop=50,
            smooth_steps=5,
            max_steps_per_epoch=None,
            freeze_model=False,
            model_init_state=None,
            seed=None,
            logdir=None,
            eval_train=True,
            eval_test=False,
            avg_params=True,
            **kwargs,
    ):

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.logger = get_module_logger("Qniverse")
        self.logger.info("Qniverse Model...")

        self.model_type = model_type
        self.model = eval(model_type)(SimpleNamespace(**model_config)).to(device)
        if model_init_state:
            self.model.load_state_dict(torch.load(model_init_state, map_location="cpu")["model"])
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)
        else:
            self.logger.info("# model params: %d" % sum([p.numel() for p in self.model.parameters()]))

        self.optimizer = optim.Adam(list(self.model.parameters()), lr=lr)

        self.model_config = model_config
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.smooth_steps = smooth_steps
        self.max_steps_per_epoch = max_steps_per_epoch
        self.seed = seed
        self.logdir = logdir
        self.eval_train = eval_train
        self.eval_test = eval_test
        self.avg_params = avg_params

        self.criterion = RankMSELoss()

        if self.logdir is not None:
            if os.path.exists(self.logdir):
                self.logger.warn(f"logdir {self.logdir} is not empty")
            os.makedirs(self.logdir, exist_ok=True)

        self.global_step = -1

    def train_epoch(self, data_set):

        self.model.train()

        data_set.train()

        data_set.setup_data()
        max_steps = len(data_set)
        if self.max_steps_per_epoch is not None:
            max_steps = min(self.max_steps_per_epoch, max_steps)

        count = 0
        total_loss = 0
        total_count = 0

        for batch in tqdm(data_set, total=max_steps):
            count += 1
            if count > max_steps:
                break

            self.global_step += 1

            data, label, index = batch["data"], batch["label"], batch["index"]

            feature = data[:, :, : -1]
            
            feature = torch.nan_to_num(feature, nan=0.0)
            label = torch.nan_to_num(label, nan=0.0)

            # feature [batch_size, seq_len, num_fea]
            pred = self.model(feature).squeeze()
            
            pred = torch.nan_to_num(pred, nan=0.0)

            # pred [batch_size, horizon]
            loss = self.criterion(pred, label)
            # print(pred, label)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_count += len(pred)

        total_loss /= total_count

        return total_loss

    def test_epoch(self, data_set, return_pred=False):

        self.model.eval()
        data_set.eval()

        preds = []
        metrics = []
        total_inference_time = 0.0
        for batch in tqdm(data_set):
            data, label, index = batch["data"], batch["label"], batch["index"]

            feature = data[:, :, : -1]

            feature = torch.nan_to_num(feature, nan=0.0)
            label = torch.nan_to_num(label, nan=0.0)
            with torch.no_grad():
                start_time = time.time()
                pred = self.model(feature).squeeze()
                end_time = time.time()
                pred = torch.nan_to_num(pred, nan=0.0)

            total_inference_time += end_time - start_time

            X = np.c_[
                pred.cpu().numpy(),
                label.cpu().numpy(),
            ]
            columns = ["score", "label"]

            pred = pd.DataFrame(X, index=index.cpu().numpy(), columns=columns)

            metrics.append(evaluate(pred))

            if return_pred:
                preds.append(pred)

        metrics = pd.DataFrame(metrics)
        metrics = {
            "InfT": total_inference_time / len(data_set),
            "MSE": metrics.MSE.mean(),
            "MAE": metrics.MAE.mean(),
            "IC": metrics.IC.mean(),
        }


        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index)
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)
        print("preds")
        print(preds)
        return metrics, preds
    
    def fit(self, dataset, evals_result=dict()):
        
        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])

        best_score = -1
        best_epoch = 0
        stop_rounds = 0
        best_params = {
            "model": copy.deepcopy(self.model.state_dict()),
        }
        params_list = {
            "model": collections.deque(maxlen=self.smooth_steps),
        }
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["test"] = []

        # train
        self.global_step = -1

        for epoch in range(self.n_epochs):
            self.logger.info("Epoch %d:", epoch)

            self.logger.info("training...")
            self.train_epoch(train_set)

            self.logger.info("evaluating...")
            # average params for inference
            params_list["model"].append(copy.deepcopy(self.model.state_dict()))
            self.model.load_state_dict(average_params(params_list["model"]))

            valid_metrics = self.test_epoch(valid_set)[0]
            evals_result["valid"].append(valid_metrics)
            self.logger.info("\tvalid metrics: %s" % valid_metrics)

            if self.eval_test:
                test_metrics = self.test_epoch(test_set)[0]
                evals_result["test"].append(test_metrics)
                self.logger.info("\ttest metrics: %s" % test_metrics)

            if valid_metrics["IC"] > best_score:
                best_score = valid_metrics["IC"]
                stop_rounds = 0
                best_epoch = epoch
                best_params = {
                    "model": copy.deepcopy(self.model.state_dict()),
                }
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    self.logger.info("early stop @ %s" % epoch)
                    break

            # restore parameters
            self.model.load_state_dict(params_list["model"][-1])

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_params["model"])

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)

        if self.logdir:
            self.logger.info("save model & pred to local directory")

            torch.save(best_params, self.logdir + "/model.bin")

            fig_list = model_performance_graph(preds, show_notebook=False)
            fig_name = [f"{self.model_type}_cumulative_return", f"{self.model_type}_distribution_return", f"{self.model_type}_IC",
                        f"{self.model_type}_monthly_IC", f"{self.model_type}_distribution_IC", f"{self.model_type}_auto_corr"]
            for i, fig in enumerate(fig_list):
                fig: plotly.graph_objs.Figure = fig
                fig.write_image(self.logdir + f'/{fig_name[i]}.jpg')
            print("Vis Finished!")

            preds.to_pickle(self.logdir + "/pred.pkl")

            info = {
                "config": {
                    "model_config": self.model_config,
                    "lr": self.lr,
                    "n_epochs": self.n_epochs,
                    "early_stop": self.early_stop,
                    "smooth_steps": self.smooth_steps,
                    "max_steps_per_epoch": self.max_steps_per_epoch,
                    "seed": self.seed,
                    "logdir": self.logdir,
                },
                "best_eval_metric": -best_score,  # NOTE: minux -1 for minimize
                "metric": metrics,
            }
            with open(self.logdir + "/info.json", "w") as f:
                json.dump(info, f)

            return preds, metrics

    def predict(self, dataset, segment="test"):

        test_set = dataset.prepare(segment)

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)
        # print("preds")
        # print(preds)

        return preds

def evaluate(pred):
    pred = pred.rank(pct=True)  # transform into percentiles
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff ** 2).mean()
    MAE = (diff.abs()).mean()
    IC = score.corr(label, method="pearson")
    return {"MSE": MSE, "MAE": MAE, "IC": IC}


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError("the %d-th model has different params" % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q
