import torch
import torch.nn as nn
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class TreeModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    def tensor_to_numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def numpy_to_tensor(self, array):
        return torch.from_numpy(array).to(next(self.parameters()).device)

    def flatten_input(self, x):
        return x.view(x.size(0), -1)


class XGBoost(TreeModelBase):
    def __init__(self, configs):
        super().__init__()
        self.model = XGBRegressor(
            n_estimators=configs.n_estimators,
            max_depth=configs.max_depth,
            learning_rate=configs.learning_rate,
            tree_method=configs.tree_method
        )
        
    def forward(self, x):
        flattened_x = self.flatten_input(x)
        features = self.tensor_to_numpy(flattened_x)  # 转换为numpy

        preds = self.model.predict(features)
        return self.numpy_to_tensor(preds).squeeze()


class LightGBM(TreeModelBase):
    def __init__(self, configs):
        super().__init__()
        self.model = LGBMRegressor(
            num_leaves=31,
            max_depth=-1,
            learning_rate=0.1,
            n_estimators=100
        )
        
    def forward(self, x):
        flattened_x = self.flatten_input(x)
        features = self.tensor_to_numpy(flattened_x)

        preds = self.model.predict(features)
        return self.numpy_to_tensor(preds).squeeze()
