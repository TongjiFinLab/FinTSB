import argparse

import qlib
import pandas as pd
import numpy as np
import ruamel.yaml as YAML
from tqdm import tqdm
from qlib.utils import init_instance_by_config
import warnings
warnings.filterwarnings('ignore')
import os,sys
os.chdir(sys.path[0])

from basktesting import calculate_portfolio_metrics, top_k_drop_strategy


# python train.py --config_file configs/config_patchtst.yaml

def main(seed, config_file="configs/config_patchtst.yaml"):

    with open(config_file) as f:
        yaml = YAML.YAML(typ='safe', pure=True)
        config = yaml.load(f)

    logdir = config["task"]["model"]["kwargs"]["logdir"]
    model_type = config["task"]["model"]["kwargs"]["model_type"]

    # initialize workflow
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        region=config["qlib_init"]["region"],
    )

    dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])

    if model_type not in ['XGBoost', 'LightGBM']:
        preds, metrics = model.fit(dataset)
    else:
        model.fit(dataset)
        preds = model.predict(dataset)
        preds = preds.to_frame(name='score')
        ori_index = preds.index
        preds = preds.reset_index(drop=True)
        label = dataset.prepare("test")[['label']].reset_index(drop=True)
        # preds = pd.concat([preds, dataset.prepare("test")[['label']]], axis=1, join='inner')
        preds = pd.concat([preds, label], axis=1)
        preds.set_index(ori_index, inplace=True)
        scores = preds['score'].values
        labels = preds['label'].values
        metrics = {"InfT": -1, "MSE": np.mean((labels - scores) ** 2), "MAE": np.mean(np.abs(labels - scores))}
        print(preds)


    per_df = preds.reset_index()
    per_df = per_df.sort_values(['datetime', 'instrument'])
    per_df['group_id'] = per_df.groupby(['datetime', 'instrument']).cumcount()
    assert (per_df.groupby(['datetime', 'instrument'])['group_id'].max() == 19).all(), "Error"

    r_metrics = {'IC': [], 'ICIR': [], 'RankIC': [], 'RankICIR': []}
    r_df = []

    strategy_config = config['strategy_config']
    for i in tqdm(range(20)):
        sub_df = per_df[per_df['group_id'] == i]
        sub_df = sub_df.set_index(['datetime', 'instrument'])
        sub_df = sub_df[['score', 'label']]

        # Ranking Metric
        ic = sub_df.groupby(level=0).apply(lambda group: group['score'].corr(group['label'])).to_numpy()
        icir = ic.mean() / ic.std()
        ic = ic.mean()
        rankic = sub_df.groupby(level=0).apply(lambda group: group['score'].corr(group['label'], method='spearman')).to_numpy()
        rankicir = rankic.mean() / rankic.std()
        rankic = rankic.mean()
        r_metrics['IC'].append(ic)
        r_metrics['ICIR'].append(icir)
        r_metrics['RankIC'].append(rankic)
        r_metrics['RankICIR'].append(rankicir)

        r_df.append(top_k_drop_strategy(sub_df, top_k=strategy_config['topk'], n=strategy_config['hold_thre'], 
                                   commission_rate=strategy_config['commission_rate']))
        
    r_df = pd.concat(r_df, ignore_index=True)
    # Portfolio Metric
    portfolio_metrics, df = calculate_portfolio_metrics(r_df)
    metrics.update(portfolio_metrics)
    rank_metrics = {key: np.mean(values) for key, values in r_metrics.items()}
    metrics.update(rank_metrics)

    df.to_csv(f'./{logdir}/backtest_result.csv')
    
    report = f"""
    **********************************
            Bachtest Report
    **********************************
         InfT:       {metrics['InfT']:.6f}
          MSE:       {metrics['MSE']:.4f}
          MAE:       {metrics['MAE']:.4f}
           IC:       {metrics['IC']:.4f}
         ICIR:       {metrics['ICIR']:.4f}
       RankIC:       {metrics['RankIC']:.4f}
     RankICIR:       {metrics['RankICIR']:.4f}
          ARR:       {metrics['ARR']:.4f}
         AVol:       {metrics['AVol']:.4f}
          MDD:       {metrics['MDD']:.4f}
          ASR:       {metrics['ASR']:.4f}
           IR:       {metrics['IC']:.4f}
    ***************************************
    """
    print(report)

    with open(f'./{logdir}/backtest_report.txt', 'w') as file:
        file.write(report)


if __name__ == "__main__":
    # set params from cmd
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--config_file", type=str, default="configs/config_wftnet.yaml", help="config file")
    args = parser.parse_args()
    main(**vars(args))
