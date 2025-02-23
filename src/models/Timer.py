# Timer: Generative Pre-trained Transformers Are Large Time Series Models
# https://arxiv.org/abs/2402.02368
# https://github.com/thuml/Large-Time-Series-Model

import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from tqdm import tqdm

df = pd.read_pickle("")

df.reset_index(inplace=True)
df.rename(columns={"datetime": "date"}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by=['instrument', 'date'], inplace=True)

lookback_length = 96
prediction_length = 1
start_forecast_date = pd.to_datetime('2000-10-09')
end_forecast_date = pd.to_datetime('2000-12-15')

model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)

all_predictions = []

for inst, inst_df in tqdm(df.groupby('instrument', sort=False)):
    inst_df.set_index('date', inplace=True)

    forecast_dates = inst_df.loc[
        (inst_df.index >= start_forecast_date) & 
        (inst_df.index <= end_forecast_date)
    ].index.unique()

    predictions = []
    for current_date in forecast_dates:
        history = inst_df.loc[:current_date - pd.Timedelta(days=1), 'label'].tail(lookback_length)
        assert len(history) == lookback_length

        seqs = torch.tensor(history.values).unsqueeze(0).float()

        pred = model.generate(seqs, max_new_tokens=prediction_length)
        predictions.append((current_date, float(pred[0, -1])))

    all_predictions.append((inst, predictions))
