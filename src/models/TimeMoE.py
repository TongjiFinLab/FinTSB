# Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts
# https://arxiv.org/abs/2409.16040
# https://github.com/Time-MoE/Time-MoE

import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from tqdm import tqdm

df = pd.read_pickle("")

df.reset_index(inplace=True)
df.rename(columns={"datetime": "date"}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by=['instrument', 'date'], inplace=True)

df = df[["date", "instrument", "label"]]

lookback_length = 96
prediction_length = 1
start_forecast_date = pd.to_datetime('2000-10-09')
end_forecast_date = pd.to_datetime('2000-12-15')

model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-200M',  # Maple728/TimeMoE-50M
    device_map="cuda",
    trust_remote_code=True,
    # low_cpu_mem_usage=True
)

all_trade_date = df["date"].unique()
forecast_dates = [date for date in all_trade_date if start_forecast_date <= date <= end_forecast_date]

df.set_index(["date", "instrument"], inplace=True)

all_predictions = []

for current_date in tqdm(forecast_dates):
    
    history = (
        df.xs(slice(None, current_date - pd.Timedelta(days=1)), level='date')
        .groupby('instrument')
        .tail(lookback_length)
    )

    instrument_order = history.index.get_level_values('instrument').unique().tolist()

    history_seqs = torch.tensor(history["label"].values.reshape(-1, lookback_length)).float().cuda()

    output = model.generate(history_seqs, max_new_tokens=prediction_length)
    pred = output[:, -prediction_length:]

    predictions_df = pd.DataFrame(pred.cpu().numpy(), index=instrument_order, columns=['prediction'])

    predictions_df["date"] = current_date
    all_predictions.append(predictions_df)

final_predictions = pd.concat(all_predictions).reset_index()
