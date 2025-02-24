# Chronos: Pretrained Models for Probabilistic Time Series Forecasting
# https://arxiv.org/abs/2403.07815
# https://github.com/amazon-science/chronos-forecasting

import torch
import pandas as pd
from chronos import BaseChronosPipeline
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

"""
| Model               | Parameters |
|---------------------|------------|
| chronos-t5-tiny     | 8M         |
| chronos-t5-mini     | 20M        |
| chronos-t5-small    | 46M        |
| chronos-t5-base     | 200M       |
| chronos-t5-large    | 710M       |
| chronos-bolt-tiny   | 9M         |
| chronos-bolt-mini   | 21M        |
| chronos-bolt-small  | 48M        |
| chronos-bolt-base   | 205M       |
"""
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-tiny",
    device_map="cuda",  # use "cpu" for CPU inference
    torch_dtype=torch.float32,
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

    history_seqs = torch.tensor(history["label"].values.reshape(-1, lookback_length)).float()

    quantiles, mean = pipeline.predict_quantiles(
        context=history_seqs,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    
    predictions_df = pd.DataFrame(mean.cpu().numpy(), index=instrument_order, columns=['prediction'])

    predictions_df["date"] = current_date
    all_predictions.append(predictions_df)

final_predictions = pd.concat(all_predictions).reset_index()
