import pandas as pd
import numpy as np
import math

def top_k_strategy(df, top_k=30, commission_rate=0.0):
    df.sort_values(by=['datetime', 'score'], ascending=[True, False], inplace=True)

    daily_returns = []
    dates = []
    for date, group in df.groupby('datetime'):
        top_k_stocks = group.head(top_k)
        daily_return = top_k_stocks['label'].mean() * (1 - commission_rate)**2
        daily_returns.append(daily_return)
        dates.append(date)

    result_df = pd.DataFrame({
        'dt': dates,
        'daily_return': daily_returns
    })

    return result_df

def top_k_drop_strategy(df, top_k=30, n=5, commission_rate=0.0):
    df.sort_values(by=['datetime', 'score'], ascending=[True, False], inplace=True)
    df = df.reset_index()
    
    daily_returns = []
    dates = []
    prev_instruments = None
    
    for date, group in df.groupby('datetime'):
        all_instruments = group['instrument'].unique()
        
        if prev_instruments is None:
            current_instruments = group.head(top_k)['instrument'].tolist()
            avg_return = group.head(top_k)['label'].mean()
            daily_return = avg_return * (1 - commission_rate)**2
        else:
            retained_candidates = list(set(prev_instruments) & set(all_instruments))
            
            retained_pool = group[group['instrument'].isin(retained_candidates)]
            retained_pool = retained_pool.sort_values(by='score', ascending=False)
            
            retained = retained_pool.head(top_k - n)['instrument'].tolist()
            
            remaining_pool = group[~group['instrument'].isin(retained)]
            new_selected = remaining_pool.head(n)['instrument'].tolist()
            
            current_instruments = retained + new_selected
            
            retained_return = retained_pool.head(top_k - n)['label'].mean()
            new_return = remaining_pool.head(n)['label'].mean()

            daily_return = (retained_return * (top_k - n) + 
                           new_return * n * (1 - commission_rate)**2) / top_k
        
        daily_returns.append(daily_return)
        dates.append(date)
        prev_instruments = current_instruments
    
    result_df = pd.DataFrame({
        'dt': dates,
        'daily_return': daily_returns
    })
    return result_df

def calculate_portfolio_metrics(df):

    daily_cumulative_return = (1 + df['daily_return']).prod()
    # ARR
    annualized_return = daily_cumulative_return ** (252 / len(df)) - 1
    # AVol
    annualized_volatility = df['daily_return'].std() * np.sqrt(252)
    
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() 
    # MDD
    running_max = df['cumulative_return'].cummax()
    drawdown = (df['cumulative_return'] - running_max) / running_max
    max_drawdown = drawdown.min()

    # ASR
    risk_free_rate = 0.00
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # IR
    benchmark_return = risk_free_rate / 252 
    excess_return = df['daily_return'] - benchmark_return
    information_ratio = excess_return.mean() / excess_return.std() * math.sqrt(20)

    portfolio_metrics = {
        'ARR': annualized_return,
        'AVol': annualized_volatility,
        'MDD': max_drawdown,
        'ASR': sharpe_ratio,
        'IR': information_ratio
    }

    return portfolio_metrics, df