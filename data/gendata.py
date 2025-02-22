import pandas as pd
import glob

patterns = ["fluctuation", "rise", "fall", "extreme"]
merged_data = []

for pattern in patterns:
    all_files = glob.glob(f"./FinTSB_cn/{pattern}/dataset_*.pkl")
    for source_id, file_path in enumerate(sorted(all_files)):
        df = pd.read_pickle(file_path)
    
        df = df.reset_index()
        df['source'] = source_id
    
        df = df.set_index(['datetime', 'instrument'])
        # print(df)
        # break
        merged_data.append(df)

print(len(merged_data))
final_df = pd.concat(merged_data)
print(final_df.shape)
final_df.to_pickle("./FinTSB_cn/merged_dataset.pkl")