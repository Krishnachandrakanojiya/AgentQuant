import os
import pandas as pd
import matplotlib.pyplot as plt

original_csv_path = r'C:\Users\krishna.kanojiya\Videos\OneDrive - Accenture\Pictures\AgentQuant\data\sample.csv'
cleaned_json_path = r'C:\Users\krishna.kanojiya\Videos\OneDrive - Accenture\Pictures\AgentQuant\data-cleaned.json'
output_path = r'C:\Users\krishna.kanojiya\Videos\OneDrive - Accenture\Pictures\AgentQuant\artifacts\data_visualization.png'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

df_original = pd.read_csv(original_csv_path)
df_cleaned = pd.read_json(cleaned_json_path)

if df_original.shape[1] >= 2:
    x_col = df_original.columns[0]
    y_col = df_original.columns[1]
    plt.figure(figsize=(10,6))
    plt.plot(df_original[x_col], df_original[y_col], color='blue', label='Original')
    if x_col in df_cleaned.columns and y_col in df_cleaned.columns:
        plt.plot(df_cleaned[x_col], df_cleaned[y_col], color='green', label='Cleaned')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.title('Original vs Cleaned Data')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()