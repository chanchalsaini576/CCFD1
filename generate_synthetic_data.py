"""
Generates a synthetic dataset matching the schema of the Kaggle
"Credit Card Fraud Detection" (ULB) dataset: Time, V1-V28 (PCA-transformed
features), Amount, Class. Used only because this environment has no
internet access to download the real dataset from Kaggle.

To use the REAL dataset instead: download creditcard.csv from
https://www.kaggle.com/mlg-ulb/creditcardfraud and place it at
data/raw/creditcard.csv — the rest of the pipeline needs no changes,
since the column schema matches exactly.
"""

import pandas as pd
import numpy as np
import os



np.random.seed(42)

n_normal = 20000
n_fraud = 100  # ~0.5% fraud rate, matching the real dataset's imbalance




def make_rows(n, fraud=False):
    data = {}
    data['Time'] = np.random.uniform(0, 172792, n)
    for i in range(1, 29):
        # Fraud transactions have shifted distributions on some PCA
        # components, mimicking the separability seen in the real dataset
        if fraud and i in [1, 2, 3, 4, 10, 12, 14, 17]:
            data[f'V{i}'] = np.random.normal(loc=-3, scale=2.5, size=n)
        else:
            data[f'V{i}'] = np.random.normal(loc=0, scale=1, size=n)
    if fraud:
        data['Amount'] = np.random.exponential(scale=250, size=n)
    else:
        data['Amount'] = np.random.exponential(scale=88, size=n)
    data['Class'] = 1 if fraud else 0
    return pd.DataFrame(data)


def generate(save_path="data/raw/creditcard.csv"):
    df_normal = make_rows(n_normal, fraud=False)
    df_fraud = make_rows(n_fraud, fraud=True)

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Synthetic dataset saved to {save_path}")
    print(f"Shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")


if __name__ == "__main__":
    generate()
