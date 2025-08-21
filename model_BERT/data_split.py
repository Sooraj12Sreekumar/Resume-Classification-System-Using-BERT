import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "data/processed/"
INPUT_FILE = os.path.join(DATA_PATH,"Preprocessed_Final_Data.csv")

def split_data():
    df = pd.read_csv(INPUT_FILE)

    train_df, test_df = train_test_split(df, test_size=0.2,random_state=42,stratify=df['Category'])

    train_df, val_df = train_test_split(train_df,test_size=0.1,random_state=42,stratify=train_df['Category'])


    os.makedirs(DATA_PATH,exist_ok=True)

    train_df.to_csv(os.path.join(DATA_PATH,"train.csv"),index=False)
    test_df.to_csv(os.path.join(DATA_PATH,"test.csv"),index=False)
    val_df.to_csv(os.path.join(DATA_PATH,"val.csv"),index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}, Val: {len(val_df)}")

if __name__ == "__main__":
    split_data()