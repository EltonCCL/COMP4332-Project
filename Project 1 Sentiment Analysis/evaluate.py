import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

if __name__ == "__main__":
    ans = pd.read_csv("data/valid.csv", usecols=["id", "label"])
    pred = pd.read_csv("data/valid_pred.csv", usecols=["id", "label"])
    df = pd.merge(ans, pred, how="left", on=["id"])
    df.fillna(0, inplace=True)
    acc = accuracy_score(df["label_x"], df["label_y"])
    p, r, f1, _ = precision_recall_fscore_support(df["label_x"], df["label_y"], average="macro")
    print("accuracy:", acc, "\tprecision:", p, "\trecall:", r, "\tf1:", f1)
