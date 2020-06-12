import pandas as pd
import pandas_profiling
from sklearn.model_selection import train_test_split
import numpy as np


def encoder():
    df = pd.read_csv("../input/livedoor.tsv", sep="\t", encoding="utf-8")

    categories = df.columns[2:]
    df["textID"] = np.arange(len(df))
    df["category"] = 0
    for column in categories:
        df.loc[df[column] == 1, "category"] = column

    processed_df = df.loc[:, ["textID", "article", "category"]]

    print(processed_df.head(20))

    processed_df.to_csv(
        "../input/livedoor_traintest.csv", encoding="utf-8", index=False
    )


def report(file_name, report_name):
    df = pd.read_csv(file_name, encoding="utf-8")
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(report_name)


def main():
    df = pd.read_csv("../input/livedoor_traintest.csv", encoding="utf-8")
    X_train, X_test, y_train, y_test = train_test_split(
        df.loc[:, ["textID", "article"]],
        df.category.values,
        test_size=0.2,
        random_state=42,
        stratify=df.category.values,
    )

    train_df = pd.DataFrame(
        {
            "textID": X_train.textID.values,
            "article": X_train.article.values,
            "category": y_train,
        }
    )

    test_df = pd.DataFrame(
        {
            "textID": X_test.textID.values,
            "article": X_test.article.values,
            "category": y_test,
        }
    )

    train_df.to_csv("../input/train.csv", encoding="utf-8", index=False)
    test_df.to_csv("../input/test.csv", encoding="utf-8", index=False)


if __name__ == "__main__":
    main()
    # report("../input/train.csv", "train_report.html")
    # report("../input/test.csv", "test_report.html")
