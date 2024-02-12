"""Generate report based on evaluated csv files"""
import pandas as pd
from sklearn.metrics import accuracy_score


def filter_df(df, col, val):
    """Filter dataframe by column and value"""
    return df[df[col] == val]


def gender_acc(df):
    """Return the accuracy score of the gender vs gender predictions"""
    return accuracy_score(df['gender'], df['gender_preds'])


def acc_by_col(df, col, writer):
    """Generate predictions accuracy by column"""
    for unique in df[col].unique():
        col_df = df[df[col] == unique]
        col_acc = gender_acc(col_df)
        writer.write(f"{unique} predictions accuracy: {round(col_acc, 2)} \n")


def gen_report(df, pred_name, out):
    """Wrapper report class"""
    fp = open(out, mode="a", encoding="utf-8")

    fp.write(f"\n# {pred_name} info:\n")
    fp.write("-"*20)

    fp.write("\n## General Accuracy\n")
    gen_acc = gender_acc(df)
    gen_miss = df[df['gender'] != df['gender_preds']]
    fp.write(f"Prediction error count: {len(gen_miss)} \n")
    fp.write(f"Prediction accuracy score: {round(gen_acc, 2)} \n")

    fp.write("\n## Accuracy by gender \n")
    male_df = filter_df(df, 'gender', 'Male')
    male_acc = gender_acc(male_df)
    female_df = filter_df(df, 'gender', 'Female')
    female_acc = gender_acc(female_df)
    fp.write(f"Male: {round(male_acc, 2)} \n")
    fp.write(f"Female: {round(female_acc, 2)} \n")

    fp.write("\n## Accuracy by race \n")
    acc_by_col(df, 'race', fp)

    fp.write("\n## Accuracy by age \n")
    acc_by_col(df, 'age', fp)
    fp.write("-"*20)
    fp.close()


if __name__ == "__main__":
    ROOT = "/home/lazye/Documents/ufrgs/mcs/clip/clip-bias-explore/\
fair-face-classification"
    RESULTS_PATH = ROOT + "/data/results"
    REPORT_PATH = ROOT + "/data/report"
    arg_sum_df = pd.read_csv(RESULTS_PATH+"/arg_sum_synms.csv")
    arg_top_df = pd.read_csv(RESULTS_PATH+"/arg_top_synms.csv")
    gen_report(arg_sum_df, "age_race_gender Average Sum Synonym Prediction",
               REPORT_PATH+"/arg_report.txt")
    gen_report(arg_top_df, "age_race_gender Top 1 Synonym Prediction",
               REPORT_PATH+"/arg_report.txt")
