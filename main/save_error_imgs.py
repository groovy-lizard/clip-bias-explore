import pandas as pd
import matplotlib.pyplot as plt


def save_img(path, title, label, out):
    img = plt.imread(path)
    plt.imshow(img)
    plt.title(title)
    plt.xlabel(label)
    plt.savefig(out, bbox_inches='tight')


def batch_save(df, df_name, path, outdir):
    for i in range(len(df)):
        df_row = df.iloc[i]
        fname = df_row['file']
        fpath = f"{path}/{fname}"
        title = df_row['gender']
        xlabel = df_row['synm']
        im_name = f"{df_name}_{i}"
        outpath = f"{outdir}/{im_name}.png"
        print(f"saving {im_name} to {outdir}")
        save_img(fpath, title, xlabel, outpath)


if __name__ == "__main__":

    ROOT = "/home/lazye/Documents/ufrgs/mcs/clip/clip-bias-explore/\
fair-face-classification"
    RESULTS_PATH = ROOT + "/data/results"
    IMG_PATH = "/home/lazye/Documents/ufrgs/mcs/datasets/FairFace/"

    # top synm preds df
    topk_df = pd.read_csv(RESULTS_PATH+"/arg_top_synms.csv")
