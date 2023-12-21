"""
compare_models.py

This script reads in the results files for both systems A and B. It generates confusion matrices
for both systems and radar plots for comparison.

usage: python compare_models.py <model A basename> <model B basename>
"""

import sys, getopt
import os.path
import json
import pandas as pd
import seaborn as sns
import evaluate
import matplotlib.pyplot as plt
from evaluate.visualization import radar_plot


def do_confusion_matrices(modelAcsv, modelBcsv):

    # Read the saved confusion matrix for system A from the CSV file
    df_conf_matrix = pd.read_csv(modelAcsv, index_col=0)

    # Display the loaded confusion matrix as a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 6}, linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('System A Confusion Matrix')
    plt.savefig("system_a_confusion_matrix.png")
    plt.show(block=False)

    # Read the saved confusion matrix for system B from the CSV file
    df_conf_matrix = pd.read_csv(modelBcsv, index_col=0)

    # Display the loaded confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 8}, linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('System B Confusion Matrix')
    plt.savefig("system_b_confusion_matrix.png")
    plt.show(block=False)


def do_radar_plots(modelAjson, modelBjson):

    ma_file = open(modelAjson)
    ma_dict = json.load(ma_file)

    ma_overall = {}
    ma_tags = []
    ma_tags_stats = []
    for key, value in ma_dict.items():
        if key in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']:
            ma_overall[key] = value
        else:
            ma_tags.append(key)
            ma_tags_stats.append(value)

    mb_file = open(modelBjson)
    mb_dict = json.load(mb_file)

    mb_overall = {}
    mb_tags = []
    mb_tags_stats = []
    for key, value in mb_dict.items():
        if key in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']:
            mb_overall[key] = value
        else:
            mb_tags.append(key)
            mb_tags_stats.append(value)

    ABdata = [ma_overall, mb_overall]
    model_names = ["Model A", "Model B"]

    fig1 = plt.figure(1, figsize=(9,5))
    plot = radar_plot(data=ABdata, model_names=model_names,
                      config={"theta_tick_lbls": {"fontsize": 10},
                              "rgrid_tick_lbls_args": {"fontsize": 10},
                              "theta_tick_lbls_brk_lng_wrds": False,
                              "outer_ring": {"visible": False},
                              "theta_tick_lbls_pad": 4, 
                              "bbox_to_anchor": (1.42,1),
                              "incl_endpoint": True}, fig=fig1 )

    plt.title("Overall Comparison Between Systems A and B", y=1.08)
    plt.savefig("system_a_and_b.png")
    plot.show()

    fig2 = plt.figure(2, figsize=(9,5))
    plot = radar_plot(data=ma_tags_stats, model_names=ma_tags,
                      config={"theta_tick_lbls": {"fontsize": 10},
                              "rgrid_tick_lbls_args": {"fontsize": 10},
                              "outer_ring": {"visible": False},
                              "theta_tick_lbls_pad": 4, 
                              "bbox_to_anchor": (1.42,1)}, fig=fig2)
    plt.title("System A Tags", y=1.08)
    plt.savefig("system_a_tags.png")
    plot.show()

    fig3 = plt.figure(3, figsize=(9,5))
    plot = radar_plot(data=mb_tags_stats, model_names=mb_tags,
                      config={"theta_tick_lbls": {"fontsize": 10},
                              "rgrid_tick_lbls_args": {"fontsize": 10},
                              "outer_ring": {"visible": False},
                              "theta_tick_lbls_pad": 4, 
                              "bbox_to_anchor": (1.42,1)}, fig=fig3)
    plt.title("System B Tags", y=1.08)
    plt.savefig("system_b_tags.png")
    plot.show()



def main(argv):

    help_message = "usage: python compare_models.py <model A basename> <model B basename>"

    try:
        opts, args = getopt.getopt(argv, "h")

        for opt, _ in opts:
            if opt == '-h':
                print(help_message)
                sys.exit(0)

    except getopt.GetoptError:
        sys.exit(help_message)

    if len(args) != 2:
        sys.exit(help_message)

    modelAbase, modelBbase = args
    modelAjson = f"{modelAbase}.json"
    global modelAcsv
    modelAcsv = f"{modelAbase}.csv"
    modelBjson = f"{modelBbase}.json"
    global modelBcsv
    modelBcsv = f"{modelBbase}.csv"

    for result in (modelAjson, modelAcsv, modelBjson, modelBcsv):
        if not os.path.exists(result):
            sys.exit(f"Files (.json/.csv) with {result} basename do not exist")

    do_radar_plots(modelAjson, modelBjson)
    do_confusion_matrices(modelAcsv, modelBcsv)
    input("Press Enter key to exit...")


if __name__ == "__main__":
    main(sys.argv[1:])

