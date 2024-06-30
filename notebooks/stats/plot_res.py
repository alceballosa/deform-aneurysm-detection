import numpy as np
from matplotlib import pyplot as plt
from names import get_names, get_names_dict


dict_dense = {
    "0.2": [.921,.960,.976],
    "0.3": [.921,.952,.960],
    "0.5": [.833,.849,.865],
    "0.2_ext": [.911,.921,.950],
    "0.3_ext": [.891,.901,.931], 
    "0.5_ext": [.723,.752,.762]
}

def plot_all_results(
    df_all,
    dict_metrics,
    exp_is,
    ti_versions,
    ext_versions,
    result_to_plot="Se",
    iouthr=0.5,
):
    thrs = df_all["thr"].unique()
    # result_to_plot = "Size"  # or "Size" or "Se"

    fig, ax = plt.subplots(5, len(exp_is), figsize=(4 * len(exp_is)+4, 12+4), dpi=250)
    exp_names, viz_names = get_names()
    dict_names = get_names_dict()

    

    if len(ax.shape) == 1:
        ax = ax.reshape(-1, 1)

    for i, exp_i in enumerate(exp_is):

        exp_name = exp_i

        df_exp = df_all[df_all["model"] == exp_name]
        thrs = df_exp["thr"].unique()

        thr_colors = {"0.1": "green", "0.2": "green", "0.5": "red"}
        if result_to_plot == "Se":
            col_viz = [
                f"Se@FP0.5 - IoU {iouthr}",
                f"Se@FP1 - IoU {iouthr}",
                f"Se@FP2 - Iou {iouthr}",
            ]
            col_names = ["Se@FP0.5", "Se@FP1", "Se@FP2"]
        else:
            col_names = ["small_tpr", "medium_tpr", "large_tpr"]
            col_viz = [
                f"Small TPR, IoU {iouthr}",
                f"Medium TPR, IoU {iouthr}",
                f"Large TPR, IoU {iouthr}",
            ]

        for thr in thrs:

            df_thr = df_exp[df_exp["thr"] == thr]
            if len(df_thr) == 0:
                continue
            ax[0, i].plot(
                df_thr["chptk"],
                df_thr[col_names[0]],
                label="Eval",
                color="red",
                linestyle="--",
                marker="d",
                markersize=3.5,
            )
            # horizontal line
            ax[0, i].axhline(y=dict_dense[str(thr)][0], color="purple", linestyle="--", label="Dense eval", linewidth = 0.75)

            ax[0, i].set_title(dict_names[exp_i] + " - " + col_viz[0])
            ax[1, i].plot(
                df_thr["chptk"],
                df_thr[col_names[1]],
                label="Eval",
                color="red",
                linestyle="--",
                marker="d",
                markersize=3.5,
            )
            ax[1, i].axhline(y=dict_dense[str(thr)][1], color="purple", linestyle="--", label="Dense eval", linewidth = 0.75)
            ax[1, i].set_title(dict_names[exp_i] + " - " + col_viz[1])
            ax[2, i].plot(
                df_thr["chptk"],
                df_thr[col_names[2]],
                label="Eval",
                color="red",
                linestyle="--",
                marker="d",
                markersize=3.5,
            )
            ax[2, i].set_title(dict_names[exp_i] + " - " + col_viz[2])
            ax[2, i].axhline(y=dict_dense[str(thr)][2], color="purple", linestyle="--", label="Dense eval", linewidth = 0.75)
        df_metrics = dict_metrics[exp_i]
        win_size = 50
        ax[3, i].plot(
            df_metrics["iteration"],
            df_metrics["cat_loss"].rolling(window=win_size).mean(),
            label="Clf Loss",
            color="black",
        )
        ax[3, i].plot(
            df_metrics["iteration"],
            df_metrics["iou_loss"].rolling(window=win_size).mean(),
            label="IoU Loss",
            color="blue",
        )
        ax[4, i].plot(
            df_metrics["iteration"],
            df_metrics["center_loss"].rolling(window=win_size).mean(),
            label="Localization Loss",
            color="orange",
        )
        ax[4, i].plot(
            df_metrics["iteration"],
            df_metrics["size_loss"].rolling(window=win_size).mean(),
            label="Size Loss",
            color="magenta",
        )
        ax[4, i].set_title(dict_names[exp_i] + " - Losses")
        ax[3, i].set_title(dict_names[exp_i] + " - Losses")
        ax[0, i].set_xlabel("Iterations")
        ax[0, i].set_ylabel("Sensitivity")
        ax[1, i].set_xlabel("Iterations")
        ax[1, i].set_ylabel("Sensitivity")
        ax[3, i].set_xlabel("Iterations")
        ax[3, i].set_ylabel("Loss")
        ax[4, i].set_xlabel("Iterations")
        ax[4, i].set_ylabel("Loss")
        if result_to_plot == "Se":
            y_lim_lower = 0.6
            y_lim_upper = 1.01
        else:
            y_lim_lower = 0.3
            y_lim_upper = 1.01

        minor_ticks = np.arange(0, 1.05, 0.05)
        for j in range(5):
            ax[j, i].set_xlim(20000, 70000)
            ax[j, i].set_yticks(minor_ticks, minor=True)
            ax[j, i].grid(which="both")
            ax[j, i].legend(fontsize="6")
        ax[3, i].set_ylim(0, 0.66)
        ax[4, i].set_ylim(0, 0.03)
        ax[0, i].set_ylim(y_lim_lower, y_lim_upper)
        ax[1, i].set_ylim(y_lim_lower, y_lim_upper)
        ax[2, i].set_ylim(y_lim_lower, y_lim_upper)
    if len(ext_versions) == len(exp_is):
        for i, exp_i in enumerate(ext_versions):
            exp_tr = exp_i
            df_exp = df_all[df_all["model"] == exp_tr]
            if(len(df_exp) == 0):
                continue
            thrs = df_exp["thr"].unique()
            for thr in thrs:
                df_thr = df_exp[df_exp["thr"] == thr]
                if len(df_thr) == 0:
                    continue
                ax[0, i].plot(
                    df_thr["chptk"],
                    df_thr[col_names[0]],
                    label="Xt",
                    color="blue",
                    linestyle="dashdot",
                    marker="D",
                    markersize=3.5,
                )
                for j in range(3):
                    ax[j, i].axhline(y=dict_dense[str(thr)+"_ext"][j], color="magenta", linestyle="-", label="Dense ext", linewidth = 0.75)


                # ax[0,i].set_title(viz_names[exp_i]+" - "+col_names[0])
                ax[1, i].plot(
                    df_thr["chptk"],
                    df_thr[col_names[1]],
                    label="Xt",
                    color="blue",
                    linestyle="dashdot",
                    marker="D",
                    markersize=3.5,
                )

                # ax[1,i].set_title(viz_names[exp_i]+" - " + col_names[1])
                ax[2, i].plot(
                    df_thr["chptk"],
                    df_thr[col_names[1]],
                    label="Xt",
                    color="blue",
                    linestyle="dashdot",
                    marker="D",
                    markersize=3.5,
                )
                # ax[2,i].set_title(viz_names[exp_i]+" - " + col_names[2])
            for j in range(5):

                ax[j, i].legend()

    if len(ti_versions) > 0:
        for i, exp_i in enumerate(ti_versions):
            exp_tr = exp_i
            df_exp = df_all[df_all["model"] == exp_tr]
            if(len(df_exp) == 0):
                continue
            thrs = df_exp["thr"].unique()
            for thr in thrs:
                df_thr = df_exp[df_exp["thr"] == thr]
                if len(df_thr) == 0:
                    continue
                ax[0, i].plot(
                    df_thr["chptk"],
                    df_thr[col_names[0]],
                    label="Train",
                    color="black",
                    marker="o",
                    markersize=3.5,
                )

                # ax[0,i].set_title(viz_names[exp_i]+" - "+col_names[0])
                ax[1, i].plot(
                    df_thr["chptk"],
                    df_thr[col_names[1]],
                    label="Train",
                    color="black",
                    marker="o",
                    markersize=3.5,
                )
                # ax[1,i].set_title(viz_names[exp_i]+" - " + col_names[1])
                ax[2, i].plot(
                    df_thr["chptk"],
                    df_thr[col_names[1]],
                    label="Train",
                    color="black",
                    marker="o",
                    markersize=3.5,
                )
                # ax[2,i].set_title(viz_names[exp_i]+" - " + col_names[2])
            for j in range(5):

                ax[j, i].legend()
            # ax[2,i].plot(x_ticks,
            #        df_thr[col_names[2]],
            #        color = thr_colors[str(thr)], linestyle = "--" )
            # ax[2,i].set_title(viz_names[exp_i]+" - " + col_names[2])
    # fig.suptitle("Recurrent Deformable Transformer Models - Sensitivity vs. Iterations @ different number of rec. steps\n")

    plt.legend()
    plt.tight_layout()
