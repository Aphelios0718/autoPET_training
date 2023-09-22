import os
import sys
import time


def get_log_path(log_dir):
    return log_dir + "log-" + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".log"


class Logger(object):
    def __init__(self, log_dir, stream=sys.stdout):
        self.terminal = stream
        self.log = open(str(get_log_path(log_dir)), "a")
        os.makedirs(str(self.log)) if not os.path.exists(str(self.log)) else ...

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def plot_two_figs(sets, train_loss_list, val_loss_list, val_dice_list):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex="none")
    axes[0].plot(
        [i + 1 for i in range(len(train_loss_list))], train_loss_list, "blue", lw=0.5
    )
    axes[0].set_title("train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_facecolor("lightgrey")
    axes[0].set_alpha(0.2)
    axes[0].grid(color="white", linewidth=0.5, alpha=0.3)

    axes[1].plot(
        [i + 1 for i in range(len(val_loss_list))], val_loss_list, "blue", lw=0.5
    )
    axes[1].plot(
        [i + 1 for i in range(len(val_dice_list))],
        val_dice_list,
        "green",
        lw=0.5,
        linestyle="--",
    )
    axes[1].set_title("val loss & dice")
    axes[1].set_xlabel("val interval")
    axes[1].set_facecolor("lightgrey")
    axes[1].set_alpha(0.2)
    axes[1].grid(color="white", linewidth=0.5, alpha=0.3)

    plt.autoscale(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(sets.n_epochs // 10))

    # fig saving path
    fig_path = f"{sets.dst_folder}/progress.png"
    plt.savefig(fig_path)
    plt.close()


def save_training_info(sets, train_loss_list, val_loss_list, val_dice_list):
    import pandas as pd

    # dst_pkl_name = f"{sets.dst_folder}/training_info.pickle"
    dst_excel_name = f"{sets.dst_folder}/training_info.xlsx"

    epoch_info = {"train loss": train_loss_list}
    val_interval_info = {"val loss": val_loss_list, "val dice": val_dice_list}
    df1 = pd.DataFrame(epoch_info)
    df2 = pd.DataFrame(val_interval_info)

    with pd.ExcelWriter(dst_excel_name) as writer:
        df1.to_excel(writer, sheet_name="train")
        df2.to_excel(writer, sheet_name="validation")
