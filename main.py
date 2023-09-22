import os
import torch
import datetime
from monai.data import DataLoader
from monai.metrics import DiceMetric
from monai.losses import DiceFocalLoss
from monai.transforms import AsDiscrete

from utils.settings import parse_opts
from monai.utils import set_determinism
from train_and_val import train, validate

from datasets.seg_datasets import PETCT_seg
from utils.check_data_validity import check_dataset

from utils.get_model import generate_model
from utils.plot import plot_two_figs, save_training_info

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=UserWarning, message="Num foregrounds 0")


def main(args):
    print(f"current setting is:\n", args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model, parameters = generate_model(args)
    model = model.cuda()

    loss_func = DiceFocalLoss(
        include_background=False, to_onehot_y=True, softmax=True
    ).cuda()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    params = [{"params": parameters, "lr": args.learning_rate}]

    optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    train_ds = PETCT_seg(phase="train", args=args, reader_keys=["suv", "ct", "seg"])
    val_ds = PETCT_seg(phase="val", args=args, reader_keys=["suv", "ct", "seg"])

    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    check_dataset(train_dataloader, f"sample_img/{args.dataset_name}_train.png")
    check_dataset(val_dataloader, f"sample_img/{args.dataset_name}_val.png")

    train_loss_list, val_loss_list, val_dice_list = [], [], []
    best_metric, best_metric_epoch = 0, 0

    post_pred = AsDiscrete(argmax=True, to_onehot=args.n_seg_classes)
    post_label = AsDiscrete(to_onehot=args.n_seg_classes)

    for epoch in range(args.n_epochs):

        print(f"\nepoch {epoch + 1}/{args.n_epochs}")
        train_loss = train(model, train_dataloader, loss_func, optimizer)
        print(f"mean train loss: {train_loss:.4f}")
        train_loss_list.append(train_loss)

        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_dice = validate(
                model,
                val_dataloader,
                loss_func,
                dice_metric,
                post_pred,
                post_label,
                args.patch_size,
            )
            val_loss_list.append(val_loss)
            val_dice_list.append(val_dice)

            print(f"mean val dice: {val_dice:.4f}, mean val loss: {val_loss:.4f}")
            plot_two_figs(args, train_loss_list, val_loss_list, val_dice_list)
            save_training_info(args, train_loss_list, val_loss_list, val_dice_list)

            if val_dice > best_metric:
                best_metric = val_dice
                best_metric_epoch = epoch + 1
                torch.save(model, os.path.join(args.save_folder, "best_model.pth"))
                print(
                    "saved new best metric model"
                    f"\nbest metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )

        scheduler.step()
        torch.save(model, f"{args.save_folder}/latest.pth")
    print(
        f"training completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}"
    )


if __name__ == "__main__":
    set_determinism(seed=1)
    # hyper parameters
    args = parse_opts()
    args.model = "unet"
    args.dataset_name = "autopet"
    args.data_root = "data/autopet"
    args.resume_path = "checkpoints/autopet/segmentation/2023-09-20/unet/best_model.pth"

    today = datetime.date.today()
    # model saving path
    args.save_folder = (
        f"./checkpoints/{args.dataset_name}/{args.task}/{today}/{args.model}"
    )
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # log saving path
    log_dst_dir = f"./logs/{args.dataset_name}/{args.task}/{today}/{args.model}"
    if not os.path.exists(log_dst_dir):
        os.makedirs(log_dst_dir)
    args.dst_folder = log_dst_dir

    main(args)
