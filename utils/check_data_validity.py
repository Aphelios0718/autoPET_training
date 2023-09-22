import matplotlib
import numpy as np
from monai.utils import first
import matplotlib.pyplot as plt
matplotlib.use("Agg")


def check_dataset(dataloader, fname):
    check_data = first(dataloader)
    petct, label = check_data["petct"][0], check_data["seg"][0][0]
    z_sample = label.shape[0] // 2
    print(f"image shape: {petct.shape}, label shape: {label.shape}")
    plt.figure("check data", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("PET/SUV")
    plt.imshow(np.flipud(petct[0, z_sample, :, :]), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("CT")
    plt.imshow(np.flipud(petct[1, z_sample, :, :]), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("label")
    plt.imshow(np.flipud((label[z_sample, :, :])))
    plt.savefig(fname)


def check_petct_dataset_twostep(dataloader, fname):
    """

    @param dataloader:
    @param fname:
    """
    check_data = first(dataloader)
    suv_05, suv, ct, label = (
        check_data["suv_05"][0],
        check_data["suv"][0],
        check_data["ct"][0],
        check_data["seg"][0][0],
    )

    plt.figure("check dataset")

    plt.subplot(2, 2, 1)
    plt.title("suv_05")
    plt.imshow(suv_05[0, 32, :, :], cmap="gray")
    plt.subplot(2, 2, 2)

    plt.title("suv")
    plt.imshow(suv[0, 32, :, :], cmap="gray")
    plt.subplot(2, 2, 3)

    plt.title("ct")
    plt.imshow(ct[0, 32, :, :])
    plt.subplot(2, 2, 4)

    plt.title("label")
    plt.imshow(label[32, :, :])
    plt.savefig(fname)
