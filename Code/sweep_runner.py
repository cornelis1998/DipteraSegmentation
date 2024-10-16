# This is a sample Python script.
import torch
import wandb

from Trainer import MAETrainer


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run = wandb.init()
    hyperparameters = run.config

    # sweep parameters
    keys = hyperparameters.keys()
    use_imagenet = hyperparameters["use_imagenet"] if "use_imagenet" in keys else True
    train_encoder = hyperparameters["train_encoder"] if "train_encoder" in keys else True
    mask_fraction = hyperparameters["mask_fraction"] if "mask_fraction" in keys else 0
    resnet_version = hyperparameters["resnet_version"] if "resnet_version" in keys else "resnet18"
    lr = hyperparameters["lr"] if "lr" in keys else 4
    lr = 10**(-lr)

    # other parameters
    data_paths = ["/mnt/tb/Insects/cropped"]
    set = "SET6"

    unet_train_path = f"/mnt/tb/Insects/Sets/{set}/Train"
    encoder_mask_ratio = mask_fraction
    unet_mask_ratio = 0
    print(f"Using imagenet: {use_imagenet}, train encoder: {train_encoder}, lr: {lr}")

    test_dict = {
        "Test_seen": {
            "path": f"/mnt/tb/Insects/Sets/{set}/Test_seen",
            "mask_ratio": unet_mask_ratio,
            "seen": True
        },
        "Test_unseen": {
            "path": f"/mnt/tb/Insects/Sets/{set}/Test_unseen",
            "mask_ratio": unet_mask_ratio,
            "seen": False
        }

    }

    config = {
        "use_imgnet": use_imagenet,
        "pre_train_encoder": train_encoder,
        "lr": lr,
        "encoder_mask_ratio": encoder_mask_ratio,
        "unet_mask_ratio": unet_mask_ratio,
        "Dataset": set,
        "resnet_version": resnet_version,
    }

    trainer = MAETrainer(
        encoder_paths=data_paths,
        unet_train_path=unet_train_path,
        unet_test_dict=test_dict,
        config=config,
        use_pretrained=use_imagenet,
        encoder_mask_ratio=encoder_mask_ratio,
        unet_mask_ratio=unet_mask_ratio,
        lr=lr,
        resnet_version=resnet_version,
        wandb_run=run)

    if train_encoder:
        trainer.train_encoder()

    trainer.train_unet()
    trainer.run.finish()
