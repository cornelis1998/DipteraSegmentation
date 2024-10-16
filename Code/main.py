# This is a sample Python script.

from TrainAlterations.Train_UNET_MAE import SegmentationPreTrainer
from Trainer import MAETrainer


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # sweep parameters
    use_imagenet = True
    train_encoder = False


    # other parameters


    print_hi('PyCharm')
    data_paths = ["/mnt/tb/Insects/cropped"]
    set = "SET8_distance"

    unet_train_path = f"/mnt/tb/Insects/Sets/{set}/Train"
    encoder_mask_ratio = 0.6
    unet_mask_ratio = 0.0
    lr = 1e-4

    test_dict ={
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
        "Dataset": set,
    }

    trainer = MAETrainer(
        encoder_paths=data_paths,
        unet_train_path=unet_train_path,
        unet_test_dict=test_dict,
        config=config,
        use_pretrained=use_imagenet,
        encoder_mask_ratio=encoder_mask_ratio,
        unet_mask_ratio=unet_mask_ratio,
        lr=lr)

    if train_encoder:
        trainer.train_encoder()

    trainer.train_unet()
    trainer.run.finish()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
