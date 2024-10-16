import numpy as np
import torch
import torch.nn
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# import wandb

import segmentation_models_pytorch as smp

from Datasets import MAE_Dataset, MUnet_Dataset
from Models import AEUnet
from Support import multi_transform, extract_focus
from support import weighted_dice_loss, iou, segment_instances, agregate_results, set_seed


class MAETrainer:
    def __init__(self, encoder_paths, unet_train_path, unet_test_dict, config, wandb_run=None, lr=1e-4,
                 encoder_mask_ratio=0.5, unet_mask_ratio=0, patch_size=8, use_pretrained=True,
                 resnet_version="resnet18"
                 ):

        set_seed(42)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.lr = lr
        weights = "imagenet" if use_pretrained else None
        self.model = AEUnet(use_pretrained=use_pretrained, resnet_version=resnet_version).to(self.device)

        # MAE dataset
        dataset = MAE_Dataset(encoder_paths, num_channels=3, mask_ratio=encoder_mask_ratio, patch_size=patch_size)
        self.data_loader = DataLoader(dataset, batch_size=3, shuffle=True)

        # UNET datasets
        unet_trainset = MUnet_Dataset(unet_train_path, num_channels=3, mask_ratio=unet_mask_ratio, patch_size=patch_size)
        self.unet_train_loader = DataLoader(unet_trainset, batch_size=3, shuffle=True)

        # UNET test datasets
        self.unet_testloaders = {}
        for species in unet_test_dict.keys():
            dict = {}


            data_path = unet_test_dict[species]['path']
            print(f"Loading test dataset from {species}, path: {data_path}")
            dataset = MUnet_Dataset(data_path, num_channels=3, mask_ratio=unet_mask_ratio, patch_size=patch_size)

            dict["loader"] = DataLoader(dataset, batch_size=1, shuffle=False)
            dict["seen"] = unet_test_dict[species]["seen"]

            self.unet_testloaders[species] = dict

        self.patience = 10
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.log_interval = 100
        if wandb_run is None:
            self.run = wandb.init(project="ResUNET", config=config)
        else:
            self.run = wandb_run
            self.run.config.update(config, allow_val_change=False)
        wandb.watch(self.model, log="all", log_freq=self.log_interval)  # Log model gradients and parameters
        self.total_epochs = 0

    def train_encoder(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        criterion = nn.MSELoss()

        best_interval_loss = float('inf')
        best_interval = 0
        interval_loss = 0
        interval_frames = 0

        frames_done = 0
        current_interval = 0
        resume = True

        while resume:

            for imgs, label in tqdm(self.data_loader, desc="Encoder batch"):
                frames_done += imgs.shape[0]
                interval_frames += imgs.shape[0]

                imgs = imgs.to(self.device)
                label = label.to(self.device)

                preds = self.model.reconstruct(imgs)
                preds = torch.nn.Sigmoid()(preds)


                loss = criterion(preds, label)

                interval_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



                if interval_frames >= self.log_interval:
                    imgs = imgs.cpu().detach()
                    label = label.cpu().detach()
                    preds = preds.cpu().detach()

                    imgs = imgs[0:1,0,:,:]
                    label = label[0:1,0,:,:]
                    preds = preds[0:1,0,:,:]

                    height, width = imgs.shape[1], imgs.shape[2]
                    border_column = torch.ones(1, height, 10)

                    # concatenate the images
                    output_image = torch.cat([imgs, border_column, preds, border_column, label], dim=2)

                    avg_interval_loss = interval_loss / interval_frames

                    # Log images
                    self.run.log({
                        "output_image": [wandb.Image(output_image)],
                        "encoder_loss": avg_interval_loss
                    }, step=current_interval+self.total_epochs + 1, commit=True)

                    if avg_interval_loss < best_interval_loss:
                        best_interval_loss = avg_interval_loss
                        best_interval = current_interval
                        self._save_and_log_best_model()
                    elif best_interval + self.patience < current_interval:
                        resume = False
                        break

                    interval_loss = 0
                    interval_frames = 0
                    current_interval += 1

        self.total_epochs = current_interval + 1

    def train_unet(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = weighted_dice_loss

        best_epoch = 0
        best_iou = 0

        best_epoch_results = {}

        current_epoch = 0
        while True:
            if best_epoch + self.patience < current_epoch:
                break


            epoch_loss = 0
            epoch_iou = 0

            for imgs, labels, weights in tqdm(self.unet_train_loader, desc="Unet batch"):

                # print(f"weights min: {weights.min()}, max: {weights.max()}")

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                weights = weights.to(self.device)

                preds = self.model.segment(imgs)
                preds = torch.nn.Sigmoid()(preds)

                loss = criterion(preds, labels, weights)
                epoch_loss += loss.item()

                metric_pred = preds.cpu().detach()
                label_cpu = labels.cpu().detach()
                for i in range(len(metric_pred)):
                    epoch_iou += iou(metric_pred[i], label_cpu[i])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            avg_epoch_loss = epoch_loss / len(self.unet_train_loader)
            avg_iou = epoch_iou / len(self.unet_train_loader.dataset)

            logs = {}
            if current_epoch % 20 == 0 : # or avg_iou < best_iou
                logs = self._evaluate_unet()
                # save the model for this epoch
                torch.save(self.model.state_dict(), f"saves/epoch_{current_epoch}.pt")
            logs["train_loss"] = avg_epoch_loss
            logs["train_iou"] = avg_iou


            # Log images
            self.run.log(logs, step=current_epoch+self.total_epochs + 1, commit=True)

            if epoch_iou > best_iou:
                best_iou = epoch_iou
                best_epoch = current_epoch

                best_epoch_results = logs

                self.run.summary["best_iou"] = best_iou
                self._save_and_log_best_model()


            current_epoch += 1

        self.run.log(best_epoch_results, step=current_epoch+self.total_epochs + 1, commit=True)
        current_epoch += 1

        self.total_epochs += current_epoch

    def _evaluate_unet(self):
        iou_thresholds = [0.0] + [0.5 + i * 0.05 for i in range(10)]
        # iou_thresholds = [0.0]
        print(f"evaluating unet")
        seen_results = []
        unseen_results = []

        unseen_loss = 0
        seen_loss = 0
        unseen_samples = 0
        seen_samples = 0

        # unet_test_loader = self.unet_testloaders["SET3A"]
        for loader in self.unet_testloaders.keys():
            item = self.unet_testloaders[loader]
            unet_test_loader = item["loader"]
            result, loss, samples = self.test_dataloader(unet_test_loader, iou_thresholds=iou_thresholds)

            if item["seen"]:
                seen_results.extend(result)
                seen_loss += loss
                seen_samples += samples
            else:
                unseen_results.extend(result)
                unseen_loss += loss
                unseen_samples += samples


        avg_total_loss = (unseen_loss + seen_loss) / (unseen_samples + seen_samples)
        avg_unseen_loss = unseen_loss / unseen_samples
        avg_seen_loss = seen_loss / seen_samples

        result = agregate_results(seen_results, "seen", iou_thresholds=iou_thresholds)
        result.update(agregate_results(unseen_results, "unseen", iou_thresholds=iou_thresholds))
        result["test_total_loss"] = avg_total_loss
        result["test_unseen_loss"] = avg_unseen_loss
        result["test_seen_loss"] = avg_seen_loss
        return result

    def test_dataloader(self, unet_test_loader, iou_thresholds=[0.5]):
        device = self.device

        results = []
        total_loss = 0

        for i, (img, mask, weights) in tqdm(enumerate(unet_test_loader)):
            img = img.to(device)
            mask = mask.to(device)
            weights = weights.to(device)

            with torch.no_grad():
                # print(f"sample {i} out of {len(unet_test_loader)}")
                # preds = self.make_transformed_combined_mask(img)
                preds = multi_transform(img, self.model, device=device, threshold=0.5)
                preds = extract_focus(preds, min_area=50, max_detections=4)

                # unsqueeze(0) the preds, but in np


                # print(f"preds after focus: {preds.shape}, mask shape: {mask.shape}")

                loss = weighted_dice_loss(preds, mask, weights)
                total_loss += loss.item()

                preds = preds.cpu().detach().numpy() if isinstance(preds, torch.Tensor) else preds
                mask = mask.cpu().detach().numpy() if isinstance(mask, torch.Tensor) else mask

                # print(f"preds shape: {preds.shape}, mask shape: {mask.shape}")
                for i in range(len(preds)):
                    result = segment_instances(preds[i][0], mask[i][0],iou_thresholds=iou_thresholds)
                    results.append(result)

        return results, total_loss, len(unet_test_loader)




    def _save_and_log_best_model(self):
        torch.save(self.model.state_dict(), f"saves/best_model.pt")
        self.run.save("saves/best_model.pt")