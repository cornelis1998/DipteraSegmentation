import random

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff
import wandb
from scipy.ndimage import label


def calc_haufdorff(mask1, mask2):
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu()
        mask1 = mask1[0].detach()
        mask1 = mask1.numpy().astype(np.uint8)


    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu()
        mask2 = mask2[0].detach()
        mask2 = mask2.numpy().astype(np.uint8)


    mask1 = mask1 > 0.5
    mask2 = mask2 > 0.5


    mask1_edge = mask1 ^ binary_erosion(mask1)
    mask2_edge = mask2 ^ binary_erosion(mask2)

    mask1_edge_points = np.argwhere(mask1_edge == 1)
    mask2_edge_points = np.argwhere(mask2_edge == 1)

    hausdorff = directed_hausdorff(mask1_edge_points, mask2_edge_points)
    hausdorff2 = directed_hausdorff(mask2_edge_points, mask1_edge_points)
    return max(hausdorff[0], hausdorff2[0])

def dice_loss(preds, masks):
    smooth = 1e-6
    preds_flat = preds.view(preds.size(0), -1)
    masks_flat = masks.view(masks.size(0), -1)
    intersection = (preds_flat * masks_flat).sum(dim=1)
    dice = (2. * intersection + smooth) / (preds_flat.sum(dim=1) + masks_flat.sum(dim=1) + smooth)
    return 1 - dice.mean()

def weighted_dice_loss(preds, labels, weights, device = torch.device("cuda")):
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, device=device)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, device=device)
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, device=device)

    preds = preds.to(device)
    labels = labels.to(device)
    weights = weights.to(device)

    loss = dice_loss(preds, labels)
    return (loss * weights).mean()

def iou(preds, labels, smooth=1e-6):
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# def f1_score(preds, labels):
#     preds = (preds > 0.5).float()
#     labels = (labels > 0.5).float()
#
#     tp = (preds * labels).sum().to(torch.float32)
#     tn = ((1 - preds) * (1 - labels)).sum().to(torch.float32)
#     fp = (preds * (1 - labels)).sum().to(torch.float32)
#     fn = ((1 - preds) * labels).sum().to(torch.float32)
#
#     precision = tp / (tp + fp + 1e-8)
#     recall = tp / (tp + fn + 1e-8)
#
#     f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
#     return f1
#

def segment_instances(pred_mask, true_mask, iou_thresholds=[0.5]):
    results = {}

    pred_instances, pred_labels = label(pred_mask)
    true_instances, true_labels = label(true_mask)

    # get all IoUs
    results["ious"] = calc_all_ious(pred_instances, true_instances, pred_labels, true_labels)
    results["thresholded_values"] = {}

    # calc all thresholded values
    for threshold in iou_thresholds:
        results["thresholded_values"][threshold] = calc_thresholded_values(pred_instances, true_instances, pred_labels, true_labels, threshold)

    return results



def calc_all_ious(pred_instances, true_instances, pred_labels, true_labels):
    ious = []
    for pred_label in range(1, pred_labels + 1):
        pred_mask = pred_instances == pred_label
        best_iou = 0
        for true_label in range(1, true_labels + 1):
            true_mask = true_instances == true_label
            intersection = np.logical_and(pred_mask, true_mask)
            union = np.logical_or(pred_mask, true_mask)
            iou = np.sum(intersection) / np.sum(union)

            if iou > best_iou:
                best_iou = iou

        ious.append(best_iou)
    return ious

def calc_thresholded_values(pred_instances, true_instances, pred_labels, true_labels, iou_threshold):
    ious = []
    tp = 0
    fp = 0
    fn = 0
    hausdorfs = []
    matched_true_instances = set()

    # for each instance in the predicted mask
    # find the instance in the true mask with the highest overlap
    # calculate the IoU
    for i in range(1, pred_labels + 1):
        pred_instance = pred_instances == i
        max_iou = 0
        best_instance = 0

        for j in range(1, true_labels + 1):
            true_instance = true_instances == j
            intersection = np.logical_and(pred_instance, true_instance)
            union = np.logical_or(pred_instance, true_instance)
            iou = np.sum(intersection) / np.sum(union)

            if iou > max_iou:
                max_iou = iou
                best_instance = j

        ious.append(max_iou)
        if max_iou >= iou_threshold:
            tp += 1
            matched_true_instances.add(best_instance)
            # Calculate Hausdorff distance
            overlap_mask = true_instances == best_instance
            dist = calc_haufdorff(pred_instance, overlap_mask)
            hausdorfs.append(dist)
        else:
            fp += 1

    # Calculate FN for ground truth instances that were not matched
    for j in range(1, true_labels + 1):
        if j not in matched_true_instances:
            fn += 1

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "hausdorfs": hausdorfs
    }


def agregate_results(results, seen="Seen", iou_thresholds=[0.5]):
    avg_iou = np.mean([iou for result in results for iou in result["ious"]])
    table_data = []

    base_mse = 0
    base_rmse = 0

    for threshold in iou_thresholds:
        thresholded_values = [result["thresholded_values"][threshold] for result in results]
        tp = sum([result["tp"] for result in thresholded_values])
        fp = sum([result["fp"] for result in thresholded_values])
        fn = sum([result["fn"] for result in thresholded_values])
        hausdorfs = [hausdorff for result in thresholded_values for hausdorff in result["hausdorfs"]]

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        mse = np.mean([h ** 2 for h in hausdorfs])
        rmse = np.sqrt(mse)
        hausdorff_50pct = np.percentile(hausdorfs, 50) if len(hausdorfs) > 0 else None
        hausdorff_75pct = np.percentile(hausdorfs, 75) if len(hausdorfs) > 0 else None
        hausdorff_90pct = np.percentile(hausdorfs, 90) if len(hausdorfs) > 0 else None
        hausdorff_95pct = np.percentile(hausdorfs, 95) if len(hausdorfs) > 0 else None

        table_data.append({
            "iou_threshold": threshold,
            f"{seen}_precision": precision,
            f"{seen}_recall": recall,
            f"{seen}_f1": f1,
            f"{seen}_hausdorff_50pct": hausdorff_50pct,
            f"{seen}_hausdorff_75pct": hausdorff_75pct,
            f"{seen}_hausdorff_90pct": hausdorff_90pct,
            f"{seen}_hausdorff_95pct": hausdorff_95pct,
            f"{seen}_mse": mse,
            f"{seen}_rmse": rmse
        })

        # print(f"tabledata: {table_data}")
        if threshold < 0.01:
            base_mse = mse
            base_rmse = rmse

    table = wandb.Table(
        columns=["iou_threshold", f"{seen}_precision", f"{seen}_recall", f"{seen}_f1", f"{seen}_mse", f"{seen}_rmse", f"{seen}_hausdorff_50pct",
                 f"{seen}_hausdorff_75pct", f"{seen}_hausdorff_90pct", f"{seen}_hausdorff_95pct"],
        data=[[row["iou_threshold"], row[f"{seen}_precision"], row[f"{seen}_recall"], row[f"{seen}_f1"],
                 row[f"{seen}_mse"], row[f"{seen}_rmse"],
               row[f"{seen}_hausdorff_50pct"], row[f"{seen}_hausdorff_75pct"], row[f"{seen}_hausdorff_90pct"], row[f"{seen}_hausdorff_95pct"]] for row
              in table_data])

    # calculate mAP
    precision = [row[f"{seen}_precision"] for row in table_data]
    recall = [row[f"{seen}_recall"] for row in table_data]
    pairs = [(r, p) for r, p in zip(recall, precision)]
    pairs.sort(key=lambda x: x[0])
    precision, recall = zip(*pairs)
    last_recall = 0
    mAP = 0
    for i in range(len(recall)):
        r_diff = recall[i] - last_recall
        last_recall = recall[i]
        mAP += r_diff * precision[i]

    # mAP = np.trapz(precision, recall)

    return {
        f"{seen}_mean_iou": avg_iou,
        f"{seen}_table": table,
        f"{seen}_mAP": mAP,
        f"{seen}_base_mse": base_mse,
        f"{seen}_base_rmse": base_rmse
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

