import cv2
import numpy as np
import scipy
import torch
from scipy.ndimage import label


def multi_transform(img, model, device=torch.device("cuda"), threshold=0.5):
    # print(f"img shape: {img.shape}")
    img = img.to(device)
    rotations = torch.cat([img, img.rot90(1, [2, 3]), img.rot90(2, [2, 3]), img.rot90(3, [2, 3])], dim=0)
    mirrors = torch.cat([img.flip(2), img.flip(3)], dim=0)

    rotation_preds = model.segment(rotations)
    rotation_preds = torch.nn.Sigmoid()(rotation_preds)
    rotation_preds = rotation_preds > threshold

    mirrors_preds = model.segment(mirrors)
    mirrors_preds = torch.nn.Sigmoid()(mirrors_preds)
    mirrors_preds = mirrors_preds > threshold

    img = img.cpu()
    rotations = rotations.cpu()
    mirrors = mirrors.cpu()

    # rotate the predictions back
    rotations_mask_back = [rotation_preds[0], rotation_preds[1].rot90(3, [1, 2]), rotation_preds[2].rot90(2, [1, 2]),
                           rotation_preds[3].rot90(1, [1, 2])]
    mirrors_mask_back = [mirrors_preds[0].flip(1), mirrors_preds[1].flip(2)]

    total_mask_back = rotations_mask_back + mirrors_mask_back
    total_mask_back = torch.stack(total_mask_back, dim=0).cpu()

    return (torch.sum(total_mask_back, dim=0) > 4).unsqueeze(0)

# The portion for making points along perimiter of mask
def find_perimeter_points(tensor):
    """Finds and sorts the perimeter points in the tensor using contour detection."""
    # Convert tensor to a numpy array and then to an OpenCV-compatible format

    array = tensor.numpy().astype(np.uint8) if isinstance(tensor, torch.Tensor) else tensor.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Extract the first contour and reshape it to get the perimeter points
    contour = contours[0].reshape(-1, 2)

    return contour


def calculate_cumulative_distances(perimeter_points):
    """Calculates the cumulative distances between consecutive perimeter points."""
    distances = [0]  # Start with zero distance
    for i in range(1, len(perimeter_points)):
        dist = np.linalg.norm(perimeter_points[i] - perimeter_points[i - 1])
        distances.append(distances[-1] + dist)
    return distances


def find_equally_spaced_points(perimeter_points, cumulative_distances, N):
    """Finds N equally spaced points along the perimeter."""
    total_perimeter_length = cumulative_distances[-1]
    target_distances = np.linspace(0, total_perimeter_length, N, endpoint=False)

    equally_spaced_points = []
    idx = 0

    for td in target_distances:
        while cumulative_distances[idx] < td:
            idx += 1
        equally_spaced_points.append(perimeter_points[idx])

    return equally_spaced_points


def extract_focus(frame, min_area=0, max_detections=2):
    pred_instances, pred_labels = label(frame)

    # get the areas of the detections
    valid_labels = []
    for i in range(1, pred_labels + 1):
        area = np.sum(pred_instances == i)
        if area >= min_area:
            valid_labels.append((i, area))

    valid_labels = sorted(valid_labels, key=lambda x: x[1], reverse=True)
    valid_labels = valid_labels[:max_detections]
    assert len(valid_labels) <= max_detections

    mask = np.zeros_like(frame)
    for i, _ in valid_labels:
        mask[pred_instances == i] = 1

    return mask


def load_matlab_DLTs(matlab_path, default = 0):
    org_data = scipy.io.loadmat(matlab_path, simplify_cells=True)

    dlt_1 = org_data["data"]["cal"]["coeff"]["DLT_1"]
    dlt_2 = org_data["data"]["cal"]["coeff"]["DLT_2"]
    dlt_3 = org_data["data"]["cal"]["coeff"]["DLT_3"]



    if len(dlt_1) == 11:
        dlt_1 = np.append(dlt_1, default)

    if len(dlt_2) == 11:
        dlt_2 = np.append(dlt_2, default)

    if len(dlt_3) == 11:
        dlt_3 = np.append(dlt_3, default)

    dlts = [dlt_1, dlt_2, dlt_3]
    return dlts