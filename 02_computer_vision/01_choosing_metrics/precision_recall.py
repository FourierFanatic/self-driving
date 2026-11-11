import numpy as np

from iou import calculate_ious
from utils import get_data


def precision_recall(ious, gt_classes, pred_classes):
    """
    calculate precision and recall
    args:
    - ious [array]: NxM array of ious
    - gt_classes [array]: 1xN array of ground truth classes
    - pred_classes [array]: 1xM array of pred classes
    returns:
    - precision [float]
    - recall [float]
    """
    # print(gt_classes)
    # print(pred_classes)
    # print("mat", gt_classes[:, None])

    tp = ((gt_classes[:, None] == pred_classes)*(ious >= 0.5)).sum()
    # print("tp",tp)
    # clas = gt_classes[:,None] == pred_classes
    # io = ious >= 0.5
    # print("clas",clas)
    # print("io",io)
    # print("mult",clas*io)
    # print("count",(clas*io).sum())
    precision = tp / (ious >= 0.5).sum()
    # print(precision)
    # print("gt", gt_classes.size)
    recall = tp / (gt_classes.size)
    # print(recall)
    
    return precision, recall


if __name__ == "__main__": 

    # my test
    # precision, recall = precision_recall(np.array([[0.5, 0.3, 0.1],[0.9, 0.1, 0.7]]), np.array([1,2]), np.array([1,1,2]))


    ground_truth, predictions = get_data()
    
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    gt_classes = [g['classes'] for g in ground_truth if g['filename'] == filename][0]
    gt_classes = np.array(gt_classes)
    

    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    pred_classes = [p['classes'] for p in predictions if p['filename'] == filename][0]
    pred_classes = np.array(pred_classes)
    
    ious = calculate_ious(gt_bboxes, pred_boxes)
    precision, recall = precision_recall(ious, gt_classes, pred_classes)