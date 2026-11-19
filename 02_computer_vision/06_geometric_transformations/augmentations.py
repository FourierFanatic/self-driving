import copy
import json

import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from utils import check_results, display_results


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    - [xmin, ymin, xmax, ymax]
    """
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])
    
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    union = gt_area + pred_area - intersection
    return intersection / union, [xmin, ymin, xmax, ymax]


def hflip(img, bboxes):
    """
    horizontal flip of an image and annotations
    args:
    - img [PIL.Image]: original image
    - bboxes [list[list]]: list of bounding boxes
    return:
    - flipped_img [PIL.Image]: horizontally flipped image
    - flipped_bboxes [list[list]]: horizontally flipped bboxes
    """
    # IMPLEMENT THIS FUNCTION
    img_arr = np.array(img)
    img_arr = img_arr[:,::-1]
    flipped_img = Image.fromarray(img_arr)

    # for box in bboxes:
    flipped_bboxes = []
    for box in bboxes:
        flipped_bboxes.append([box[0],img.size[0] - box[3],box[2],img.size[0] - box[1]])


    return flipped_img, flipped_bboxes


def resize(img, boxes, size):
    """
    resized image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - size [array]: 1x2 array [width, height]
    returns:
    - resized_img [PIL.Image]: resized image
    - resized_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    resized_image = img.resize(size)

    scale_w = size[0]/img.size[0]
    scale_h = size[1]/img.size[1]
    # print(img.size)
    # print("scales",scale_w, scale_h)

    # print(boxes[3][1])
    # print(boxes[3][1]*scale_w)
    # resized_boxes = copy.copy(boxes)
    resized_boxes = np.array(boxes)
    resized_boxes[:,[0,2]] = resized_boxes[:,[0,2]]*scale_h
    resized_boxes[:,[1,3]] = resized_boxes[:,[1,3]]*scale_w
    return resized_image, resized_boxes


def random_crop(img, boxes, classes, crop_size, min_area=100):
    """
    random cropping of an image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - crop_size [array]: 1x2 array [width, height]
    - min_area [int]: min area of a bbox to be kept in the crop
    returns:
    - cropped_img [PIL.Image]: resized image
    - cropped_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    w, h = img.size
    x1 = np.random.randint(0, w - crop_size[0])
    y1 = np.random.randint(0, h - crop_size[1])
    x2 = x1 + crop_size[0]
    y2 = y1 + crop_size[1]

    cropped_image = img.crop((x1,y1,x2,y2))

    cropped_boxes = []
    cropped_classes = []

    for box,cl in zip(boxes,classes):
        inter,_ = calculate_iou(box,[y1,x1,y2,x2])
        # print("box",box)
        # print("crop",[y1,x1,y2,x2])
        # print("inter is", inter)
        if inter==0: continue
        a = max(x1,box[1])
        b = max(y1,box[0])
        c = min(x2,box[3])
        d = min(y2,box[2])

        cropped_boxes.append([b-y1,a-x1,d-y1,c-x1])
        cropped_classes.append(cl)
        # print("appended",[b,a,d,c])
    return cropped_image, cropped_boxes, cropped_classes

if __name__ == '__main__':
    # fix seed to check results
    np.random.seed(48)
    
    # open annotations
    with open('data/ground_truth.json') as f:
        ground_truth = json.load(f)
    
    # filter annotations and open image
    filename = ground_truth[-1]['filename']
    bboxes = ground_truth[-1]['boxes']
    classes = ground_truth[-1]['classes']
    # print(filename)
    
    
    # check horizontal flip, resize and random crop
    img1 = Image.open('data/images/'+filename)
    img_check = Image.open('data/augmented/flipped.png')
    # fig, [ax1,ax2] = plt.subplots(1,2,figsize=(20,10))

    img2,flipped_bboxes = hflip(img1,bboxes)
    display_results(img1, bboxes, img2, flipped_bboxes)
    # print("size:",img1.size)
    # ax1.imshow(img1)
    # ax2.imshow(img2)
    # ax1.axis('off')
    # ax2.axis('off')
    # plt.tight_layout()
    # plt.show()

    # bbcheck = np.load('data/augmented/flipped.npy')
    # print(len(flipped_bboxes))
    # print(flipped_bboxes)
    # print(bboxes)
    # print(bbcheck.size)
    # print(bbcheck)

    check_results(img2, flipped_bboxes, 'hflip')

    # Check resize

    img_resized, boxes_resized = resize(img1, bboxes, [640,640])
    display_results(img1, bboxes, img_resized, boxes_resized)
    check_results(img_resized, boxes_resized, 'resize')

    # Check crop

    cropped_image, cropped_boxes, cropped_classes = random_crop(img1,bboxes,classes, [512,512], 100)
    display_results(img1, bboxes, cropped_image, cropped_boxes)
    check_results(cropped_image, cropped_boxes, 'random_crop', cropped_classes)

    # use check_results defined in utils.py for this
