from utils import get_data

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob
import os

def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    # IMPLEMENT THIS FUNCTION

    paths = glob.glob('./data/images/*')

    # img = plt.imread(paths[0])
    # plt.imshow(img)
    # plt.show()

    gtdict = {}
    for gt in ground_truth:
        gtdict[gt['filename']] = gt

    # fig, axes = plt.subplots(4,5, figsize=(20,10))

    # for i, ax in enumerate(axes.flat):
    #     print(i)
    #     img = plt.imread(paths[i])
    #     ax.imshow(img)
    #     ax.axis("off")

    fig, axes = plt.subplots(4,5,figsize=(20,10))

    for i in range(20):
        x = i//5
        y = i%5

        axes[x,y].axis('off')

        img = plt.imread(paths[i])
        axes[x,y].imshow(img)

        colors = {1:[1,0,0], 2:[0,1,0], 4:[0,0,1]}

        filename = os.path.basename(paths[i])
        for bbox, cl in zip(gtdict[filename]['boxes'],gtdict[filename]['classes']):
            rec = Rectangle((bbox[1],bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], facecolor = 'none', edgecolor=colors[cl])
            axes[x,y].add_patch(rec)


    #     ax.imgshow(paths[i])

    # for i in range(20):
    #     x = i/4
    #     y = i%5

    #     axes[x,y].imgshow(paths[i])
    plt.show()


if __name__ == "__main__": 
    ground_truth, _ = get_data()
    viz(ground_truth)