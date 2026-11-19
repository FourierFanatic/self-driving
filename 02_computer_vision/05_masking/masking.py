from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    # IMPLEMENT THIS FUNCTION
    img = Image.open(path)
    img_array  = np.array(img)
    # print(img_array.shape)
    # print("Hello there!!!")
    mask = (img_array > color_threshold).all(-1)
    # print(mask.shape)

    return img, mask




def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
    # IMPLEMENT THIS FUNCTION

    fig, axes = plt.subplots(1,3, figsize=(20,10))

    axes[0].imshow(img)
    axes[1].imshow(mask)

    axes[2].imshow(img*mask[...,None])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)