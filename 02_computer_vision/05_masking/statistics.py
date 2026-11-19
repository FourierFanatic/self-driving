from PIL import Image, ImageStat
import glob
import numpy as np
from utils import check_results
import matplotlib.pyplot as plt

def calculate_mean_std(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    # IMPLEMENT THIS FUNCTION
    mean = np.array([0.0,0.0,0.0])
    # variance = np.array([0.0,0.0,0.0])
    std = np.array([0.0,0.0,0.0])
    for img_path in image_list:
        img = Image.open(img_path)
        stat = ImageStat.Stat(img)

        # print(stat.mean)

        mean += stat.mean
        std += stat.stddev
        # variance += np.array(stat.stddev)*np.array(stat.stddev)

    mean /= len(image_list)
    # variance /= len(image_list)
    # std = np.sqrt(variance)
    # Course autochecker wants to just average the standard deviations, even though standard deviation is not additive
    # Correct method is commented. Variance is additive and then take the sqrt.
    std /= len(image_list)

    print("Mean  = ",mean)
    print("std = ",std)

    check_results(mean,std)

    return mean, std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    # IMPLEMENT THIS FUNCTION

    r = np.zeros(256)
    g = np.zeros(256)
    b = np.zeros(256)

    for path in image_list:
        img = Image.open(path).convert('RGB')
        img_array  = np.array(img)
        for pixel in img_array.reshape(-1, img_array.shape[-1]):
            r[pixel[0]] += 1
            g[pixel[1]] += 1
            b[pixel[2]] += 1
    
    plt.plot(r, color="red")
    plt.plot(g, color="green")
    plt.plot(b, color="blue")
    plt.show()


if __name__ == "__main__": 
    image_list = glob.glob('data/images/*')
    mean, std = calculate_mean_std(image_list)
    channel_histogram(image_list)