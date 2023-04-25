import matplotlib.pyplot as plt
import os

def save_image(image, path, title=''):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.title(title)
    plt.savefig(path)
    plt.close()
