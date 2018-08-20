import numpy as np
import random

def add_noise(img, num=5000):
    # 随机生成椒盐
    rows, cols, dims = img.shape
    for i in range(num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y, :] = 255
    return img

def add_rotate(image, min=0, max=5):
    random_angle = random.randint(0 - min, max)
    return image.rotate(random_angle, expand=1)