from Config import gen_crop_bg
import os


if __name__ == '__main__':
    import PIL.Image as Image

    img_dir = "resource/bgimg/"
    img_dir_out = "resource/bgimg/images/"
    x, y = 480, 48
    for bg in os.listdir(img_dir):
        gen_crop_bg(Image.open(os.path.join(img_dir + bg)), 500, img_dir_out, x, y)