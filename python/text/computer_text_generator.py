import cv2
import math
import random
import os
import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageFilter


class ComputerTextGenerator(object):
    @classmethod
    def generate(cls, text, font, text_color):
        image_font = ImageFont.truetype(font=font, size=32)
        text_width, text_height = image_font.getsize(text)

        txt_img = Image.new('L', (text_width, text_height), 255)

        txt_draw = ImageDraw.Draw(txt_img)
        if text_color < 0:
            text_color = random.randint(1, 127)
        #random.randint(1, 80)
        txt_draw.text((0, 0), text, fill=text_color, font=image_font)

        return txt_img

    @classmethod
    def text_color(self):
        return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))
