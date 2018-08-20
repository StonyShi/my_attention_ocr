from Config import Config, GenImage, Progbar, sparse_tuple_from_label,ctc_sparse_from_label,\
    print_net,print_net_line,calculate_distance,get_gb2312,get_gb2312_file,get_logger

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import os,logging
import string, os, json, random,re,errno
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from tqdm import tqdm
from multiprocessing import Pool
import requests
from bs4 import BeautifulSoup
import jieba
import argparse

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Generate synthetic text data for text recognition.')
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default="out/",
    )
    parser.add_argument(
        "--font_dir",
        type=str,
        nargs="?",
        help="The fonts directory",
        default="fonts",
    )
    parser.add_argument(
        "--bg_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default="resource/bgimg/images",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        default=""
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), or cn (Chinese).",
        default="en"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="The number of images to be created.",
        default=1000
    )
    parser.add_argument(
        "-rs",
        "--random_sequences",
        action="store_true",
        help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
        default=False
    )
    parser.add_argument(
        "-let",
        "--include_letters",
        action="store_true",
        help="Define if random sequences should contain letters. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-num",
        "--include_numbers",
        action="store_true",
        help="Define if random sequences should contain numbers. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-sym",
        "--include_symbols",
        action="store_true",
        help="Define if random sequences should contain symbols. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
        default=1
    )
    parser.add_argument(
        "-mxw",
        "--max_length",
        type=int,
        nargs="?",
        help="letter max_length",
        default=20
    )
    parser.add_argument(
        "-miw",
        "--min_length",
        type=int,
        nargs="?",
        help="letter min_length",
        default=10
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        default=False
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        help="Define the number of thread to use for image generation",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        default="jpg",
    )

    parser.add_argument(
        "-wk",
        "--use_wikipedia",
        action="store_true",
        help="Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s",
        default=False,
    )
    parser.add_argument(
        "-news",
        "--use_news",
        action="store_true",
        help="Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s",
        default=True,
    )

    parser.add_argument(
        "-new_w",
        "--new_width",
        type=int,
        nargs="?",
        help="save image IMAGE_WIDTH",
        default=320
    )
    parser.add_argument(
        "-new_h",
        "--new_height",
        type=int,
        nargs="?",
        help="save image IMAGE_HEIGHT",
        default=32
    )
    parser.add_argument(
        "-fs",
        "--font_size",
        type=int,
        nargs="?",
        help=" image font size",
        default=28
    )
    parser.add_argument(
        "-aug",
        "--image_aug",
        action="store_true",
        help=" image aug",
        default=False
    )

    return parser.parse_args()

def create_strings_from_new(minimum_length, count, lang, max_length=200):
    sentences = []
    url = "http://news.baidu.com/ns?word=%E6%96%B0%E9%97%BB&pn={}&cl=2&ct=1&tn=news&rn=20&ie=utf-8&bt=0&et=0"
    if lang == 'en':
        if random.random() > 0.5:
            if random.random() > 0.5:
                url = "http://www.globaltimes.cn/life/food/index{}.html#list"
            else:
                url = "http://www.globaltimes.cn/beijing/index{}.html#list"
        else:
            if random.random() > 0.5:
                url = "http://www.globaltimes.cn/china/profile/index{}.html#list"
            else:
                url = "http://www.globaltimes.cn/china/society/index{}.html#list"
    index = 2
    if lang == 'cn':
        index = 10
    while len(sentences) < count:
        # We fetch a random page
        page = requests.get(url.format(index))
        if lang == 'cn':
            index = index + 10
        else:
            index = index + 1

        soup = BeautifulSoup(page.text, 'html.parser')

        for script in soup(["script", "style"]):
            script.extract()

        # Only take a certain length
        lines = list(filter(
            lambda s:
            len(s.split(' ')) > minimum_length
            and len(s) > 1
            and not "Wikipedia" in s
            and not "Wikipedia" in s
            and not "百度首页" in s
            and not "跳到导航" in s
            and not "维基百科" in s,
            [
                ' '.join(re.findall(r"[\w']+", s.strip()))[:] for s in soup.get_text().splitlines()
            ]
        ))
        __lines = []
        for line in lines:
            if(len(line)) > max_length:
                if lang == 'cn':
                    seg_list = jieba.cut(line, cut_all=False)
                    __lines.extend(seg_list)
                else:
                    seg_list = line.split(' ')
                    __lines.extend(seg_list)
            else:
                __lines.append(line)
        # Remove the last lines that talks about contributing
        sentences.extend(__lines)
    return sentences

def create_strings_from_wikipedia(minimum_length, count, lang, max_length=200):
    """
        Create all string by randomly picking Wikipedia articles and taking sentences from them.
    """
    sentences = []
    if lang == 'cn':
        lang = 'zh'

    while len(sentences) < count:
        # We fetch a random page
        page = requests.get('https://{}.wikipedia.org/wiki/Special:Random'.format(lang))

        soup = BeautifulSoup(page.text, 'html.parser')

        for script in soup(["script", "style"]):
            script.extract()

        # Only take a certain length
        lines = list(filter(
            lambda s:
                len(s.split(' ')) > minimum_length
                and len(s) > 1
                and not "Wikipedia" in s
                and not "Wikipedia" in s
                and not "自由的百科全书" in s
                and not "跳到导航" in s
                and not "维基百科" in s,
            [
                ' '.join(re.findall(r"[\w']+", s.strip()))[:] for s in soup.get_text().splitlines()
            ]
        ))
        __lines = []
        for line in lines:
            if (len(line)) > max_length:
                if lang == 'cn':
                    seg_list = jieba.cut(line, cut_all=False)
                    __lines.extend(seg_list)
                else:
                    seg_list = line.split(' ')
                    __lines.extend(seg_list)
            else:
                __lines.append(line)
        # Remove the last lines that talks about contributing
        #sentences.extend(lines[0:max([1, len(lines) - 5])])
        sentences.extend(__lines)

    return sentences
def create_strings_from_file(filename):
    strings = []
    with open(filename, 'r', encoding="utf8") as f:
        lines = [l.strip()[:] for l in f.readlines()]
        strings = lines
    return strings
##python gen_run.py -t 3 -fs 28 -new_h 32 -new_w 320 -w 2 -c 200 -news -mxw 18 -miw 15 -l cn -e png -aug  --output_dir out
##python gen_run.py  -t 5 -fs 28 -new_h 32 -new_w 320 -w 2 -c 200 -news -mxw 18 -miw 15 -l cn -e png -aug --output_dir out
##python gen_run.py -t 5 -fs 28 -new_h 32 -new_w 320 -w 2 -c 100 -news -mxw 18 -miw 15 -l cn -e png --output_dir out2
if __name__ == '__main__':

    # Argument parsing
    args = parse_arguments()
    print(args)

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    IMAGE_WIDTH = args.new_width
    IMAGE_HEIGHT = args.new_height
    font_size = args.font_size
    bg_dir = args.bg_dir
    language = args.language
    output_dir = args.output_dir
    max_length = args.max_length
    min_length = args.min_length
    count = args.count
    extension = args.extension
    font_dir = args.font_dir


    config = Config(gb2312=True)
    gen_img = GenImage(config=config,
                       width=IMAGE_WIDTH,
                       height=IMAGE_HEIGHT,
                       min_size=min_length,
                       max_size=max_length,
                       fonts=font_dir,
                       font_size=font_size)
    print(gen_img.fonts)

    if args.use_wikipedia:
        words = create_strings_from_wikipedia(args.length, count*2, language, max_length=max_length)
    elif args.input_file != '':
        words = create_strings_from_file(args.input_file)
    else:
        words = create_strings_from_new(args.length, count*2, language, max_length=max_length)

    bg_list = gen_img.get_img_file(bg_dir)
    bg_list.append(None)
    colors = [] # gen_img.choice_text_color(text_color)
    colors.append(None)

    # def generate_from_tuple(batch_size):
    #     gen_img.gen_img_dir(1, output_dir, bg_dir=bg_dir, ext=extension, words=words)
    #
    #
    # p = Pool(args.thread_count)
    # for _ in tqdm(p.imap_unordered(generate_from_tuple, zip([1]*count)), total=count):
    #     pass
    # p.terminate()

    print("gen_img_dir >>> size: %d, dir: %s  " % (count, output_dir))

    def generate_from_index(index):
        gen_img.gen_one_img_dir(index, output_dir, bg_list=bg_list, colors=colors,  ext=extension, words=words, threads=args.thread_count, aug=args.image_aug)

    p = Pool(args.thread_count)
    for _ in tqdm(p.imap_unordered(generate_from_index, [i for i in range(0, count)]), total=count):
        pass
    p.terminate()

