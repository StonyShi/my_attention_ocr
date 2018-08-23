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
import pickle
import time

from text.string_generator import create_strings_from_wikipedia, create_strings_from_new

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
        default="datasets/data",
    )
    parser.add_argument(
        "-wk",
        "--use_wikipedia",
        action="store_true",
        help="Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s",
        default=False,
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
        "-w",
        "--word",
        type=str,
        nargs="?",
        help="cn news word",
        default="新闻"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="The number of images to be created.",
        default=10000
    )

    parser.add_argument(
        "-max",
        "--max_len",
        type=int,
        nargs="?",
        help="The text max len.",
        default=18
    )

    return parser.parse_args()



#python gen_news.py -c 200 -l cn -w 新闻
#python gen_news.py -c 200 -l cn -wk
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

    output_dir = args.output_dir
    count = args.count
    language = args.language
    word = args.word
    max_len = args.max_len

    if args.use_wikipedia:
        words = create_strings_from_wikipedia(2, count*1, language, max_length=max_len)
    else:
        words = create_strings_from_new(2, count*1, language, max_length=max_len, word=word)

    print("words len: ", len(words))
    if args.use_wikipedia:
        fv = "wiki"
    else:
        fv = "news"

    tv = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    file_name = "%s_%s_%s_%d.pickle"%(fv, language, tv, len(words))
    file = open(os.path.join(output_dir, file_name), 'wb')
    pickle.dump(words, file)
    file.close()
    print("save pickle:%s"%(os.path.join(output_dir, file_name)))

    # ww = load_store_data("wiki", output_dir)
    # print(ww)