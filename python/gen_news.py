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

    return parser.parse_args()


def create_strings_from_new(minimum_length, count, lang, word, max_length=20):
    sentences = []
    cn_list = [
        "http://news.baidu.com/ns?word="+word+"&pn={}&cl=2&ct=1&tn=news&rn=20&ie=utf-8&bt=0&et=0"
        ]
    en_list = ["https://www.newsweek.com/search/site/news?page={}"
        , "https://www.newsweek.com/search/site/word?page={}"
        , "http://www.globaltimes.cn/life/food/index{}.html#list"
        , "http://www.globaltimes.cn/beijing/index{}.html#list"
        , "http://www.globaltimes.cn/china/profile/index{}.html#list"
        , "http://www.globaltimes.cn/china/society/index{}.html#list"]

    def get_url(index):
        if lang == 'cn':
            url = random.choice(cn_list)
        if lang == 'en':
            url = random.choice(en_list)
        return url.format(index)
    index = 2
    if lang == 'cn':
        index = 10
    while len(sentences) < count:
        # We fetch a random page
        page = requests.get(get_url(index))
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
        __lines = list(filter(
            lambda s: len(s.strip()) > 1,
            __lines
        ))
        sentences.extend(__lines)
    return sentences

def create_strings_from_wikipedia(minimum_length, count, lang, max_length=20):
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
        __lines = list(filter(
            lambda s: len(s.strip()) > 1,
            __lines
        ))
        sentences.extend(__lines)

    return sentences
def load_store_data(type, data_dir):
    def get_data(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
    words = []
    for f in os.listdir(data_dir):
        if not f.endswith('.pickle'):
            continue
        if not f.startswith(type):
            continue
        word = get_data(os.path.join(data_dir, f))
        if word is not None:
            words.extend(word)
    return words
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

    if args.use_wikipedia:
        words = create_strings_from_wikipedia(2, count*1, language, max_length=20)
    else:
        words = create_strings_from_new(2, count*1, language, word=word,max_length=20)

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