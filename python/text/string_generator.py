import os
import random
import re
import string
import requests
import jieba
import pickle

from bs4 import BeautifulSoup

def create_strings_from_new(minimum_length, count, lang, max_length=200):
    sentences = []
    cn_list = [
        "http://news.baidu.com/ns?word=%E4%BB%8A%E6%97%A5%E5%A4%B4%E6%9D%A1&pn={}&cl=2&ct=1&tn=news&rn=20&ie=utf-8&bt=0&et=0"
        , "http://news.baidu.com/ns?word=%E6%96%B0%E9%97%BB&pn={}&cl=2&ct=1&tn=news&rn=20&ie=utf-8&bt=0&et=0"]
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
        __lines = list(filter(
            lambda s: len(s.strip()) > 1,
            __lines
        ))
        sentences.extend(__lines)

    return sentences

def create_strings_from_file(filename):
    strings = []
    with open(filename, 'r', encoding="utf8") as f:
        lines = [l.strip()[:] for l in f.readlines()]
        strings = lines
    return strings

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


def get_font_file(font_dir):
    bg_list = []
    if font_dir == None:
        return bg_list
    for f in os.listdir(font_dir):
        try:
            if not f.endswith(('.TTF', '.ttf', '.ttc', '.TTC')):
                continue
            bg_list.append(os.path.join(font_dir, f))
        except Exception as e:
            pass
    return bg_list