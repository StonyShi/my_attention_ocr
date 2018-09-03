import os
import random
import re
import string
import requests
import jieba
import pickle

from bs4 import BeautifulSoup

def create_strings_from_new(minimum_length, count, lang, max_length=20, word="新闻"):
    sentences = []
    cn_list = [
        "http://news.baidu.com/ns?word=" + word + "&pn={}&cl=2&ct=1&tn=news&rn=20&ie=utf-8&bt=0&et=0"
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
        __lines = lines
        # __lines = []
        # for line in lines:
        #     if(len(line)) > max_length*2:
        #         for i in range(0, len(line), max_length):
        #             start = i
        #             end = i + max_length
        #             strs = (line[start:end]).strip()
        #             __lines.append(strs)
        #         # if lang == 'cn':
        #         #     seg_list = jieba.cut(line, cut_all=False)
        #         #     __lines.extend(seg_list)
        #         # else:
        #         #     seg_list = line.split(' ')
        #         #     __lines.extend(seg_list)
        #     else:
        #         __lines.append(line)
        # Remove the last lines that talks about contributing
        __lines = list(filter(
            lambda s: len(s.strip()) > 2,
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
        __lines = lines
        # __lines = []
        # for line in lines:
        #     if (len(line)) > max_length*2:
        #         for i in range(0, len(line), max_length):
        #             start = i
        #             end = i + max_length
        #             strs = (line[start:end]).strip()
        #             __lines.append(strs)
        #         # if lang == 'cn':
        #         #     seg_list = jieba.cut(line, cut_all=False)
        #         #     __lines.extend(seg_list)
        #         # else:
        #         #     seg_list = line.split(' ')
        #         #     __lines.extend(seg_list)
        #     else:
        #         __lines.append(line)
        # Remove the last lines that talks about contributing
        #sentences.extend(lines[0:max([1, len(lines) - 5])])
        __lines = list(filter(
            lambda s: len(s.strip()) > 2,
            __lines
        ))
        sentences.extend(__lines)

    return sentences


def merge_file(input_dir, out_file):
    files = os.listdir(input_dir)
    files = list(filter(lambda x: x.endswith(('.TXT', '.txt')), files))

    index = 0
    out = open(out_file, 'w')

    def __writer_line(line):
        out.write("%s\n" % line)

    for fx in files:
        filename = os.path.join(input_dir, fx)
        try:
            with open(filename, 'r', encoding="utf8") as f:
                for l in f.readlines():
                    l = l.strip()
                    if (len(l) > 1):
                        index = index + 1
                        __writer_line(l)
        except Exception as e:
            print("file open error: %s"%filename)
            pass
    print("count index = %d" % index)
    out.close()

def create_strings_from_file(filename, max_length):

    parentdir = (os.path.abspath(__file__))
    if not os.path.exists(filename):
        parentdir = os.path.dirname(parentdir)
        for i in range(3):
            if os.path.exists(os.path.join(parentdir, filename)):
                filename = os.path.join(parentdir, filename)
                break
            if not os.path.exists(os.path.join(parentdir, filename)):
                parentdir = os.path.dirname(parentdir)


    strings = []
    with open(filename, 'r', encoding="utf8") as f:
        lines = [l.strip()[:] for l in f.readlines()]
        strings = lines
    #print("strings: ", strings)
    return strings

def get_filename(filename, depth=3):
    parentdir = (os.path.abspath(__file__))
    if not os.path.exists(filename):
        parentdir = os.path.dirname(parentdir)
        for i in range(depth):
            if os.path.exists(os.path.join(parentdir, filename)):
                filename = os.path.join(parentdir, filename)
                break
            if not os.path.exists(os.path.join(parentdir, filename)):
                parentdir = os.path.dirname(parentdir)
    return filename

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