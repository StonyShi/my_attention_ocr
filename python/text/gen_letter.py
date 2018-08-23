import os, sys, re, random

def read_dict(filename='resource/new_dic2.txt', null_character=u'\u2591'):
    import tensorflow as tf
    import logging

    parentdir = (os.path.abspath(__file__))
    for i in range(3):
        if os.path.exists(filename):
            break
        if not os.path.exists(filename):
            parentdir = os.path.dirname(parentdir)
            filename = os.path.join(parentdir, filename)

    print("read_dict >>>>> filename: ", filename)
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                charset[" "] = 0
                logging.warning('incorrect charset file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2)  # .decode('utf-8')
            if char == '<nul>':
                char = null_character
            #charset[code] = char
            charset[char] = code
    return charset



class GenLetter(object):
    def __init__(self, min_size, max_size, charset=None):
        self.min_size = min_size
        self.max_size = max_size
        #if charset == None:
        chinese_dict = read_dict()
        words = list(chinese_dict.keys())
        self.charset = ''.join(words)
        # else:
        #     self.charset = charset

    def is_valid_char(self, char):
        if "/" in char:
            return True
        if "网站声明" in char:
            return True
        charset = self.charset
        for c in char:
            if c not in charset:
                return True
        return False

    def get_letter(self, wds):
        letter = []
        len_str = 0
        size = random.randrange(self.min_size, self.max_size)

        def _append(c):
            if len(c) == 1:
                letter.append(c)
            elif len(c) > 1:
                if len(c) > 2:
                    letter.append(" ")
                letter.extend(list(c))

        min_size = self.min_size
        max_size = self.max_size
        for i in range(size):
            char = random.choice(wds)
            #遇到长文本截断返回
            char_len = len(char)
            #符号长度直接返回
            if min_size < char_len < (max_size+1) and not self.is_valid_char(char):
                return list(char)

            if (char_len > max_size*2):
                for xx in range(max_size):
                    pos = random.randrange(0, char_len - max_size - 1)
                    strs = (char[pos:(pos + size)]).strip()
                    if not self.is_valid_char(strs):
                        return list(strs)

            if (len_str + len(char)) > size:
                _append(char)
                break
            while True:
                if not self.is_valid_char(char):
                    len_str = len_str + len(char)
                    _append(char)
                    break
                char = random.choice(wds)
        if len(letter) > self.max_size:
            return letter[:self.max_size]
        return letter