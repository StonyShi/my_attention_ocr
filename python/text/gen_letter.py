import os, sys, re, random
try:
    from data_generator import FakeTextDataGenerator
except Exception as e:
    import sys, os
    parentdir = os.path.dirname(os.path.abspath(__file__))
    # print("---------------")
    # print(parentdir)
    sys.path.insert(0, parentdir)
    from data_generator import FakeTextDataGenerator

def read_dict(filename='resource/new_dic2.txt', null_character=u'\u2591'):
    import tensorflow as tf
    import logging
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
    def __init__(self,min_size,max_size):
        self.min_size = min_size
        self.max_size = max_size
        chinese_dict = read_dict()
        words = list(chinese_dict.keys())
        self.charset = ''.join(words)
        self.dataGenerator = FakeTextDataGenerator()
    def is_valid_char(self, char):
        charset = self.charset
        if len(char) > 1:
            for c in char:
                if c not in charset:
                    return True
            return False
        else:
            return char in charset
    def get_letter(self, wds):
        letter = []
        len_str = 0
        size = random.randrange(self.min_size, self.max_size)

        def _append(c):
            if len(c) == 1:
                letter.append(c)
            elif len(c) > 1:
                # letter.append(" ")
                letter.extend(list(c))

        for i in range(size):
            c = random.choice(wds)
            if (len_str + len(c)) > size:
                _append(c)
                break
            while True:
                if not self.is_valid_char(c):
                    len_str = len_str + len(c)
                    _append(c)
                    break
                c = random.choice(wds)
        if len(letter) > self.max_size:
            return letter[:self.max_size]
        return letter

    def gen_image(self, index, words, fonts, out_dir, height, extension,
                  skewing_angle, random_skew, blur, random_blur,
                  background_type, distorsion_type, distorsion_orientation,
                  is_handwritten, name_format, new_width,new_height):
        text = ''.join(self.get_letter(words))
        font = random.choice(fonts)
        self.dataGenerator.generate(index, text, font, out_dir, height, extension,
                  skewing_angle, random_skew, blur, random_blur,
                  background_type, distorsion_type, distorsion_orientation,
                  is_handwritten, name_format, new_width,new_height)
