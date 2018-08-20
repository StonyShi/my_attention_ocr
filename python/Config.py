# ```
# ````
import string, os, json, random, time, sys, logging,re
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
import cv2
from tqdm import tqdm
from multiprocessing import Pool


def gen_crop_bg(im, size, out_dir, width, height):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    bX, bY = im.size
    x, y = width, height
    for i in range(size):
        xOffset = random.randrange(0, (bX-x))  # 长度,横轴; 左边:0, 右边: backX-imageX
        yOffset = random.randrange(0, (bY-y))  # 高度,竖轴; 顶部:0, 底部: backY-imageY，
        box = [xOffset, yOffset, (x+xOffset), (y+yOffset)]
        ii = im.crop(box=box) # x0,y0,x1,y1 (left, upper, right, lower)
        #print(ii.size)
        img_filename = '{:05}_{:02}.jpg'.format(i, random.randrange(10, 90))
        ii.save(os.path.join(out_dir, img_filename))

def gen_letter_json(out_name):
    charset = string.ascii_letters + string.digits
    save_json(out_name, charset)


def gen_smple_chinese(out_name):
    cs = []
    for head in range(0xb0, 0xf7):
        for body in range(0xa1, 0xfe):
            val = f'{head:x}{body:x}'
            try:
                cs.append(bytes.fromhex(val).decode('gb2312'))
            except UnicodeDecodeError:
                pass
    charset = string.ascii_letters + string.digits + ''.join(cs)
    save_json(out_name, charset)

def get_gb2312():
    cs = []
    #第一字节0xB0-0xF7（对应区号：16－87
    #第二个字节0xA1-0xFE（对应位号：01－94）
    for head in range(0xb0, 0xf7):
        for body in range(0xa1, 0xfe):
            val = f'{head:x}{body:x}'
            try:
                cs.append(bytes.fromhex(val).decode('gb2312'))
            except UnicodeDecodeError:
                pass
    return cs

def get_gb2312_file(name='resource/gb2312_list.txt'):
    if os.path.exists(name):
        return name
    cs = get_gb2312()
    with open(name, 'w') as f:
        for c in cs:
            f.write(c)
    return name

def read_dict(filename='resource/new_dic2.txt', null_character=u'\u2591'):
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
def gen_chinese(out_name):
    cs = []
    for c in range(0x4e00, 0x9fbf):
        try:
            cs.append(chr(c))
        except UnicodeDecodeError:
            pass
    charset = string.ascii_letters + string.digits + ''.join(cs)
    save_json(out_name, charset)


def save_json(out_name, charset):
    encode_maps = {}
    for i, char in enumerate(charset):
        encode_maps[char] = i
    with open(out_name, 'a') as outfile:
        json.dump(encode_maps, outfile, ensure_ascii=False)


class Config(object):
    def __init__(self, charset='0123456789', gb2312=False):
        if gb2312:
            chinese_dict = read_dict()
            words = list(chinese_dict.keys())
            # gb2312_name = get_gb2312_file()
            # chinese_word = open(gb2312_name, 'r').read()
            # charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?, " + chinese_word
            charset = words
        self.__doInit__(charset)

    def __doInit__(self, charset):
        self.charset = charset
        self.SPACE_INDEX = 0
        self.SPACE_TOKEN = ''

        unique = set(charset)
        unique = list(unique)
        unique.sort()
        # Supported characters

        self.CHAR_VECTOR = unique
        # Number of classes
        self.NUM_CLASSES = len(self.CHAR_VECTOR) + 1

        self.ENCODE_MAPS = {}
        self.DECODE_MAPS = {}
        for i, char in enumerate(self.CHAR_VECTOR, 1):
            self.ENCODE_MAPS[char] = i
            self.DECODE_MAPS[i] = char
        self.ENCODE_MAPS[self.SPACE_TOKEN] = self.SPACE_INDEX
        self.DECODE_MAPS[self.SPACE_INDEX] = self.SPACE_TOKEN

    def get_char_index(self, char):
        return self.ENCODE_MAPS.get(char)

    def get_index_char(self, index):
        return self.DECODE_MAPS.get(index)

    def gen_word(self):
        with open('chinese_word.txt', 'w') as f:
            for c in range(0x4e00, 0x9fbf):
                f.write(chr(c))

    def get_word(self):
        return self.charset

    def get_charset(self):
        return self.CHAR_VECTOR

    def text_to_ids(self, text):
        ids = [self.get_char_index(x) for x in text]
        ids = [x for x in ids if x]
        # if len(ids) == 0:
        #     ids.append(0)
        return ids

    def ids_to_text(self, ids):
        return ''.join([self.get_index_char(x) for x in ids if (0 < x < self.NUM_CLASSES)])

    def is_valid_char(self, char):
        charset = self.charset
        if len(char) > 1:
            for c in char:
                if c not in charset:
                    return True
            return False
        else:
            return char in charset

    def decode_dense_code(self, dense_code):
        return [self.ids_to_text(code) for code in dense_code]

    def decode_pred(self, y_pred):
        # [max_time x batch_size x num_classes] > [batch_size x max_time x num_classes]
        y_pred_code = y_pred.transpose((1, 0, 2)).argmax(axis=2)
        return [self.ids_to_text(code) for code in y_pred_code]


class GenImage(object):
    def __init__(self, config: Config, width=128, height=32, max_size=8, min_size=3, font_size=30, fonts="fonts"):
        self.config = config
        self.width = width
        self.height = height
        self.min_size = min_size
        self.max_size = max_size
        self.charset = config.charset

        if os.path.isdir(fonts):
            font_list = self.get_font_file(fonts)
            _fonts = []
            for f in font_list:
                _fonts.append(ImageFont.truetype(f, int(font_size)))
            self.fonts = _fonts
        else:
            self.fonts = None
            self.font = ImageFont.truetype(fonts, int(font_size))

    def rest_size(self, width, height, max_size=8, min_size=3, fonts="fonts/card-id.TTF"):
        self.width = width
        self.height = height
        self.min_size = min_size
        self.max_size = max_size
        self.font = ImageFont.truetype(fonts, (height - 2))

    def bg_color(self):
        return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

    def text_color(self):
        return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

    def get_letter(self, wds):
        letter = []
        len_str = 0
        size = random.randrange(self.min_size, self.max_size)
        def _append(c):
            if len(c) == 1:
                letter.append(c)
            elif len(c) > 1:
                #letter.append(" ")
                letter.extend(list(c))
        for i in range(size):
            c = random.choice(wds)
            if (len_str + len(c)) > size:
                _append(c)
                break
            while True:
                if not self.config.is_valid_char(c):
                    len_str = len_str + len(c)
                    _append(c)
                    break
                c = random.choice(wds)
        if len(letter) > self.max_size:
            return letter[:self.max_size]
        return letter

    def is_none(self, oj):
        try:
            if np.any(oj) == None:
                return True
        except Exception as e:
            pass
        return False

    def get_font_file(self, font_dir):
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


    def get_img_file(self, bg_dir):
        bg_list = []
        if bg_dir == None:
            return bg_list
        for f in os.listdir(bg_dir):
            try:
                if not f.endswith(('.gif', '.jpg', '.png')):
                    continue
                bg_list.append(os.path.join(bg_dir, f))
            except Exception as e:
                pass
        return bg_list


    def gen_one_img_dir(self, index, out_dir, bg_list, colors, words=None, aug=True, ext="jpg", threads=1):
        img, letter = self.gen_image(bg_img=random.choice(bg_list), text_color=random.choice(colors), words=words)
        img_filename = '{:09}_{}_{:03}.{}'.format(index, letter, random.randrange(10, 900), ext)
        if aug:
            image = img
            if random.random() > 0.50:
                image = add_rotate2(image)
            image = np.array(image)

            if random.random() > 0.50:
                image = add_noise(image, min=10, max=300)
            aug_random = random.random()
            if aug_random > 0.50:
                image = add_erode(image)
            else:
                image = add_dilate(image)

            filename = os.path.join(out_dir, img_filename)
            cv2.imwrite(filename, image)
        else:
            img.save(os.path.join(out_dir, img_filename))


    def gen_img_dir(self, batch_size, out_dir, bg_dir=None, text_color=None, words=None, aug=True, ext="jpg", threads=1):
        print("gen_img_dir >>> size: %d, dir: %s  "%(batch_size,out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        bg_list = self.get_img_file(bg_dir)
        bg_list.append(None)
        colors = self.choice_text_color(text_color)
        colors.append(None)

        def do_gen_dir_img(args):
            img, letter = self.gen_image(bg_img=random.choice(bg_list), text_color=random.choice(colors), words=words)
            img_filename = '{:09}_{}_{:03}.{}'.format(args[0], letter, random.randrange(10, 900), ext)
            if aug:
                image = img
                if random.random() > 0.50:
                    image = add_rotate2(image)
                image = np.array(image)

                if random.random() > 0.50:
                    image = add_noise(image, min=10, max=300)
                aug_random = random.random()
                if aug_random > 0.50:
                    image = add_erode(image)
                else:
                    image = add_dilate(image)

                filename = os.path.join(out_dir, img_filename)
                cv2.imwrite(filename, image)
            else:
                img.save(os.path.join(out_dir, img_filename))

        for i in range(batch_size):
            do_gen_dir_img([i])

    def is_valid_char2(self, name, words):
        for c in name:
            if c not in words:
                return True
        return False
    def gen_image(self, bg_img=None, text_color=None, words=None):
        if words:
            wds = words
        else:
            wds = self.charset
        letter = self.get_letter(wds)
        while self.is_valid_char2((''.join(letter).strip()), self.charset):
            letter = self.get_letter(wds)
        img = self._gen_image(letter, bg_img, text_color)
        letter = ''.join(letter).strip()
        return img, letter

    def _gen_image(self, letter, bg_img=None, text_color=None):
        width, height = self.width, self.height
        strs = '%s' % ''.join(letter) + " "
        text_len = len(letter)
        if self.fonts == None:
            font = self.font
        else:
            font = random.choice(self.fonts)

        font_width, font_height = font.getsize(strs)

        letter_width = font_width / text_len + 2 # 每个字符占位宽度
        #imw = (font_width + int(letter_width) + 5)
        imw = (font_width + int(letter_width) + int(text_len*2) + 5)
        imh = (font_height + 5)

        imw = max(imw, width)
        imh = max(imh, height)

        dw = (imw - font_width) / 5
        dh = (imh - font_height) / 3

        image, isNew = self.choice_image(bg_img, (imw, imh))
        #random text_color if text_color is none
        if self.is_none(text_color):
            text_color = self.text_color()

        draw = ImageDraw.Draw(image)
        if isNew:
            for x in range(imw):
                for y in range(imh):
                    draw.point((x, y), fill=self.bg_color())
        # for t in range(len(letter)):
        #     draw.text((letter_width + (t * letter_width), dh), letter[t], font=font, fill=text_color)

        draw.text((0, 0), "  "+strs, font=font, fill=text_color)

        return image.resize((width, height), Image.BICUBIC)

    def choice_text_color(self, text_color):
        colors = []
        if self.is_none(text_color):
            colors.append(None)
            return colors
        else:
            if isinstance(text_color, list):
                colors.extend(text_color)
            elif isinstance(text_color, np.ndarray):
                colors.extend(np.squeeze(text_color))
            else:
                colors.append(text_color)
        return colors

    def choice_image(self, bg_img, bg_shape, mode="RGB", bg_color=(255, 255, 255)):
        #默认 (255, 255, 255)  # 白色背景
        image = None
        isNew = False
        if self.is_none(bg_img):
            image = Image.new(mode, bg_shape, bg_color)  # 白色背景
            isNew = True
        else:
            if isinstance(bg_img, str):
                image = Image.open(bg_img)
            elif isinstance(bg_img, Image.Image):
                image = bg_img
            elif isinstance(bg_img, np.ndarray):
                image = Image.fromarray(bg_img)
            else:
                image = Image.new(mode, bg_shape, bg_color)  # 白色背景
                isNew = True
            image = image.resize(bg_shape)
        return image, isNew

    @staticmethod
    def draw_boxes(image, boxes, line_color=(255, 0, 0)):
        draw = ImageDraw.Draw(image)
        line_width = 1
        for item in boxes:
            # x0,y1,x2,y3   #xl, yb, xr, yt
            xs = item[0]
            ys = item[1]
            xe = item[2]
            ye = item[3]
            draw.line([(xs, ys), (xs, ye), (xe, ye), (xe, ys), (xs, ys)], width=line_width, fill=line_color)
        return image

    def mark_one(self, text_color=None):
        wds = self.charset
        letter = self.get_letter(wds)

        if self.fonts == None:
            font = self.font
        else:
            font = random.choice(self.fonts)

        width, height = self.width, self.height
        strs = '%s'%''.join(letter) + " "
        text_len = len(letter)

        font_width, font_height = self.font.getsize(strs)

        letter_width = font_width / text_len  # 每个字符占位宽度
        imw = (font_width + int(letter_width) + 5)
        imh = (font_height + 5)

        dw = (imw - font_width) / 5
        dh = (imh - font_height) / 3

        imw = max(imw, width)
        imh = max(imh, height)

        image = Image.new('RGBA', (imw, imh))
        draw = ImageDraw.Draw(image)

        # random text_color if text_color is none
        if self.is_none(text_color):
            text_color = self.text_color()

        # for t in range(len(letter)):
        #     draw.text((letter_width + (t * letter_width), dh), letter[t], font=font, fill=text_color)

        draw.text((0, 0), "  " + strs, font=font, fill=text_color)
        return image, letter

    def mark_image(self, bg_img=None, bg_shape=(512,512), text_color=None):
        #bg_shape (imw, imh)
        max_size = int(bg_shape[1]/self.height)
        box_size = random.randrange(2, max_size)

        #background = Image.new('RGBA', bg_shape, self.bg_color())
        background, isNew = self.choice_image(bg_img, bg_shape, mode="RGBA", bg_color=self.bg_color())
        boxes = []

        colors = self.choice_text_color(text_color)
        if len(colors) == 0:
            colors.append(None)

        while box_size > 0:
            image, letter = self.mark_one(text_color=random.choice(colors))
            x, y = image.size
            bX, bY = background.size
            # [0:imgW,0:imgH, :]
            xOffset = random.randrange(0, (bX-x))  # 长度,横轴; 左边:0, 右边: backX-imageX
            yOffset = random.randrange(0, (bY-y))  # 高度,竖轴; 顶部:0, 底部: backY-imageY，

            #box = [yOffset, y+yOffset, xOffset, x+xOffset]    #y1:y2, x1:x2
            box = [xOffset, yOffset, (x+xOffset), (y+yOffset)] #x0,y1,x2,y3   #xl, yb, xr, yt
            if not self.is_overlap(box, boxes):
                background.paste(image, (xOffset, yOffset, x + xOffset, y + yOffset), image)  # 横，竖
                boxes.append(box)
                box_size = box_size - 1
        return background, boxes

    def gen_mark_img_dir(self, batch_size, out_dir, bg_dir=None, bg_shape=(512,512), text_color=None):
        print("gen_mark_img_dir >>> size: %d, dir: %s  "%(batch_size,out_dir))
        bg_list = self.get_img_file(bg_dir)
        if len(bg_list) == 0:
            bg_list.append(None)
        colors = self.choice_text_color(text_color)
        if len(colors) == 0:
            colors.append(None)
        for i in range(batch_size):
            background, boxes = self.mark_image(bg_img=random.choice(bg_list),
                                                bg_shape=bg_shape,
                                                text_color=random.choice(colors))

            img_filename = '{:05}_{}_{:02}.jpg'.format(i, self.boxes_str(boxes), random.randrange(10, 90))
            background.save(os.path.join(out_dir, img_filename))

    def is_overlap(self, box, boxes):
        for b in boxes:
            #相交面积 > 0
            if self.overlap_area(box, b) > 0:
                return True
        return False

    def overlap_area(self,box1, box2):
        xl1, yb1, xr1, yt1 = box1
        xl2, yb2, xr2, yt2 = box2
        xmin = max(xl1, xl2)
        ymin = max(yb1, yb2)
        xmax = min(xr1, xr2)
        ymax = min(yt1, yt2)
        width = xmax - xmin  #两右下角顶点的小x减去两左上顶点的大x
        height = ymax - ymin
        if width <= 0 or height <= 0:
            return 0
        return width * height

    def boxes_str(self, boxes):
        bb = []
        for box in boxes:
            bb.append(','.join(['%s' % (x) for x in box]))
        return '|'.join(bb)

    def gray_img(self, image):
        return image.convert('L')

    def format_gray_img(self, image):
        w,h = image.size
        format_img = np.array(image.convert('L')).flatten() / 255.0
        return format_img.reshape(w, h, 1)

    def format_rgb_img(self, image):
        w, h = image.size
        format_img = np.array(image.convert('RGB')).flatten() / 255.0
        return format_img.reshape(w, h, 3)

    #conver gray, reshape [width, height]
    def format_img(self, image):
        w,h = image.size
        format_img = np.array(image.convert('L')).flatten() / 255.0
        return format_img.reshape(w, h)


def sparse_tuple_from_label(label_list, dtype=np.int32):

    """Create a sparse representention of x.
    Args:
        label_list: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(label_list):
        indices.extend(zip([n]*len(seq), range(0,len(seq),1)))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(label_list), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def ctc_sparse_from_label(label_list, dtype=np.int32):
    indices, values, shape = sparse_tuple_from_label(label_list,dtype)
    return tf.SparseTensor(indices, values, shape)

def ctc_label_dense_to_sparse(y_true):
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(y_true, zero)
    indices = tf.where(where)
    values = tf.gather_nd(y_true, indices)
    return tf.SparseTensor(indices, values, y_true.shape)

def ctc_batch_cost(y_true, y_pred, input_length):
    """Runs CTC loss algorithm on each batch element.
    # Arguments
        y_true: tensor `(samples, max_string_length)`
                containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)`
                containing the prediction, or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence
                length for each batch item in `y_pred`.
                [time_step] * batch_size | np.zeros([batch_size, 1]);input_length[i] = time_step(img_w//pool_size)
    # Returns
        Tensor with shape (samples,1) containing the
            CTC loss of each element.
    """
    input_length = tf.to_int32(tf.squeeze(input_length))
    sparse_labels = tf.to_int32(ctc_label_dense_to_sparse(y_true))

    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)

    return tf.expand_dims(ctc.ctc_loss(inputs=y_pred,
                                       labels=sparse_labels,
                                       sequence_length=input_length), 1)

def cnn_ctc_batch_cost(y_true, y_pred, input_length):
    return ctc_batch_cost(y_true, y_pred, input_length)

def rnn_ctc_batch_cost(y_true, y_pred, input_length):
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return ctc_batch_cost(y_true, y_pred, input_length)


def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = spars_tensor[1][m]
        decoded.append(str)
    return decoded

def print_net(net):
    print('%-35s | %-20s | %-35s'%(net.op.name, net.op.type, net.shape))

def print_net_line():
    print('%-35s | %-20s | %-35s'%(("-"*35), ("-"*20), ("-"*35)))

def get_logger(log_dir):
    logger = logging.getLogger("output")
    logger.setLevel(logging.INFO)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    output_path = os.path.join(log_dir, 'output.log')
    fh = logging.FileHandler(output_path, mode='a')
    fh.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(asctime)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s,%(levelname)s,%(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def calculate_distance(y_true_str, y_pred_str):
    total_distance = 0.0
    total_distance_ed = 0.0
    for i in range(len(y_true_str)):
        s1 = y_true_str[i]
        s2 = y_pred_str[i]
        edit_dist = levenshtein_distance(s1, s2)
        # print(edit_dist)
        total_distance += float(edit_dist)
        total_distance_ed += float(edit_dist) / len(s1)
    return total_distance, total_distance_ed, (total_distance / len(y_true_str)), (total_distance_ed / len(y_true_str))


class Progbar(object):
    def __init__(self, steps, epochs, width=30):
        self.steps = steps
        self.epochs = epochs
        self.width = width
        self.start = time.time()
        self.total_time = 0.00

    def format_time(self, eta):
        if eta > 3600:
            eta_format = ('%d:%02d:%02d' %
                          (eta // 3600, (eta % 3600) // 60, eta % 60))
        elif eta > 60:
            eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
            eta_format = '%.4fs' % eta
        return eta_format

    def log_begin(self, step):
        print("Epoch %d/%d" % (step, self.steps))

    def log_loss2(self, epoch, loss, msg=''):
        eta = time.time() - self.start
        self.total_time += eta

        eta = self.format_time(eta)
        t_eta = self.format_time(self.total_time)

        epochs = self.epochs
        percent = int(float(epoch) * 100 / float(epochs)) + 1
        width = int(self.width)

        # print('percent:%10.8s%s'%(str(percent),'%'), end='\r')
        pp = int(percent / 3.3)
        curprog = int(float(pp) * width / float(width)) + 1
        if (width - curprog) > 1:
            prog = (curprog * '>') + ((width - curprog) * '.')
        else:
            prog = ((width) * '=')
        sys.stdout.write('\r')
        sys.stdout.write("%d/%d | %3s%% [%30s] - %s/%s - loss: %.4f - %s" % (
            epoch, epochs, percent, prog, eta, t_eta, loss, msg))
        sys.stdout.flush()
        self.start = time.time()

    def log_loss(self, epoch, loss, acc=0):
        self.log_loss2(epoch, loss, ("acc: %.2f"%acc))

    def log_end(self):
        sys.stdout.write('\n')
        self.start = time.time()
        self.total_time = 0.00



#添加旋转
def add_rotate2(image, angle=5):
    _angle = random.randrange(0-angle, angle)
    return image.rotate(_angle, expand=1)
#适当腐蚀
def add_erode(img_arr,min=1, max=4):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(random.randrange(min,max), random.randrange(min,max)))
    img_arr = cv2.erode(img_arr, kernel)
    return img_arr

#膨胀
def add_dilate(img_arr,min=1, max=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(random.randrange(min,max), random.randrange(min,max)))
    img_arr = cv2.dilate(img_arr, kernel)
    return img_arr

#添加点噪声
def add_noise(img_arr, min=300, max=500):
    for i in range(random.randrange(min, max)):
        temp_x = np.random.randint(0, img_arr.shape[0])
        temp_y = np.random.randint(0, img_arr.shape[1])
        img_arr[temp_x][temp_y] = 255
    return img_arr

