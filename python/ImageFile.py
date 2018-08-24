import tensorflow as tf
import numpy as np
import os,string,random
import cv2
from Config import Config,sparse_tuple_from_label

def random_rotation(img: tf.Tensor, max_rotation: float = 0.1, crop: bool = True) -> tf.Tensor:  # adapted from SeguinBe
    """
    Rotates an image with a random angle
    see https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae
    :param img: Tensor
    :param max_rotation: maximum angle to rotate (radians)
    :param crop: boolean to crop or not the image after rotation
    :return:
    """
    with tf.name_scope('RandomRotation'):
        rotation = tf.random_uniform([], -max_rotation, max_rotation, name='pick_random_angle')
        rotated_image = tf.contrib.image.rotate(img, rotation, interpolation='BILINEAR')
        if crop:
            rotation = tf.abs(rotation)
            original_shape = tf.shape(rotated_image)[:2]
            h, w = original_shape[0], original_shape[1]
            old_l, old_s = tf.cond(h > w, lambda: [h, w], lambda: [w, h])
            old_l, old_s = tf.cast(old_l, tf.float32), tf.cast(old_s, tf.float32)
            new_l = (old_l * tf.cos(rotation) - old_s * tf.sin(rotation)) / tf.cos(2 * rotation)
            new_s = (old_s - tf.sin(rotation) * new_l) / tf.cos(rotation)
            new_h, new_w = tf.cond(h > w, lambda: [new_l, new_s], lambda: [new_s, new_l])
            new_h, new_w = tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)
            bb_begin = tf.cast(tf.ceil((h - new_h) / 2), tf.int32), tf.cast(tf.ceil((w - new_w) / 2), tf.int32)
            # Test sliced
            rotated_image_crop = tf.cond(
                tf.logical_and(bb_begin[0] < h - bb_begin[0], bb_begin[1] < w - bb_begin[1]),
                true_fn=lambda: rotated_image[bb_begin[0]:h - bb_begin[0], bb_begin[1]:w - bb_begin[1], :],
                false_fn=lambda: img,
                name='check_slices_indices'
            )
            # rotated_image_crop = rotated_image[bb_begin[0]:h - bb_begin[0], bb_begin[1]:w - bb_begin[1], :]

            # If crop removes the entire image, keep the original image
            rotated_image = tf.cond(tf.equal(tf.size(rotated_image_crop), 0),
                                    true_fn=lambda: img,
                                    false_fn=lambda: rotated_image_crop,
                                    name='check_size_crop')

        return rotated_image


def random_padding(image: tf.Tensor, max_pad_w: int = 5, max_pad_h: int = 10) -> tf.Tensor:
    """
    Given an image will pad its border adding a random number of rows and columns
    :param image: image to pad
    :param max_pad_w: maximum padding in width
    :param max_pad_h: maximum padding in height
    :return: a padded image
    """
    # TODO specify image shape in doc

    w_pad = list(np.random.randint(0, max_pad_w, size=[2]))
    h_pad = list(np.random.randint(0, max_pad_h, size=[2]))
    paddings = [h_pad, w_pad, [0, 0]]

    return tf.pad(image, paddings, mode='REFLECT', name='random_padding')


def augment_data(image: tf.Tensor) -> tf.Tensor:
    """
    Data augmentation on an image (padding, brightness, contrast, rotation)
    :param image: Tensor
    :return: Tensor
    """
    with tf.name_scope('DataAugmentation'):
        # Random padding
        image = random_padding(image)

        # TODO : add random scaling
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = random_rotation(image, 0.05, crop=True)

        if image.shape[-1] >= 3:
            image = tf.image.random_hue(image, 0.2)
            image = tf.image.random_saturation(image, 0.5, 1.5)

        return image

def gen_crop_bg(im, size, out_dir, width, height):
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

def test_gen_crop_bg():
    import PIL.Image as Image
    bgimgs = os.listdir("./bgimg2")
    x, y = 360, 42
    for bg in bgimgs:
        gen_crop_bg(Image.open("./bgimg2/" + bg), 1000, "./bgimg/images/", x, y)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_image_files(image_dir,check=False):
    count = 0
    filenames = []
    labels = []
    for f in os.listdir(image_dir):
        try:
            if not f.endswith(('.gif', '.jpg', '.png')):
                continue
            fp = os.path.join(image_dir, f)
            if not os.path.isabs(fp):
                fp = os.path.abspath(fp)
            if not os.path.exists(fp):
                continue
            if check:
                #Image.open(fn)
                cv2.imread(fp)
            image_name = f.split('_')[1]
            filenames.append(fp)
            labels.append(image_name)
            count += 1
        except Exception as e:
            print("fn:%s,error: %s", fp, e)
            os.remove(fp)
    return filenames, labels
def check_image_files(image_dir, cv2=True):
    rmlist = []
    for f in os.listdir(image_dir):
        try:
            if not f.endswith(('.gif', '.jpg', '.png')):
                os.remove(os.path.join(image_dir, f))
                continue
            fp = os.path.join(image_dir, f)
            if not os.path.exists(fp):
                continue
            if cv2:
                cv2.imread(fp)
            else:
                import PIL.Image as Image
                Image.open(fp)
        except Exception as e:
            print("fn:%s,error: %s", fp, e)
            rmlist.append(fp)
            os.remove(fp)
    return rmlist

def sparse_tuple_from_label2(labels_ids):
    i, v, s = sparse_tuple_from_label(labels_ids)
    return tf.SparseTensor(i, v, s)

def encoding_str(labels_str_list):
    return [str(labels_str, encoding='utf-8') for labels_str in labels_str_list]

class ImageFileIterator:
    """
    data_augmentation Data augmentation on an image (padding, brightness, contrast, rotation)
    """
    def __init__(self, image_dir, config:Config,check=False):
        self.image_dir = image_dir
        self.config = config
        filenames, labels = get_image_files(image_dir,check)
        self.filenames = filenames
        self.labels = labels
        self.labels_ids = None #[config.text_to_ids(code) for code in labels]

    def input_pipeline3(self, batch_size, target_height, target_width, input_channels, num_epochs=None,
                       augmentation=False, num_parallel=4):
        filenames = self.filenames
        labels_str = self.labels
        def _parse_function(filename, labels_str):
            image_content = tf.read_file(filename, name='filename_reader')
            image = tf.cond(
                tf.image.is_jpeg(image_content),
                lambda: tf.image.decode_jpeg(image_content, channels=input_channels, name='image_decoding_op',
                                             try_recover_truncated=True),
                lambda: tf.image.decode_png(image_content, channels=input_channels,
                                            name='image_decoding_op'))

            # Data augmentation
            if augmentation:
                image = augment_data(image)

            #new_height, new_width
            #image = tf.image.resize_images(image, size=output_shape)
            #image = image / 255.0
            image = tf.image.resize_image_with_crop_or_pad(
                image,
                target_height,
                target_width
            )
            img_width = tf.shape(image)[1]

            features = {'labels_str': labels_str, 'images': image, "images_widths": img_width,  'filename':filename}
            return features, features.get("labels_str")

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels_str))

        dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel).shuffle(batch_size)
        # -- Shuffle, repeat, and batch features
        dataset = dataset.batch(batch_size).repeat(num_epochs).prefetch(4)
        iterator = dataset.make_one_shot_iterator()
        features, labels_sparse = iterator.get_next()
        return features, labels_sparse

    def input_pipeline(self, batch_size, target_height, target_width, input_channels, num_epochs=None, augmentation=False,num_parallel=4):
        filenames = self.filenames
        labels_str = self.labels
        labels_ids = self.labels_ids
        labels_sparse = sparse_tuple_from_label2(labels_ids)

        def _parse_function(filename, labels_sparse, labels_str):
            image_content = tf.read_file(filename, name='filename_reader')
            image = tf.cond(
                tf.image.is_jpeg(image_content),
                lambda: tf.image.decode_jpeg(image_content, channels=input_channels, name='image_decoding_op',
                                             try_recover_truncated=True),
                lambda: tf.image.decode_png(image_content, channels=input_channels,
                                            name='image_decoding_op'))

            # Data augmentation
            if augmentation:
                image = augment_data(image)

            #new_height, new_width
            #image = tf.image.resize_images(image, size=output_shape)
            #image = image / 255.0
            image = tf.image.resize_image_with_crop_or_pad(
                image,
                target_height,
                target_width
            )
            img_width = tf.shape(image)[1]

            features = {'labels_str': labels_str, 'images': image, "images_widths": img_width, 'labels_sparse': labels_sparse, 'filename':filename}
            return features, labels_sparse

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels_sparse, labels_str))

        dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel).shuffle(batch_size)
        # -- Shuffle, repeat, and batch features
        dataset = dataset.batch(batch_size).repeat(num_epochs).prefetch(4)
        iterator = dataset.make_one_shot_iterator()
        features, labels_sparse = iterator.get_next()
        return features, labels_sparse

    def input_pipeline2(self, batch_size, input_channels, target_height, target_width, num_epochs=None, augmentation=False):
        filenames = self.filenames
        labels_str = self.labels
        labels_ids = self.labels_ids
        labels_sparse = sparse_tuple_from_label2(labels_ids)

        images_tensor = tf.convert_to_tensor(filenames, dtype=tf.string)

        input_queue = tf.train.slice_input_producer([images_tensor, labels_str, labels_sparse], num_epochs=num_epochs)

        labels = input_queue[1]
        labels_sp = input_queue[2]
        images_content = tf.read_file(input_queue[0])

        image = tf.cond(
            tf.image.is_jpeg(images_content),
            lambda: tf.image.decode_jpeg(images_content, channels=input_channels, name='image_decoding_op',
                                         try_recover_truncated=True),
            lambda: tf.image.decode_png(images_content, channels=input_channels,
                                        name='image_decoding_op'))

        images = tf.image.convert_image_dtype(image, tf.float32)
        if augmentation:
            images = augment_data(images)
        ##new_height, new_width
        new_size = tf.constant([target_height, target_width], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch, labels_sp = tf.train.shuffle_batch([images, labels, labels_sp],
                                                          batch_size=batch_size,
                                                          capacity=50000,
                                                          min_after_dequeue=10000)
        return image_batch, label_batch, labels_sp


if __name__ == '__main__':
    import PIL.Image as Image

    bgimgs = os.listdir("./bgimg2")
    x, y = 360, 42
    for bg in bgimgs:
        gen_crop_bg(Image.open("./bgimg2/" + bg), 1000, "./bgimg/images/", x, y)
