import argparse
import os, errno
import random
import re
import string

from tqdm import tqdm
from text.string_generator import (
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_from_new,
    load_store_data,
    get_font_file
)
from text.data_generator import FakeTextDataGenerator
from text.gen_letter import GenLetter
from multiprocessing import Pool

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
        default="resource/bgimg",
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
        "-r",
        "--random",
        action="store_true",
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        default=False
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        help="Define the height of the produced images",
        default=30,
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
        default="png",
    )
    parser.add_argument(
        "-k",
        "--skew_angle",
        type=int,
        nargs="?",
        help="Define skewing angle of the generated text. In positive degrees",
        default=0,
    )
    parser.add_argument(
        "-rk",
        "--random_skew",
        action="store_true",
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
        default=False,
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
        "-bl",
        "--blur",
        type=int,
        nargs="?",
        help="Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius",
        default=0,
    )
    parser.add_argument(
        "-rbl",
        "--random_blur",
        action="store_true",
        help="When set, the blur radius will be randomized between 0 and -bl.",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--background",
        type=int,
        nargs="?",
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures",
        default=0,
    )
    parser.add_argument(
        "-hw",
        "--handwritten",
        action="store_true",
        help="Define if the data will be \"handwritten\" by an RNN",
    )
    parser.add_argument(
        "-na",
        "--name_format",
        type=int,
        help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings",
        default=0,
    )
    parser.add_argument(
        "-d",
        "--distorsion",
        type=int,
        nargs="?",
        help="Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
        default=0
    )
    parser.add_argument(
        "-do",
        "--distorsion_orientation",
        type=int,
        nargs="?",
        help="Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
        default=0
    )
    parser.add_argument(
        "-redata",
        "--store_data",
        nargs="?",
        type=str,
        help="load wiki or news data",
        default="",
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

    return parser.parse_args()

def load_dict(lang):
    """
        Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(os.path.join('dicts', lang + '.txt'), 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = d.readlines()
    return lang_dict

def load_fonts(lang):
    """
        Load all fonts in the fonts directories
    """

    if lang == 'cn':
        return [os.path.join('fonts/cn', font) for font in os.listdir('fonts/cn')]
    else:
        return [os.path.join('fonts/latin', font) for font in os.listdir('fonts/latin')]


def main():
    """
        Description: Main function
    """

    # Argument parsing
    args = parse_arguments()
    print(args)

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise



    # Creating synthetic sentences (or word)

    IMAGE_WIDTH = args.new_width
    IMAGE_HEIGHT = args.new_height
    bg_dir = args.bg_dir
    language = args.language
    output_dir = args.output_dir
    min_length = args.min_length
    max_length = args.max_length
    count = args.count
    extension = args.extension
    font_dir = args.font_dir

    print("gen_img_dir >>> size: %d, dir: %s" % (count, output_dir))

    # Create font (path) list
    fonts = get_font_file(font_dir)
    print("fonts: ")
    print(fonts)


    if args.use_wikipedia:
        if args.store_data != '':
            words = load_store_data("wiki", args.store_data)
        else:
            words = create_strings_from_wikipedia(args.length, count * 2, language, max_length=max_length)
    elif args.input_file != '':
        words = create_strings_from_file(args.input_file,max_length)
    else:
        if args.store_data != '':
            words = load_store_data("news", args.store_data)
        else:
            words = create_strings_from_new(args.length, count * 2, language, max_length=max_length)

    gen_letter = GenLetter(min_length, max_length)
    #dataGenerator = FakeTextDataGenerator()

    strings = []
    for i in range(count):
        char = ''.join(gen_letter.get_letter(words))
        while gen_letter.is_valid_char(char):
            char = ''.join(gen_letter.get_letter(words))
        strings.append(char)
    string_count = len(strings)
    g_fonts = []
    for i in range(count):
        g_fonts.append(random.choice(fonts))

    #print(">>>>>>>>strings: \n", strings)

    p = Pool(args.thread_count)
    for _ in tqdm(p.imap_unordered(
            FakeTextDataGenerator.generate_from_tuple,
            zip(
                [i for i in range(0, string_count)],
                strings,
                g_fonts,
                [args.output_dir] * string_count,
                [args.format] * string_count,
                [args.extension] * string_count,
                [args.skew_angle] * string_count,
                [args.random_skew] * string_count,
                [args.blur] * string_count,
                [args.random_blur] * string_count,
                [args.background] * string_count,
                [args.distorsion] * string_count,
                [args.distorsion_orientation] * string_count,
                [args.handwritten] * string_count,
                [args.name_format] * string_count,
                [args.new_width] * string_count,
                [args.new_height] * string_count
            )
    ), total=args.count):
        pass
    p.terminate()


#-redata datasets/data
#python gen_run2.py -b 3  -w 2 -f 28 -na 1 -c 500 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out -redata datasets/data
if __name__ == '__main__':
    main()
