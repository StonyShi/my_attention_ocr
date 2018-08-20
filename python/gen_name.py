import os,random,argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate text words')
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="write count words",
        default=50000
    )
    return parser.parse_args()


if __name__ == '__main__':

    # Argument parsing
    args = parse_arguments()
    print(args)


    with open("resource/xing.txt", 'r', encoding="utf8") as f:
        xing = [l.strip() for l in f.readlines()]


    with open("resource/ming.txt", 'r', encoding="utf8") as f:
        ming = [l.strip() for l in f.readlines()]

    with open("resource/name.txt", 'r', encoding="utf8") as f:
        names = [l.strip() for l in f.readlines()]

    f = open('name_list.txt', mode='w')
    for i in range(args.count):
        f.write('姓名 %s%s\n'%(random.choice(xing),random.choice(ming)))
        f.write('所有人 %s%s\n'%(random.choice(xing), random.choice(ming)))

    for name in names:
        f.write('姓名 %s\n' % (name))
        f.write('所有人 %s\n' % (name))