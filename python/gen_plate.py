import time,argparse
import random,string
import numpy as np
import itertools

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

    x = []
    for i in range(1990, 2020):
        x.append(i)
    vv = ["%s" % v for v in x]
    def get_plate():
        return random.choice(["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"])

    def get_y():
        return random.choice(vv)
    f = open('plate_list.txt',mode='w')
    f.write("中华人民共和国机动车行驶证")
    f.write('\n')
    f.write("中华人民共和国机动车驾驶证")
    f.write('\n')
    f.write("中华人民共和国机动车行驶证")
    f.write('\n')
    f.write("中华人民共和国机动车驾驶证")
    f.write('\n')
    f.write("使用性质 非运营")
    f.write('\n')
    f.write("车辆类型 小型轿车")
    f.write('\n')
    f.write("使用性质 非运营")
    f.write('\n')
    f.write("车辆类型 小型轿车")
    f.write('\n')
    f.write("车辆类型 大型货车")
    f.write('\n')
    f.write("车辆类型 中型客车")
    f.write('\n')
    f.write("准驾车型 C1")
    f.write('\n')
    f.write("准驾车型 C2")
    f.write('\n')
    f.write("准驾车型 C3")
    f.write('\n')
    f.write("准驾车型 A1")
    f.write('\n')
    f.write("准驾车型 A2")
    f.write('\n')
    f.write("准驾车型 A3")
    f.write('\n')
    f.write("准驾车型 B1")
    f.write('\n')
    f.write("准驾车型 B2")
    f.write('\n')
    f.write("准驾车型 B3")
    f.write('\n')
    for i in range(args.count):
        f.write('号牌号码 %s %s%s%s'%(get_plate(), random.choice(string.ascii_uppercase),  ''.join(random.sample(string.digits,k=4)), random.choice(string.ascii_uppercase+string.digits)))
        f.write('\n')
        f.write('初次领证日期 %s-%02d-%02d'%(get_y(), random.randrange(1,13), random.randrange(1,31)))
        f.write('\n')
        f.write('注册日期 %s-%02d-%02d'%(get_y(), random.randrange(1,13), random.randrange(1,31)))
        f.write('\n')
        f.write('发证日期 %s-%02d-%02d'%(get_y(), random.randrange(1,13), random.randrange(1,31)))
        f.write('\n')
        f.write('车辆识别代号 %s%s%s%s'%(random.choice(string.ascii_uppercase) , ''.join(random.sample(string.digits,k=4)), random.choice(string.ascii_uppercase) , ''.join(random.sample(string.digits,k=5))))
        f.write('\n')
        f.write('发动机号码 %s%s%s%s'%(random.choice(string.ascii_uppercase) , ''.join(random.sample(string.digits,k=2)), random.choice(string.ascii_uppercase) , ''.join(random.sample(string.digits,k=4))))
        f.write('\n')
    f.write("中华人民共和国机动车行驶证")
    f.write('\n')
    f.write("中华人民共和国机动车驾驶证")
    f.write('\n')
    f.write("中华人民共和国机动车行驶证")
    f.write('\n')
    f.write("中华人民共和国机动车驾驶证")
    f.write('\n')
    f.write("使用性质 非运营")
    f.write('\n')
    f.write("车辆类型 小型轿车")
    f.write('\n')
    f.write("使用性质 非运营")
    f.write('\n')
    f.write("车辆类型 小型轿车")
    f.write('\n')
    f.write("车辆类型 大型货车")
    f.write('\n')
    f.write("车辆类型 中型客车")
    f.write('\n')
    f.write("准驾车型 C1")
    f.write('\n')
    f.write("准驾车型 C2")
    f.write('\n')
    f.write("准驾车型 C3")
    f.write('\n')
    f.write("准驾车型 A1")
    f.write('\n')
    f.write("准驾车型 A2")
    f.write('\n')
    f.write("准驾车型 A3")
    f.write('\n')
    f.write("准驾车型 B1")
    f.write('\n')
    f.write("准驾车型 B2")
    f.write('\n')
    f.write("准驾车型 B3")
    f.write('\n')
