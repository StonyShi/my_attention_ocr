from text.string_generator import load_store_data
from text.gen_letter import read_dict



if __name__ == '__main__':

    c = read_dict()
    # print(c.keys())
    f = open("my_char.txt", 'w')
    for key in c.keys():
        f.write("%s\n"%key)
