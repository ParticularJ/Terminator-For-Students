import argparse

FLAGS = None


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('', default='')
    return args.parse_args()



if __name__ == '__main__':
    FLAGS = parser()