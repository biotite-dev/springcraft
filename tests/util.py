from os.path import dirname, join, realpath


def data_dir():
    return join(dirname(realpath(__file__)), "data")
