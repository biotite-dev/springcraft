from os.path import join, dirname, realpath


def data_dir():
    return join(dirname(realpath(__file__)), "data")
