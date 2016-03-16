import errno
import os
from itertools import tee


def make_sure_path_exists(path):
    """
    Ensures that the path specified exists, and makes it if necessary
    :type path: basestring
    :param path: path to file.
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_pairwise_list(iterable):
    """
    Produces a pairwise list over an iterable
    e.g. [x,y,z,w] - > [(x,y),(y,z), (z,w)]
    :param iterable: iterable.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_number_lines(file):
    """
    Gets the number of lines in the specified file
    :param file: File to read.
    :return: integer number of lines in file.
    """
    num_lines = sum(1 for line in open(file, 'r'))
    return num_lines
