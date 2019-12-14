import argparse
from distutils.util import strtobool


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_name', type=str)
    return parser
