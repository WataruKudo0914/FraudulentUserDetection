import argparse
# from distutils.util import strtobool


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument(
        '--exp4_rate_list', '-r',
        type=lambda x: map(float, x.split(',')),
        default=[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15,
                 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument(
        '--experiments',
        type=lambda x: x.split(','),
        default=['1', '2', '3', '4']
    )
    parser.add_argument(
        '--exp4_select',
        type=lambda x: x.split(','),
        default=['train', 'eval']
    )
    return parser
