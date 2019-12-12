import argparse


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-name", type=str, required=True)
    parser.add_argument("--source-col", type=str, default='user_id')
    parser.add_argument("--target-col", type=str, default='product_id')
    parser.add_argument("--weight-col", type=str, default='rating')
    parser.add_argument("--text-col", type=str, default='review')
    parser.add_argument("--alpha1", type=int)
    parser.add_argument("--alpha2", type=int)
    parser.add_argument("--beta1", type=int)
    parser.add_argument("--beta2", type=int)
    parser.add_argument("--gamma1", type=int)
    parser.add_argument("--gamma2", type=int)
    parser.add_argument("--gamma3", type=int)
    parser.add_argument("--init-data-name", type=str, default=None)

    return parser.parse_args()
