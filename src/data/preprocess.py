"""
生の評価データ -> (Signed Network, 教師データ)
"""

from src.data import alpha_otc, amazon, epinions, amazon_extra


def main():
    # alpha
    alpha_otc.main('alpha')
    # otc
    alpha_otc.main('otc')
    # amazon
    amazon.main()
    # epinions
    epinions.main()
    # amazon_electronics
    amazon_extra.main('amazon_electronics')
    # amazon_sports
    amazon_extra.main('amazon_sports')
    pass


if __name__ == '__main__':
    main()
