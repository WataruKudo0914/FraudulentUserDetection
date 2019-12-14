"""
生の評価データ -> (Signed Network, 教師データ)
"""

from src.data import alpha_otc, amazon, epinions, amazon_extra


def main():
    """
    生の評価ネットワークデータ(./data/raw/)
    -> signed network & labels & node features (./data/processed/<data_name>/)
    """
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
    # amazon_baby
    amazon_extra.main('amazon_baby')
    # amazon_beauty
    amazon_extra.main('amazon_beauty')
    


if __name__ == '__main__':
    main()
