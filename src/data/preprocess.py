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
    # amazon_home
    amazon_extra.main('amazon_home')
    # amazon_musicc
    amazon_extra.main('amazon_music')
    # amazon_app
    amazon_extra.main('amazon_app')


if __name__ == '__main__':
    main()
