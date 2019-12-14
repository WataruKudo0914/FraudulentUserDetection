"""
各種実験を行う，設定は以下の3つ
1. 10分割交差検証
    学習:評価 = 9:1 で10分割交差検証
2. ロバストネス検証 ① (学習データ比率=0.03)
3. ロバストネス検証 ② (学習データの比率を変えて検証）
    学習データ: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    各レートにつき30回ランダム実験
"""
from src.arg_parser import get_parser
from src.models.sdgcn import main

parser = get_parser()
args = parser.parse_args()
main.ten_fold_cv(args.data_name)
pass
