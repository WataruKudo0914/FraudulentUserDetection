"""
各種実験を行う，設定は以下の3つ
1. 10分割交差検証
    学習:評価 = 9:1 で10分割交差検証
2. ロバストネス検証 ① (学習データ比率=0.03)
3. ロバストネス検証 ② (学習データの比率を変えて検証）
    学習データ: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    各レートにつき30回ランダム実験
4. inductive setting (sdgcn, sgcnのみ)
"""
from src.arg_parser import get_parser
from src.models import sdgcn, sgcn, rev2, rgcn
from pathlib import Path
import json

parser = get_parser()
args = parser.parse_args()

"""
model設定
"""
if args.model_name == 'sdgcn':
    model = sdgcn
elif args.model_name == 'sgcn':
    model = sgcn
elif args.model_name == 'rgcn':
    model = rgcn
elif args.model_name == 'rev2':
    model = rev2
else:
    raise ValueError

result_dir = Path("./data/results/") / args.model_name / args.data_name
result_dir.mkdir(parents=True, exist_ok=True)

"""
実験1
"""
average_auc = model.ten_fold_cv(args.data_name)
with open(result_dir / 'exp1.json', 'w') as f:
    json.dump({'auc': average_auc}, f)

"""
実験2
"""
exp2_result_df = model.robustness_experiments(
    args.data_name,
    training_rates_list=[0.03]
)
exp2_result_df.to_csv(result_dir / 'exp2.csv')

"""
実験3
"""
exp3_result_df = model.robustness_experiments(
    args.data_name,
    training_rates_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)
exp3_result_df.to_csv(result_dir / 'exp3.csv')
"""
実験4
"""
exp4_result_df = model.inductive_learning_eval(
    args.data_name,
    iter_num=30
)
exp4_result_df.to_csv(result_dir / 'exp4.csv')
