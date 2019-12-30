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
# import comet_ml in the top of your file
from comet_ml import Experiment
import json
from pathlib import Path
from src.arg_parser import get_parser
# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="K96HV1ZN57Ip54lRy1GNaOpBN",
                        project_name="fraudulentuserdetection",
                        workspace="watarukudo0914")
# from src.models import sdgcn, sgcn, rev2, rgcn

parser = get_parser()
args = parser.parse_args()
experiment.log_others(
    {
        'model_name': args.model_name,
        'data_name': args.data_name,
    }
)
"""
model設定
"""
if args.model_name == 'sdgcn':
    from src.models import sdgcn
    model = sdgcn
elif args.model_name == 'sgcn':
    from src.models import sgcn
    model = sgcn
elif args.model_name == 'rgcn':
    from src.models import rgcn
    model = rgcn
elif args.model_name == 'rev2':
    from src.models import rev2
    model = rev2
elif args.model_name == 'side':
    from src.models import side
    model = side
else:
    raise ValueError

result_dir = Path("./data/results/") / args.model_name / args.data_name
result_dir.mkdir(parents=True, exist_ok=True)

"""
実験1
"""
if '1' in args.experiments:
    result_dict = model.ten_fold_cv(experiment, args.data_name)
    with open(result_dir / 'exp1.json', 'w') as f:
        json.dump(result_dict, f)

"""
実験2
"""
if '2' in args.experiments:
    exp2_result_df = model.robustness_experiments(
        experiment,
        args.data_name,
        training_rates_list=[0.03]
    )
    exp2_result_df.to_csv(result_dir / 'exp2.csv')

"""
実験3
"""
if '3' in args.experiments:
    exp3_result_df = model.robustness_experiments(
        experiment,
        args.data_name,
        training_rates_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    exp3_result_df.to_csv(result_dir / 'exp3.csv')

"""
実験4
"""
if '4' in args.experiments:
    exp4_result_df = model.inductive_learning_eval(
        args.exp4_select,
        experiment,
        args.data_name,
        rate_list=args.exp4_rate_list,
        iter_num=30
    )
    exp4_result_df.to_csv(result_dir / 'exp4.csv')

# comet-mlに保存
experiment.log_asset_folder(result_dir)
