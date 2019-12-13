import easydict
from pathlib import Path


def get_args(data_name, rate=None):
    if rate is None:
        input_path = Path('./data/processed') / data_name
        output_path = Path('./data/intermediate') / data_name
    else:
        input_path = Path('./data/processed/early') / f'{data_name}_{rate}'
        output_path = Path('./data/intermediate') / f'{data_name}_{rate}'
    if not output_path.exists():
        output_path.mkdir()
    args = easydict.EasyDict({
        "edge_path": input_path / 'network.csv',
        "features_path": input_path / 'node_feature.csv',
        "nodes_path": input_path / 'gt.csv',
        "embedding_path": output_path / 'sgcn_embedding.pkl',
        "regression_weights_path": output_path / 'sgcn_weights.pkl',
        # '/home2/kudo/SGCN/output/inductive/{0}_model'.format(data_name), # or None
        "inductive_model_path": None,
        "log_path": './data/logs',
        "epochs": 150,
        "test_size": 0.33,
        "reduction_iterations": 128,
        "reduction_dimensions": 30,
        "seed": 42,
        "lamb": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 10e-4,
        # "layers": [64, 32,16,8],
        "layers": [32, 16, ],
        "spectral_features": False,
        "general_features": True,
        "sample_num": None,
        "class_weights": False,
        "node_under_sampling": False,
        "hidden_residual": False,
        "eval_freq": 1,
        "subgraph_training": False,
        "l1_lambda": 0.0,
    })
    return args
