# Fraudulent User Detection

## データ前処理(生の評価ネットワーク -> signed network & ラベル & ノード特徴量)
``` python -m src.data.preprocess.main ```

## フォルダ構成
```
.
├── README.md
├── configs         # ハイパーパラメータなど
├── data            # データの保存場所
│   ├── processed   # 前処理後のデータ
│   └── raw         # 生の評価ネットワーク(へのリンク)
├── reports         # 実験結果
│   └── figure
└── src             # ソースコード
    ├── data        # データ処理
    └── models      # モデル
```