# Fraudulent User Detection

## データ前処理
### 生の評価ネットワーク -> signed network & ラベル & ノード特徴量)
コード:  
``` sh scripts/load_network_and_gt.sh ```

出力：  
- `data/raw/<data_name>/network.csv`  
    生の評価ネットワーク(Amazonなら5段階評価, 等)  
- `data/raw/<data_name>/gt.csv`  
    生の評価ネットワークノードに対する不正ユーザラベル
- `data/processed/<data_name>/network.csv`  
    signed network (edge = {-1, 1})
- `data/processed/<data_name>/network.csv`  
    signed networkのノードに対応する不正ユーザラベル

実験対象データリスト

|data_name| 説明 |  
| --- | --- |
|alpha|ビットコイン取引のuser2user評価|
|otc|ビットコイン取引のuser2user評価|
|epinions|user2user|
|amazon|user2product|
|amazon_home|【新データセット】|
|amazon_music|【新データセット】|
|amazon_app|【新データセット】|




### 各ネットワークから初期のエッジを取り出す(inductive setting用)
コード:  
``` sh scripts/make_early_networks.sh ```  

出力:  
- `data/processed/early/<data_name>_<rate>/*`

## 学習 & 検証

コード:  
``` sh scripts/experiment_all.sh <data_name> <model_name> ```  
出力
- `data/results/<model_name>/<data_mame>/*`  
(例)　`sh scripts/experiment_all.sh amazon sdgcn`
## フォルダ構成
```
.
├── README.md
├── configs         # ハイパーパラメータなど
├── data            # データの保存場所
│   ├── processed   # 前処理後のデータ
│   └── raw         # 生の評価ネットワーク
├── scripts         # スクリプト
└── src             # ソースコード
    ├── data        # データ処理
    └── models      # モデル
```