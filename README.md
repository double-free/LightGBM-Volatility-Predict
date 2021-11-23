# LightGBM Volatility Predict

This is a toy project to learn LightGBM. The problem (and data) is from a Kaggle competition: [Optiver Realized Volatility Prediction](https://www.kaggle.com/c/optiver-realized-volatility-prediction). The RMSPE is around 0.19.

This note of mine has more details: [LightGBM 实战：波动率预测(2)](https://www.jianshu.com/p/80ef7f71950e).

## How to use
There is assumption on the relative path of this project (named as `lgbm_vol`):

```
yourDir
|-- LightGBM.ipynb
|-- data
`-- lgbm_vol

```

The notebook `LightGBM.ipynb` imports `lgbm_vol` and calls its functions:

```python
from lgbm_vol import feature, lgbm_train
import pandas as pd
import os.path

all_stocks = feature.get_all_stock_ids('./data/book_train.parquet')

# single stock feature
stock_ids = all_stocks
window = 100
override = True
cached_file = feature.get_cache_file_name(stock_ids, window)
if os.path.isfile(cached_file) and override == False:
    print("load from cache file: " + cached_file)
    train_data = pd.read_csv(cached_file, index_col=0)
else:
    train_data = feature.get_stock_features(stock_ids, window, "train")
    print("persist features to file: " + cached_file)
    train_data.to_csv(cached_file)

# selected features for stock groups
selected_features = []
selected_features.extend([col for col in train_data.columns if 'trade_volume_mean' in col])
selected_features.extend([col for col in train_data.columns if 'last' in col])
selected_features.extend([col for col in train_data.columns if 'vwap11_realized_volatility' in col])
# selected_features.extend([col for col in train_data.columns if 'gap' in col])
# selected_features.extend([col for col in train_data.columns if 'spread' in col])
selected_features.extend([col for col in train_data.columns if 'flip' in col])
selected_features.extend([col for col in train_data.columns if 'count' in col])
selected_features.extend([col for col in train_data.columns if 'trade_ratio_lv1_' in col])
selected_features = set(selected_features)

# stock group feature
corr = feature.get_correlation('./data/train.csv')
group_stock_features = feature.get_stock_group_features(train_data, corr, selected_features)
train_features = train_data.merge(group_stock_features, on='time_id_', how='left')

train_y = pd.read_csv('data/train.csv').query(f"stock_id in {stock_ids}")
train_y = pd.merge(train_data[['stock_id', 'time_id_']].rename(
    {'time_id_':'time_id'}, axis=1), train_y, on=['stock_id', 'time_id'])

# LGBM train
params = {
     'learning_rate': 0.06,
     'bagging_fraction': 0.72,
     'bagging_freq': 4,
     'feature_fraction': 0.6,
     'lambda_l1': 0.5,
     'lambda_l2': 1.0,
     'categorical_column':[0]}

models = lgbm_train.train(train_features.drop('time_id_', axis=1), train_y.target, 5, params)

# feature importance
f_imp = lgbm_train.get_feature_importance(models[1])
f_imp.importance.describe(percentiles=[.10, .25, .50, .75, .90])
f_imp.head(60)
```
