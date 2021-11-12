from sklearn.model_selection import KFold
import lightgbm
import numpy as np
import pandas as pd

# root mean squared percentage error
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    # eval_name, eval_result, is_higher_better
    return "RMSPE", rmspe(y_true, y_pred), False


def train(train_features, train_y_true, k, lgbm_params={}):
    kf = KFold(n_splits=k, random_state=2021, shuffle=True)
    models = []
    for train_index, test_index in kf.split(train_features):
        X_train, X_test = (
            train_features.loc[train_index],
            train_features.loc[test_index],
        )
        y_train, y_test = train_y_true.loc[train_index], train_y_true.loc[test_index]

        train_dataset = lightgbm.Dataset(
            X_train, y_train, weight=1 / np.square(y_train)
        )
        validation_dataset = lightgbm.Dataset(
            X_test, y_test, weight=1 / np.square(y_test)
        )
        model = lightgbm.train(
            params=lgbm_params,
            train_set=train_dataset,
            valid_sets=[train_dataset, validation_dataset],
            feval=feval_rmspe,
            num_boost_round=1000,
            callbacks=[lightgbm.early_stopping(100), lightgbm.log_evaluation(50)],
        )
        # get prediction score
        y_pred = model.predict(X_test)
        print("RMSPE = ", rmspe(y_test, y_pred))
        lightgbm.plot_importance(model, max_num_features=20)
        models.append(model)
    return models


def get_feature_importance(model):
    return pd.DataFrame(
        {"feature": model.feature_name(), "importance": model.feature_importance()}
    ).sort_values(by="importance", ascending=False)
