def predict(models, features):
    return sum([model.predict(features) / len(models) for model in models])


def get_submission(stock_id, time_id, target):
    row_id = stock_id.astype(str) + "-" + time_id_.astype(str)
    return pd.DataFrame({"row_id": row_id, "target": target})


def submit(models, test_features):
    target = predict(models, test_features.drop("time_id_", axis=1))
    submission = get_submission(test_features.stock_id, test_features.time_id_, target)
    submission.to_csv("submission.csv", index=False)
