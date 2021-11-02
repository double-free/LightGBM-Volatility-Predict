import pandas as pd
import numpy as np


def calculate_realized_volatility(price):
    log_return = np.log(price).diff().dropna()
    return np.sqrt(np.sum(log_return ** 2))


def cut_by_time(df, window_seconds):
    batch_id = df.groupby("time_id", group_keys=False).apply(
        lambda g: (g.seconds_in_bucket / window_seconds).astype(int)
    )
    return batch_id


def get_features(df, feature_dict):
    features = (
        df.groupby(["time_id", "batch_id"], group_keys=False)
        .agg(feature_dict)
        .reset_index()
    )

    # concat multi-level column
    features.columns = ["_".join(col) for col in features.columns]
    # now each time id has several rows of features for different time window
    # use pandas.pivot to flat these rows
    flat_features = features.pivot(index="time_id_", columns="batch_id_")
    flat_features.columns = [f"{col[0]}_{col[1]}" for col in flat_features.columns]
    return flat_features.reset_index()


def vwap(row, bid_idx, ask_idx):
    # TODO: multi-level
    return (
        row[f"bid_price{bid_idx}"] * row[f"ask_size{ask_idx}"]
        + row[f"ask_price{ask_idx}"] * row[f"bid_size{bid_idx}"]
    ) / (row[f"ask_size{ask_idx}"] + row[f"bid_size{bid_idx}"])


def get_book(stock_id):
    book = pd.read_parquet(f"data/book_train.parquet/stock_id={stock_id}")

    # VWAPs

    # level 0 price does not change
    book["vwap11"] = vwap(book, 1, 1)
    # ask level 0 is fully traded
    book["vwap12"] = vwap(book, 1, 2)
    # bid level 0 is fully traded
    book["vwap21"] = vwap(book, 2, 1)
    # bid and ask level 0 is fully traded
    book["vwap22"] = vwap(book, 2, 2)

    book["bid_ask_spread"] = book.ask_price1 - book.bid_price1
    book["bid_gap"] = book.bid_price1 - book.bid_price2
    book["ask_gap"] = book.ask_price2 - book.ask_price1

    book["bid_imbalance"] = book.bid_size1 / book.bid_size2
    book["ask_imbalance"] = book.ask_size1 / book.ask_size2

    return book


def get_book_features(book, window):
    book["batch_id"] = cut_by_time(book, window)

    feature_dict = {
        "vwap11": ["mean", "std", "max", calculate_realized_volatility],
        "vwap12": ["mean", "std", "max", calculate_realized_volatility],
        "vwap21": ["mean", "std", "max", calculate_realized_volatility],
        "vwap22": ["mean", "std", "max", calculate_realized_volatility],
        "bid_gap": ["mean", "std", "max"],
        "ask_gap": ["mean", "std", "max"],
        "bid_ask_spread": ["mean", "std", "max"],
        "bid_size1": ["mean", "std", "max", "sum"],
        "ask_size1": ["mean", "std", "max", "sum"],
        "bid_imbalance": ["mean", "std", "max"],
        "ask_imbalance": ["mean", "std", "max"],
    }

    return get_features(book, feature_dict)


def get_trade(stock_id):
    trade = pd.read_parquet(f"data/trade_train.parquet/stock_id={stock_id}").rename(
        {"size": "trade_volume"}, axis=1
    )
    trade["trade_amount"] = trade.price * trade.trade_volume
    trade["per_trade_quantity"] = trade.trade_volume / trade.order_count

    return trade


def get_trade_features(trade, window):
    trade["batch_id"] = cut_by_time(trade, window)
    feature_dict = {
        "trade_volume": ["mean", "std", "max", "sum"],
        "trade_amount": ["mean", "std", "max"],
        "per_trade_quantity": ["mean", "std", "max"],
    }

    return get_features(trade, feature_dict)


# use this to get book & trade features
def get_all_features(stock_id, window):
    book = get_book(stock_id)
    book_features = get_book_features(book, window)
    trade = get_trade(stock_id)
    trade_features = get_trade_features(trade, window)

    merged = pd.merge(book_features, trade_features, on=["time_id_"])
    return merged
