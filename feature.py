import pandas as pd
import numpy as np
import multiprocessing as mp
import glob


def get_all_stock_ids(train_dir):
    paths = glob.glob(f"{train_dir}/*")
    return sorted([int(path.split("=")[1]) for path in paths])


def get_cache_file_name(stock_ids, window):
    # encode stock list
    # every bit represent if a stock is picked or not
    # two 64bit integers shall be enough
    high = 0
    low = 0
    for stock_id in stock_ids:
        if stock_id < 64:
            low += 1 << stock_id
        else:
            high += 1 << (stock_id - 64)
    cache = f"./data/mine/{window}/{hex(high)}_{hex(low)}.csv"
    return cache


def calculate_realized_volatility(price):
    log_return = np.log(price).diff().dropna()
    return np.sqrt(np.sum(log_return ** 2))


def cut_by_time(df, window_seconds):
    batch_id = (df.seconds_in_bucket / window_seconds).astype(int)
    return pd.Series(batch_id, name="batch_id", index=df.index)


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
    flat_features.columns = [f"{col[0]}_batch{col[1]}" for col in flat_features.columns]
    return flat_features.reset_index()


def vwap(row, bid_idx, ask_idx):
    # TODO: multi-level
    return (
        row[f"bid_price{bid_idx}"] * row[f"ask_size{ask_idx}"]
        + row[f"ask_price{ask_idx}"] * row[f"bid_size{bid_idx}"]
    ) / (row[f"ask_size{ask_idx}"] + row[f"bid_size{bid_idx}"])


def get_book_features(raw_book, window):
    # VWAPs
    book = raw_book.copy()
    # level 0 price does not change
    book["vwap11"] = vwap(book, 1, 1)
    # ask level 0 is fully traded
    book["vwap12"] = vwap(book, 1, 2)
    # bid level 0 is fully traded
    book["vwap21"] = vwap(book, 2, 1)
    # bid and ask level 0 is fully traded
    book["vwap22"] = vwap(book, 2, 2)

    book["bid_ask_spread"] = book.ask_price1 - book.bid_price1

    book["total_volume_lv1"] = book.ask_size1 + book.bid_size1
    book["total_volume_lv12"] = (
        book.ask_size1 + book.bid_size1 + book.ask_size2 + book.bid_size2
    )

    # book flip (cross spread to take orders, extremely aggressive behavior)
    book["flip"] = book.ask_price1.shift(-1) <= book.bid_price1

    book["batch_id"] = cut_by_time(book, window)
    feature_dict = {
        "vwap11": ["mean", "std", calculate_realized_volatility],
        "vwap12": ["mean", "std", calculate_realized_volatility],
        "vwap21": ["mean", "std", calculate_realized_volatility],
        "vwap22": ["mean", "std", calculate_realized_volatility],
        "bid_ask_spread": ["mean", "std"],
        "total_volume_lv1": ["mean", "std", "sum"],
        "total_volume_lv12": ["mean", "std", "sum"],
        "flip": ["sum"],
        "seconds_in_bucket": "count",
    }

    return get_features(book, feature_dict)


def get_trade_features(raw_trade, raw_book, window):
    trade = raw_trade.rename({"size": "trade_volume"}, axis=1).copy()
    trade["per_trade_quantity"] = trade.trade_volume / trade.order_count

    # a complex feature, trade_volume/(ask_size + bid_size)
    # this may give insight on how much percentage of ToB is taken
    raw_book["time_seconds"] = (
        raw_book.time_id.astype(int) * 600 + raw_book.seconds_in_bucket
    )
    trade["time_seconds"] = trade.time_id.astype(int) * 600 + trade.seconds_in_bucket
    merged = pd.merge_asof(
        trade,
        raw_book[
            [
                "time_id",
                "time_seconds",
                "bid_size1",
                "ask_size1",
                "bid_size2",
                "ask_size2",
            ]
        ],
        by="time_id",
        on="time_seconds",
    )
    merged["trade_ratio_lv1"] = merged.trade_volume / (
        merged.bid_size1 + merged.ask_size1
    )
    merged["trade_ratio_lv12"] = merged.trade_volume / (
        merged.bid_size1 + merged.ask_size1 + merged.bid_size2 + merged.ask_size2
    )

    merged["batch_id"] = cut_by_time(merged, window)
    feature_dict = {
        "trade_volume": ["mean", "std", "sum"],
        "order_count": ["mean", "std", "sum"],
        "per_trade_quantity": ["mean", "std"],
        "trade_ratio_lv1": ["mean", "std"],
        "trade_ratio_lv12": ["mean", "std"],
        "seconds_in_bucket": "count",
    }

    return get_features(merged, feature_dict)


# mode = "train" or "test"
def get_one_stock_features(stock_id, window, mode):
    book_path = f"./data/book_{mode}.parquet/stock_id={stock_id}"
    raw_book = pd.read_parquet(book_path)
    book_features = get_book_features(raw_book, window)
    trade_path = f"./data/trade_{mode}.parquet/stock_id={stock_id}"
    raw_trade = pd.read_parquet(trade_path)
    trade_features = get_trade_features(raw_trade, raw_book, window)
    # left join to handle "no trade" cases for low liquidity stocks
    # it is safe because book update must >= trade update
    merged = pd.merge(
        book_features,
        trade_features,
        on=["time_id_"],
        how="left",
        suffixes=["_book", "_trade"],
    )
    merged.insert(loc=0, column="stock_id", value=stock_id)
    return merged


# use this to get book & trade features
def get_stock_features(stock_ids, window, mode):
    with mp.Pool(4) as p:
        results = p.starmap(
            get_one_stock_features,
            zip(stock_ids, [window] * len(stock_ids), [mode] * len(stock_ids)),
        )
        return pd.concat(results).reset_index(drop=True)


def get_correlation(y_path):
    vol_true = pd.read_csv(y_path).pivot(
        index="time_id", columns="stock_id", values="target"
    )
    # correlation is based on the "change rate" of volatility
    # instead of the raw volatility, I think it is comparable between stocks
    return (vol_true / vol_true.shift(1)).corr()


def get_similar_stock_features(train_data, corr, n, selected_features):
    selected_features = set(selected_features)
    df = []
    for stock_id in train_data.stock_id.unique():
        # remove itself
        top_n_stocks = corr[[stock_id]].nlargest(n + 1, stock_id).index[1:]
        similar_stock_features = (
            train_data[train_data.stock_id.isin(top_n_stocks)]
            .groupby("time_id_")
            .mean()
            .reindex(selected_features, axis=1)
        )
        similar_stock_features["stock_id"] = stock_id
        df.append(similar_stock_features)

    return (
        pd.concat(df)
        .rename({col: f"{col}_similar" for col in selected_features}, axis=1)
        .reset_index()
    )


def get_stock_group_features(train_features, corr, selected_features):
    copied_corr = corr.copy()
    from sklearn.cluster import KMeans

    # clustering = DBSCAN(eps=0.4, min_samples=2).fit(corr.values)
    clustering = KMeans(n_clusters=5, random_state=0).fit(copied_corr)
    copied_corr["group_id"] = clustering.labels_
    merged = train_features.merge(copied_corr[["group_id"]], on="stock_id")
    group_features = (
        merged.groupby(["time_id_", "group_id"])
        .mean()
        .reindex(selected_features, axis=1)
        .reset_index()
        .pivot(index="time_id_", columns="group_id")
    )
    group_features.columns = [
        f"{col[0]}_group{col[1]}" for col in group_features.columns
    ]
    return group_features.reset_index()
