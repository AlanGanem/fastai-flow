import random
import pandas as pd
import numpy as np


def df_split(df, train_frac, date_col=None, test_days=None, start_from=None):
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        train_msk, test_msk = date_split_msk(
            df[date_col], train_frac=train_frac, test_days=test_days, start_from=start_from
        )
    else:
        train_msk, test_msk = random_split_msk(df, train_frac)

    # split data
    train_df = df[train_msk]
    test_df = df[test_msk]

    return train_df, test_df


def random_split_msk(df, train_frac=0.8):
    # exact split but inefficient
    # assert 1 > train_frac > 0
    # k = int(df.shape[0] * train_frac)
    # idxs = random.sample(list(range(df.shape[0])), k)
    # train_msk = np.array([i in list(range(df.shape[0])) for i in idxs])
    # test_msk = ~train_msk
    i = 0
    while True:
        rand = np.random.random(df.shape[0])
        # check if actual frac is close to desired frac
        if abs(train_frac - (rand <= train_frac).mean()) / train_frac < 0.001:
            break
        if i > 100:
            break
        i += 1
    train_msk = rand <= train_frac

    test_msk = ~train_msk
    return train_msk, test_msk


def date_split_msk(date_data, train_frac=None, test_days=None, start_from=None):
    total_args = [1 if not i is None else 0 for i in [train_frac, test_days, start_from]]
    if sum(total_args) != 1:
        raise TypeError('Must specify exactly one of "train_frac", "test_days", "start_from"')

    if train_frac:
        split_date = pd.to_datetime(find_date_split_from_frac(date_data, train_frac))
    elif test_days:
        split_date = pd.to_datetime(date_data.max()) - pd.Timedelta(test_days, unit='D')
    elif start_from:
        split_date = start_from
    else:
        raise TypeError('Must specify exactly one of "train_frac", "test_days", "start_from"')

    train_date_msk = date_data <= split_date

    return train_date_msk, ~train_date_msk


def find_date_split_from_frac(date_data, train_frac):
    assert 1 > train_frac > 0
    assert date_data.__class__ == pd.Series

    sorted_dates = date_data.sort_values()
    unique_sorted = sorted_dates.sort_values().unique()
    assert unique_sorted.shape[0] > 1

    suboptimal_split_date = sorted_dates.iloc[int((1 - train_frac) * sorted_dates.shape[0])]
    idx = list(unique_sorted).index(suboptimal_split_date)
    date_p1 = unique_sorted[idx + 1]
    date_m1 = unique_sorted[idx - 1]

    p1_frac = (date_data <= date_m1).mean()
    p2_frac = (date_data <= date_p1).mean()

    diffs = [abs(p1_frac - train_frac), abs(p2_frac - train_frac)]
    optimal_date = [date_p1, date_m1][diffs.index(min(diffs))]
    print(optimal_date)
    return optimal_date


def drop_dependent_nan(df, dependent_vars):
    # drop label nans
    df = df.dropna(subset=dependent_vars)
    return df

def str_caster(df, cols):
    df.loc[:, cols] = df[cols].astype(str)
    return df
