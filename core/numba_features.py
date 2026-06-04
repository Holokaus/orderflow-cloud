import numpy as np


def calculate_atr_nb(high, low, close, window):
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, max(tr2, tr3))
    if n > window:
        atr[window] = np.mean(tr[1:window+1])
        for i in range(window + 1, n):
            atr[i] = (atr[i-1] * (window - 1) + tr[i]) / window
    return atr


def rolling_zscore_nb(values, window):
    n = len(values)
    result = np.zeros(n)
    for i in range(window - 1, n):
        chunk = values[i - window + 1:i + 1]
        mean = 0.0
        for v in chunk:
            mean += v
        mean /= window
        var = 0.0
        for v in chunk:
            var += (v - mean) ** 2
        std = np.sqrt(var / window) + 1e-9
        result[i] = (values[i] - mean) / std
    return result


def rolling_mean_nb(values, window):
    n = len(values)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        s = 0.0
        cnt = 0
        for j in range(start, i + 1):
            s += values[j]
            cnt += 1
        result[i] = s / cnt if cnt > 0 else 0.0
    return result


def rolling_max_nb(values, window):
    n = len(values)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        mx = values[start]
        for j in range(start + 1, i + 1):
            if values[j] > mx:
                mx = values[j]
        result[i] = mx
    return result


def rolling_min_nb(values, window):
    n = len(values)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        mn = values[start]
        for j in range(start + 1, i + 1):
            if values[j] < mn:
                mn = values[j]
        result[i] = mn
    return result


def cumsum_by_side_nb(sides, volumes, prices):
    n = len(sides)
    cum_buy_vol = np.zeros(n)
    cum_sell_vol = np.zeros(n)
    cum_buy_cnt = np.zeros(n)
    cum_sell_cnt = np.zeros(n)
    cum_value = np.zeros(n)
    cum_buy_value = np.zeros(n)
    for i in range(n):
        bv = cum_buy_vol[i - 1] if i > 0 else 0.0
        sv = cum_sell_vol[i - 1] if i > 0 else 0.0
        bc = cum_buy_cnt[i - 1] if i > 0 else 0.0
        sc = cum_sell_cnt[i - 1] if i > 0 else 0.0
        cv = cum_value[i - 1] if i > 0 else 0.0
        cbv = cum_buy_value[i - 1] if i > 0 else 0.0
        if sides[i] == 1:
            bv += volumes[i]
            bc += 1
            cbv += prices[i] * volumes[i]
        else:
            sv += volumes[i]
            sc += 1
        cv += prices[i] * volumes[i]
        cum_buy_vol[i] = bv
        cum_sell_vol[i] = sv
        cum_buy_cnt[i] = bc
        cum_sell_cnt[i] = sc
        cum_value[i] = cv
        cum_buy_value[i] = cbv
    return cum_buy_vol, cum_sell_vol, cum_buy_cnt, cum_sell_cnt, cum_value, cum_buy_value


def find_window_starts_nb(timestamps_epoch, windows_sec):
    n = len(timestamps_epoch)
    n_windows = len(windows_sec)
    indices = np.zeros((n, n_windows), dtype=np.int64)
    for w in range(n_windows):
        window = windows_sec[w]
        idx = 0
        for i in range(n):
            cutoff = timestamps_epoch[i] - window
            while idx < i and timestamps_epoch[idx] < cutoff:
                idx += 1
            indices[i, w] = idx
    return indices


def compute_trade_features_nb(
    cum_buy_vol, cum_sell_vol, cum_buy_cnt, cum_sell_cnt,
    cum_value, cum_buy_value,
    window_starts, tick_idx, window_sec,
):
    start = window_starts
    if tick_idx == start:
        return np.zeros(10)
    bv = cum_buy_vol[tick_idx] - (cum_buy_vol[start - 1] if start > 0 else 0)
    sv = cum_sell_vol[tick_idx] - (cum_sell_vol[start - 1] if start > 0 else 0)
    bc = cum_buy_cnt[tick_idx] - (cum_buy_cnt[start - 1] if start > 0 else 0)
    sc = cum_sell_cnt[tick_idx] - (cum_sell_cnt[start - 1] if start > 0 else 0)
    cv = cum_value[tick_idx] - (cum_value[start - 1] if start > 0 else 0)
    cbv = cum_buy_value[tick_idx] - (cum_buy_value[start - 1] if start > 0 else 0)
    tv = bv + sv
    tc = bc + sc
    delta = bv - sv
    features = np.zeros(10)
    features[0] = bv
    features[1] = sv
    features[2] = tv
    features[3] = delta
    features[4] = delta / (tv + 1e-9)
    features[5] = abs(delta)
    features[6] = tc
    features[7] = tc / window_sec if window_sec > 0 else 0.0
    features[8] = tv / (tc + 1e-9)
    features[9] = cv / (tv + 1e-9)
    return features


def compute_atr_from_features_nb(prices, bid_prices, ask_prices, window):
    n = len(prices)
    atr = np.zeros(n)
    for i in range(1, n):
        high = max(bid_prices[i] if i > 0 else prices[i],
                   ask_prices[i] if i > 0 else prices[i],
                   prices[i])
        low = min(bid_prices[i] if i > 0 else prices[i],
                  ask_prices[i] if i > 0 else prices[i],
                  prices[i])
        prev_close = prices[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if i < window:
            atr[i] = tr
        else:
            atr[i] = (atr[i - 1] * (window - 1) + tr) / window
    return atr
