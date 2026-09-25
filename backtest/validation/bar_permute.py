import numpy as np
import pandas as pd
from typing import List, Union, Optional

def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]],
    start_index: int = 0,
    seed: Optional[int] = None,
    extra_cols_mode: str = 'permute',
    permutation_mode: str = 'bar',
    block_length: Optional[int] = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Generate permuted OHLC data preserving statistical properties.

    Args:
        ohlc: OHLC DataFrame or list of DataFrames
        start_index: Index from which to start permutation
        seed: Random seed for reproducibility
        extra_cols_mode: How to handle extra columns (spot, futures, volume, etc.)
            - 'permute': Apply same permutation to extra columns (default, prevents leakage)
            - 'preserve': Keep original values (old behaviour, causes data leakage)
            - 'sync_close': Synchronise with permuted close prices (for futures column)
        permutation_mode: Type of permutation to apply. Each mode is a DIFFERENT null
            hypothesis — pick the one that matches the question, not the "strictest".
            Numbers below: rate of p <= 0.05 over 30 seeds, 1-bar momentum strategy
            (tests/test_sanity.py documents each behaviour).
            - 'bar' [DEFAULT]: shuffles the bars' relative moves (open vs previous close,
              high/low/close vs open). Null: "no temporal structure at all". Detects
              short-range edges (AR(1) phi=0.1 momentum: 87%), false positives on noise
              ~3%. Returns are recomputed from the permuted prices, so a lookahead leak
              helps the permutations as much as the real run: it does NOT create a
              false positive (3%) — but it is not flagged either (inflated PnL + high p
              is the signature to watch for; use pit_invariants to actually catch it).
            - 'row': shuffles whole rows, keeping each row's columns together (useful
              when the signal is a cross-column relation, e.g. a spread). WARNING: the
              returns are pre-computed and travel with the rows, so a lookahead leak
              does NOT carry over to the permutations and is certified as an edge
              (10% leak, no real edge: 67% "significant"). Only use it on features that
              passed the point-in-time checks.
            - 'block': moving-block bootstrap of the returns. It PRESERVES dependence
              shorter than the block, so its null is "no edge beyond the short-range
              autocorrelation": it cannot see an edge that lives inside a block (1-bar
              momentum: block 1 -> 87%, 2 -> 20%, 3 -> 3%, 5 and 10 -> 0%). Right when the
              question is "is there edge beyond local clustering?"; wrong for testing a
              short-horizon momentum / mean-reversion signal. Very conservative on noise.
            - 'ar1': for basis strategies (needs 'spot' and 'futures'): simulates the
              basis as an AR(1) fitted on the training part. Like 'block', it keeps the
              autocorrelation in the null, so a basis mean-reversion edge that IS that
              autocorrelation is not what it tests. (Not covered by the study above.)
        block_length: Block length for 'block'. Default 10 bars. Must be SHORTER than the
            horizon your strategy exploits, or the edge is inside the blocks and invisible.

    Returns:
        Permuted OHLC data in same format as input

    Raises:
        ValueError: If start_index is negative or data is invalid
    """
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    rng = np.random.default_rng(seed)

    if isinstance(ohlc, list):
        if not ohlc:
            raise ValueError("OHLC list cannot be empty")
        time_index = ohlc[0].index
        for i, mkt in enumerate(ohlc):
            if not np.all(time_index == mkt.index):
                raise ValueError(f"Index mismatch in market {i}")
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])

    # Row-wise permutation: shuffle entire rows to preserve column correlations
    if permutation_mode == 'row':
        perm_ohlc = []
        for reg_bars in ohlc:
            perm_bars = reg_bars.copy()

            # Create permutation indices for rows after start_index
            n_perm_rows = n_bars - start_index
            perm_indices = np.arange(start_index, n_bars)
            rng.shuffle(perm_indices)

            # Keep rows before start_index unchanged, permute rest
            perm_bars.iloc[start_index:] = reg_bars.iloc[perm_indices].values

            perm_ohlc.append(perm_bars)

        if n_markets > 1:
            return perm_ohlc
        else:
            return perm_ohlc[0]

    # AR(1) permutation: generate synthetic basis using AR(1) model
    if permutation_mode == 'ar1':
        perm_ohlc = []
        for reg_bars in ohlc:
            # Verify required columns exist
            if 'spot' not in reg_bars.columns or 'futures' not in reg_bars.columns:
                raise ValueError("AR(1) permutation requires 'spot' and 'futures' columns")

            perm_bars = reg_bars.copy()

            # Calculate original basis
            basis = (reg_bars['futures'] - reg_bars['spot']) / reg_bars['spot']

            # Need minimum training data to fit AR(1) model reliably
            min_train_size = max(50, n_bars // 3)  # At least 50 bars or 1/3 of data

            if start_index < min_train_size:
                # Use first min_train_size bars as training data for AR(1) parameters
                # This creates a train/test split to avoid data leakage
                train_end_idx = min_train_size
            else:
                # If start_index >= min_train_size, use data up to start_index
                train_end_idx = start_index

            # Fit AR(1) model ONLY on training data (no look-ahead)
            basis_train = basis.iloc[:train_end_idx]
            mu = basis_train.mean()
            basis_train_centered = basis_train - mu

            # Estimate phi using lag-1 autocorrelation on training data only
            phi = basis_train_centered.autocorr(lag=1)

            # Calculate residuals from training data only
            basis_train_lagged = basis_train_centered.shift(1).iloc[1:]
            basis_train_current = basis_train_centered.iloc[1:]
            residuals = basis_train_current - phi * basis_train_lagged

            # Generate synthetic basis using AR(1) model
            synthetic_basis = np.zeros(n_bars)

            # Keep training period data unchanged (needed as baseline)
            synthetic_basis[:train_end_idx] = basis.iloc[:train_end_idx].values

            # Generate AR(1) process for test period only
            for i in range(train_end_idx, n_bars):
                # Sample random residual from training data residuals
                epsilon = rng.choice(residuals.values)
                # AR(1): basis[t] = mu + phi * (basis[t-1] - mu) + epsilon
                synthetic_basis[i] = mu + phi * (synthetic_basis[i-1] - mu) + epsilon

            # Reconstruct futures from synthetic basis: futures = spot * (1 + basis)
            perm_bars['futures'] = perm_bars['spot'] * (1 + synthetic_basis)

            # Update close to match futures (assuming close tracks futures)
            perm_bars['close'] = perm_bars['futures']

            # Update OHLC to be consistent with new close
            # Keep relative OHLC structure but scaled to new close
            close_ratio = perm_bars['close'] / reg_bars['close']
            perm_bars['open'] = reg_bars['open'] * close_ratio
            perm_bars['high'] = reg_bars['high'] * close_ratio
            perm_bars['low'] = reg_bars['low'] * close_ratio

            perm_ohlc.append(perm_bars)

        if n_markets > 1:
            return perm_ohlc
        else:
            return perm_ohlc[0]

    # Block bootstrap: resample blocks of returns while keeping predictors unchanged
    # This tests whether the strategy exploits a genuine predictive relationship
    if permutation_mode == 'block':
        perm_ohlc = []
        for reg_bars in ohlc:
            perm_bars = reg_bars.copy()

            # Determine block length
            if block_length is None:
                # Default: 10 bars (the n^(1/3) rule of thumb for ~1000 bars). It
                # decides what the test can see: any edge whose horizon fits inside
                # a block is preserved in the null and becomes invisible (see the
                # docstring and tests/test_sanity.py).
                b_len = 10
            else:
                b_len = max(1, int(block_length))

            # Calculate log returns from close prices
            log_returns = np.log(reg_bars['close']).diff().fillna(0)

            # Generate block bootstrap sample of returns
            perm_returns = np.zeros(n_bars)

            # Keep training period unchanged
            perm_returns[:start_index] = log_returns.iloc[:start_index].values

            # Resample blocks for permuted period
            pos = start_index
            while pos < n_bars:
                # Randomly select a block start position from the permutable region
                block_start = rng.integers(start_index, n_bars - b_len + 1)

                # Calculate how many bars we can copy from this block
                remaining = n_bars - pos
                block_size = min(b_len, remaining, n_bars - block_start)

                # Copy the block of returns
                perm_returns[pos:pos + block_size] = log_returns.iloc[block_start:block_start + block_size].values

                pos += block_size

            # Reconstruct prices from permuted returns
            # Start from the first price and apply cumulative returns
            perm_log_prices = np.zeros(n_bars)
            perm_log_prices[0] = np.log(reg_bars['close'].iloc[0])

            for i in range(1, n_bars):
                perm_log_prices[i] = perm_log_prices[i-1] + perm_returns[i]

            perm_close = np.exp(perm_log_prices)

            # Update close prices
            perm_bars['close'] = perm_close

            # Scale OHLC proportionally to maintain relative structure
            close_ratio = perm_bars['close'] / reg_bars['close']
            perm_bars['open'] = reg_bars['open'] * close_ratio
            perm_bars['high'] = reg_bars['high'] * close_ratio
            perm_bars['low'] = reg_bars['low'] * close_ratio

            # If spot/futures columns exist, scale them proportionally as well
            # This keeps the basis relationship intact but tests on permuted price movements
            if 'spot' in reg_bars.columns:
                perm_bars['spot'] = reg_bars['spot'] * close_ratio
            if 'futures' in reg_bars.columns:
                perm_bars['futures'] = reg_bars['futures'] * close_ratio

            perm_ohlc.append(perm_bars)

        if n_markets > 1:
            return perm_ohlc
        else:
            return perm_ohlc[0]

    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    # Store extra columns data for permutation
    extra_cols_data = {}
    for mkt_i, reg_bars in enumerate(ohlc):
        extra_cols = [col for col in reg_bars.columns if col not in ['open', 'high', 'low', 'close']]
        if mkt_i == 0 and extra_cols:
            for col in extra_cols:
                extra_cols_data[col] = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']])

        # Get start bar
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # Open relative to last close
        r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()

        # Get prices relative to this bars open
        r_h = (log_bars['high'] - log_bars['open']).to_numpy()
        r_l = (log_bars['low'] - log_bars['open']).to_numpy()
        r_c = (log_bars['close'] - log_bars['open']).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

        # Store relative movements for extra columns (for permutation)
        extra_cols = [col for col in reg_bars.columns if col not in ['open', 'high', 'low', 'close']]
        for col in extra_cols:
            if extra_cols_mode == 'permute':
                # Calculate relative movements (log returns) for this column
                # Replace zeros/negatives with small positive value to avoid log(0) warnings
                col_values = reg_bars[col].copy()
                col_values = col_values.replace(0, np.nan)  # Replace zeros with NaN
                col_values = col_values.clip(lower=1e-10)   # Clip negative values
                log_col = np.log(col_values)
                r_col = log_col.diff().to_numpy()
                # Fill NaN at index 0 with 0 (no change from previous bar)
                r_col[0] = 0.0
                # Replace any NaN from zero values with 0
                r_col[np.isnan(r_col)] = 0.0
                extra_cols_data[col][mkt_i] = r_col[perm_index:]

    idx = np.arange(perm_n)

    # Shuffle intrabar relative values (h/l/c)
    perm1 = rng.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    perm2 = rng.permutation(idx)
    relative_open = relative_open[:, perm2]

    # Apply same permutations to extra columns if in permute mode
    if extra_cols_mode == 'permute':
        for col in extra_cols_data:
            # Use same permutation as close prices
            extra_cols_data[col] = extra_cols_data[col][:, perm1]

    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        # Copy over real data before start index 
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        
        # Copy start bar
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['open', 'high', 'low', 'close'])

        # Handle additional columns based on mode
        original_df = reg_bars
        extra_cols = [col for col in original_df.columns if col not in ['open', 'high', 'low', 'close']]

        if extra_cols:
            for col in extra_cols:
                if extra_cols_mode == 'permute':
                    # Reconstruct permuted column from permuted relative movements
                    perm_col_values = np.zeros(n_bars)
                    # Replace zeros/negatives with small positive value to avoid log(0) warnings
                    col_values_orig = original_df[col].copy()
                    col_values_orig = col_values_orig.replace(0, np.nan)
                    col_values_orig = col_values_orig.clip(lower=1e-10)
                    log_original = np.log(col_values_orig).to_numpy()
                    # Handle any NaN from zeros
                    log_original[np.isnan(log_original)] = 0.0

                    # Copy pre-permutation data
                    perm_col_values[:start_index] = log_original[:start_index]
                    perm_col_values[start_index] = log_original[start_index]

                    # Reconstruct from permuted relative movements
                    for j in range(perm_index, n_bars):
                        k = j - perm_index
                        delta = extra_cols_data[col][mkt_i][k]
                        # Handle any remaining NaN values
                        if np.isnan(delta):
                            delta = 0.0
                        perm_col_values[j] = perm_col_values[j - 1] + delta

                    perm_bars[col] = np.exp(perm_col_values)

                elif extra_cols_mode == 'sync_close':
                    # Synchronise with permuted close (for futures column)
                    if col == 'futures':
                        perm_bars[col] = perm_bars['close'].values
                    else:
                        perm_bars[col] = original_df[col].values

                else:  # 'preserve' mode
                    # Keep original values (old behaviour, causes data leakage)
                    perm_bars[col] = original_df[col].values

        perm_ohlc.append(perm_bars)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    btc_real = pd.read_parquet('BTCUSD3600.pq')
    btc_real.index = btc_real.index.astype('datetime64[s]')
    btc_real = btc_real[(btc_real.index.year >= 2018) & (btc_real.index.year < 2020)]

    btc_perm = get_permutation(btc_real)

    btc_real_r = np.log(btc_real['close']).diff() 
    btc_perm_r = np.log(btc_perm['close']).diff()

    print(f"Mean. REAL: {btc_real_r.mean():14.6f} PERM: {btc_perm_r.mean():14.6f}")
    print(f"Stdd. REAL: {btc_real_r.std():14.6f} PERM: {btc_perm_r.std():14.6f}")
    print(f"Skew. REAL: {btc_real_r.skew():14.6f} PERM: {btc_perm_r.skew():14.6f}")
    print(f"Kurt. REAL: {btc_real_r.kurt():14.6f} PERM: {btc_perm_r.kurt():14.6f}")

    eth_real = pd.read_parquet('ETHUSD3600.pq')
    eth_real.index = eth_real.index.astype('datetime64[s]')
    eth_real = eth_real[(eth_real.index.year >= 2018) & (eth_real.index.year < 2020)]
    eth_real_r = np.log(eth_real['close']).diff()
    
    print("") 

    permed = get_permutation([btc_real, eth_real])
    btc_perm = permed[0]
    eth_perm = permed[1]
    
    btc_perm_r = np.log(btc_perm['close']).diff()
    eth_perm_r = np.log(eth_perm['close']).diff()
    print(f"BTC&ETH Correlation REAL: {btc_real_r.corr(eth_real_r):5.3f} PERM: {btc_perm_r.corr(eth_perm_r):5.3f}")

    plt.style.use("dark_background")    
    np.log(btc_real['close']).diff().cumsum().plot(color='orange', label='BTCUSD')
    np.log(eth_real['close']).diff().cumsum().plot(color='purple', label='ETHUSD')
    
    plt.ylabel("Cumulative Log Return")
    plt.title("Real BTCUSD and ETHUSD")
    plt.legend()
    plt.show()

    np.log(btc_perm['close']).diff().cumsum().plot(color='orange', label='BTCUSD')
    np.log(eth_perm['close']).diff().cumsum().plot(color='purple', label='ETHUSD')
    plt.title("Permuted BTCUSD and ETHUSD")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    plt.show()



