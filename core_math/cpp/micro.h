#ifndef TRAD_ENGINE_CORE_MATH_CPP_MICRO_H
#define TRAD_ENGINE_CORE_MATH_CPP_MICRO_H

// core_math — C++ mirror of the microstructure primitives in core_math
// (micro_math.py, l2_math.py).
//
// The Python module is the REFERENCE. This is the deterministic, GC-free mirror
// for the live tick hot path, and it is kept in lockstep with the reference by
// parity tests. Never change a formula here unless the parity test agrees — a
// silent drift here is exactly the second, quietly diverging implementation the
// parity discipline exists to prevent.
//
// Every function is C-linkage + plain double buffers so it loads via ctypes from
// Python with no binding layer.

#ifdef __cplusplus
extern "C" {
#endif

// Weighted order-book imbalance — numeric core of
// l2_math.book_imbalance_signal. Each depth level weighted 1/(idx+1).
// Returns buy/(buy+sell) in [0,1], or -1.0 when total weighted volume is zero.
double book_imbalance(const double* bid_vols, int n_bids,
                      const double* ask_vols, int n_asks);

// Bulk-volume classification, mode="close_return" — mirrors
// micro_math.bulk_volume_classification. Writes per-bar buy/sell volume
// fractions into out_buy/out_sell (length n). Matches the reference exactly,
// INCLUDING the warmup: where the rolling std has fewer than
// min_periods = max(3, vol_window/3) observations it becomes eps (NOT NaN), so
// the output is finite everywhere — same as the Python side.
void bvc_close_return(const double* close, int n, int vol_window, double eps,
                      double* out_buy, double* out_sell);

// VPIN — rolling mean of |V_buy - V_sell| / V_total over the last n_buckets
// (min_periods=1, window grows until full). Mirrors vpin_from_buckets.
void vpin_from_abs_imbalance(const double* abs_imb, int n, int n_buckets,
                             double* out);

// TIB — rolling mean of the signed imbalance (min_periods=1). Mirrors
// tib_from_buckets.
void tib_from_dir_imbalance(const double* dir, int n, int n_buckets, double* out);

#ifdef __cplusplus
}
#endif

#endif  // TRAD_ENGINE_CORE_MATH_CPP_MICRO_H
