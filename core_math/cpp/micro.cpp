// core_math C++ mirror — see micro.h. Kept in lockstep with the Python
// reference in core_math by parity tests.
#include "micro.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kSqrt2 = std::sqrt(2.0);

// Sample std (ddof=1) of v[start..=end]; NaN if fewer than 2 points.
double sample_std(const std::vector<double>& v, int start, int end) {
    const int count = end - start + 1;
    if (count < 2) return kNaN;
    double mean = 0.0;
    for (int k = start; k <= end; ++k) mean += v[k];
    mean /= count;
    double ss = 0.0;
    for (int k = start; k <= end; ++k) {
        const double d = v[k] - mean;
        ss += d * d;
    }
    return std::sqrt(ss / (count - 1));
}
}  // namespace

extern "C" {

double book_imbalance(const double* bid_vols, int n_bids,
                      const double* ask_vols, int n_asks) {
    double buy = 0.0, sell = 0.0;
    for (int i = 0; i < n_bids; ++i) buy += bid_vols[i] / (i + 1.0);
    for (int i = 0; i < n_asks; ++i) sell += ask_vols[i] / (i + 1.0);
    const double total = buy + sell;
    if (total == 0.0) return -1.0;
    return buy / total;
}

void bvc_close_return(const double* close, int n, int vol_window, double eps,
                      double* out_buy, double* out_sell) {
    if (n <= 0) return;
    const int min_periods = std::max(3, vol_window / 3);

    // lr = diff(log(clip(close, eps, inf))), with lr[0] = 0.0 (reference rule).
    std::vector<double> lr(n);
    lr[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        const double ci = close[i] < eps ? eps : close[i];
        const double cp = close[i - 1] < eps ? eps : close[i - 1];
        lr[i] = std::log(ci) - std::log(cp);
    }

    for (int i = 0; i < n; ++i) {
        int start = i - vol_window + 1;
        if (start < 0) start = 0;
        const int count = i - start + 1;

        double sig = (count < min_periods) ? kNaN : sample_std(lr, start, i);
        // Reference: sig = where(isfinite(sig) & (sig > eps), sig, eps).
        if (!(std::isfinite(sig) && sig > eps)) sig = eps;

        double z = lr[i] / sig;
        if (z > 8.0) z = 8.0;
        if (z < -8.0) z = -8.0;

        const double p_buy = 0.5 * std::erfc(-z / kSqrt2);  // Phi(z)
        out_buy[i] = p_buy;
        out_sell[i] = 1.0 - p_buy;
    }
}

void vpin_from_abs_imbalance(const double* abs_imb, int n, int n_buckets,
                             double* out) {
    for (int i = 0; i < n; ++i) {
        int start = i - n_buckets + 1;
        if (start < 0) start = 0;
        const int count = i - start + 1;
        double s = 0.0;
        for (int k = start; k <= i; ++k) s += abs_imb[k];
        out[i] = s / count;  // min_periods=1: always defined
    }
}

void tib_from_dir_imbalance(const double* dir, int n, int n_buckets,
                            double* out) {
    for (int i = 0; i < n; ++i) {
        int start = i - n_buckets + 1;
        if (start < 0) start = 0;
        const int count = i - start + 1;
        double s = 0.0;
        for (int k = start; k <= i; ++k) s += dir[k];
        out[i] = s / count;
    }
}

}  // extern "C"
