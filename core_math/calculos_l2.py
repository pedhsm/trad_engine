# core_math/calculos_l2.py

def book_imbalance_signal(bids, asks, trigger_threshold=0.6):
    """
    Depth-weighted order-book imbalance signal (a simple, well-known L2 primitive).

    Computes the order-book pressure weighted by depth (levels 0 to 4): each level
    contributes with weight 1/(idx+1), so the top of book dominates.

    `bids` / `asks` are sequences of (price, volume) from best to worst.
    Returns: 'BUY', 'SELL' or None.
    """
    if not bids or not asks:
        return None

    # Depth-level weighting (level 0 = weight 1.0, level 1 = weight 0.5, etc.)
    weighted_bid_volume = 0.0
    for idx, (_, vol) in enumerate(bids):
        weight = 1.0 / (idx + 1)
        weighted_bid_volume += vol * weight

    weighted_ask_volume = 0.0
    for idx, (_, vol) in enumerate(asks):
        weight = 1.0 / (idx + 1)
        weighted_ask_volume += vol * weight

    total_volume = weighted_bid_volume + weighted_ask_volume
    if total_volume == 0:
        return None

    # Weighted imbalance: 0.0 (all sell) to 1.0 (all buy)
    imbalance = weighted_bid_volume / total_volume

    if imbalance >= trigger_threshold:
        return 'BUY'
    elif imbalance <= (1.0 - trigger_threshold):
        return 'SELL'

    return None
