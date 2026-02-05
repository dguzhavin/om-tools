from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def _weighted_choice(rng: np.random.Generator, items: np.ndarray, probs: np.ndarray, size: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(items, size=size, replace=True, p=probs)


def _make_order_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Fast order ids: 'ord_' + 32 hex chars.
    Uses rng.bytes -> hex, no numpy string reduce, no huge integer bounds.
    """
    # 16 bytes per id => 32 hex chars
    raw = rng.bytes(n * 16)                     # bytes length = n*16
    # Convert to hex via .hex() on bytes is scalar, so instead use built-in binascii
    import binascii

    hex_bytes = binascii.hexlify(raw)           # length = n*32 (ascii)
    # Split into fixed-size chunks of 32 bytes
    hex_arr = np.frombuffer(hex_bytes, dtype="S32")  # shape (n,), each is 32-byte ASCII
    # Build final ids as object strings (robust across numpy unicode quirks)
    return (b"ord_" + hex_arr).astype("S36").astype(str)


def _date_range_sample(
    rng: np.random.Generator,
    n: int,
    start: str,
    end: str,
    monthly_strength: float,
    weekend_uplift: float,
) -> pd.DatetimeIndex:
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    if end_dt < start_dt:
        raise ValueError("end_date must be >= start_date")

    all_days = pd.date_range(start_dt, end_dt, freq="D")
    months = all_days.month.values

    # Month weights: smooth sinusoid + small structured bumps, then controlled by monthly_strength
    base = np.ones(len(all_days), dtype=float)
    seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (months - 1) / 12.0)  # deterministic shape
    # add deterministic month-specific bump pattern (not random -> repeatable)
    bump_by_month = np.array([1.06, 1.00, 0.98, 1.02, 1.04, 0.97, 0.95, 0.98, 1.03, 1.05, 1.10, 1.15])
    seasonal *= bump_by_month[months - 1]

    # weekend uplift into day weights
    dow = all_days.dayofweek.values  # Mon=0..Sun=6
    is_weekend = (dow >= 5).astype(float)
    weekend_w = 1.0 + weekend_uplift * is_weekend

    # Mix seasonal with base using strength
    w = base * (1.0 - monthly_strength) + seasonal * monthly_strength
    w *= weekend_w
    w = np.clip(w, 1e-6, None)
    w /= w.sum()

    sampled_days = rng.choice(all_days.values, size=n, replace=True, p=w)
    return pd.to_datetime(sampled_days)


def _lognormal_mixture(
    rng: np.random.Generator,
    n: int,
    means: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    # Mixture over K components. Choose component, then sample lognormal per-row.
    k = len(weights)
    comp = rng.choice(np.arange(k), size=n, p=weights / weights.sum())
    mu = means[comp]
    sigma = sigmas[comp]
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def _heavy_tail_qty(rng: np.random.Generator, n: int) -> np.ndarray:
    # Mixture: mostly small quantities + occasional bulk orders (heavy tail)
    u = rng.random(n)
    qty = np.empty(n, dtype=np.int32)
    # 85%: shifted geometric-like
    m1 = u < 0.85
    qty[m1] = 1 + rng.negative_binomial(n=2, p=0.55, size=m1.sum()).astype(np.int32)
    # 14%: medium bulk (lognormal)
    m2 = (u >= 0.85) & (u < 0.99)
    qty[m2] = np.maximum(1, np.rint(rng.lognormal(mean=2.0, sigma=0.55, size=m2.sum()))).astype(np.int32)
    # 1%: very large (pareto-ish)
    m3 = u >= 0.99
    qty[m3] = np.maximum(1, np.rint((rng.pareto(a=2.5, size=m3.sum()) + 1.0) * 80)).astype(np.int32)
    return qty


def _beta_spike_discount(rng: np.random.Generator, n: int, base_rate: float) -> np.ndarray:
    """
    Discount pct in [0, 0.7], with mass near 0 and rare large discounts.
    base_rate: overall probability of "has discount".
    """
    has_disc = rng.random(n) < base_rate
    d = np.zeros(n, dtype=np.float64)

    # Among discounted:
    # 80% small (0-20%), 18% medium (10-40%), 2% big (35-70%)
    u = rng.random(n)
    idx = np.where(has_disc)[0]
    if idx.size:
        u_d = u[idx]
        small = u_d < 0.80
        medium = (u_d >= 0.80) & (u_d < 0.98)
        big = u_d >= 0.98

        d_idx = idx[small]
        if d_idx.size:
            d[d_idx] = 0.20 * rng.beta(a=1.2, b=6.0, size=d_idx.size)

        d_idx = idx[medium]
        if d_idx.size:
            d[d_idx] = 0.10 + 0.30 * rng.beta(a=2.0, b=3.5, size=d_idx.size)

        d_idx = idx[big]
        if d_idx.size:
            d[d_idx] = 0.35 + 0.35 * rng.beta(a=2.2, b=1.8, size=d_idx.size)

    return np.clip(d, 0.0, 0.70)


def _safe_to_parquet(df: pd.DataFrame, path: str) -> None:
    # Prefer pyarrow if available; otherwise use pandas default engine (may require fastparquet/pyarrow).
    try:
        import pyarrow  # noqa: F401

        df.to_parquet(path, index=False, engine="pyarrow")
    except Exception:
        # fallback: try pandas default; if it fails, user will see the exception
        df.to_parquet(path, index=False)


def _write_csv_in_chunks(df: pd.DataFrame, path: str, chunk_size: int) -> None:
    # Chunk writing without extra copies: slice views
    header = True
    n = len(df)
    for i in range(0, n, chunk_size):
        df.iloc[i : i + chunk_size].to_csv(path, mode="w" if header else "a", index=False, header=header)
        header = False


def generate_sales_dataset(
    n_rows: int = 300_000,
    seed: int | None = 42,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    n_customers: int = 50_000,
    n_products: int = 2_000,
    n_reps: int = 300,
    n_regions: int = 12,
    currency: str = "USD",
    out_path: str | None = None,
    out_format: Literal["parquet", "csv"] = "parquet",
    csv_chunk_size: int = 100_000,
    missing_rate: float = 0.02,
    refund_rate: float = 0.01,
    outlier_rate: float = 0.003,
    discount_rate: float = 0.18,
    promo_rate: float = 0.12,
    weekend_uplift: float = 0.06,
    monthly_seasonality_strength: float = 0.25,
    price_noise: float = 0.08,
) -> pd.DataFrame:
    """
    Generate synthetic but realistic sales tabular data.

    Key realism features:
      - date seasonality (month weights) + weekend uplift
      - heavy-tail qty, lognormal/gamma-like price tails via mixture
      - segment/channel/stage correlations (conversion, discount, cycle)
      - rare outliers, missing values (notes + some non-key fields)
      - refunds producing negative net_revenue (consistent rule)

    Returns a DataFrame. If out_path is provided, also saves to parquet/csv.

    Example:
        df = generate_sales_dataset(n_rows=300_000, seed=7, out_path="sales.parquet", out_format="parquet")
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be > 0")
    if n_customers <= 0 or n_products <= 0 or n_reps <= 0 or n_regions <= 0:
        raise ValueError("n_customers/n_products/n_reps/n_regions must be > 0")
    if not (0.0 <= missing_rate <= 0.5):
        raise ValueError("missing_rate must be in [0, 0.5]")
    if not (0.0 <= refund_rate <= 0.2):
        raise ValueError("refund_rate must be in [0, 0.2]")
    if not (0.0 <= outlier_rate <= 0.05):
        raise ValueError("outlier_rate must be in [0, 0.05]")
    if not (0.0 <= discount_rate <= 1.0):
        raise ValueError("discount_rate must be in [0, 1]")
    if not (0.0 <= promo_rate <= 1.0):
        raise ValueError("promo_rate must be in [0, 1]")
    if not (0.0 <= monthly_seasonality_strength <= 1.0):
        raise ValueError("monthly_seasonality_strength must be in [0, 1]")
    if price_noise < 0:
        raise ValueError("price_noise must be >= 0")

    rng = np.random.default_rng(seed)

    n = n_rows

    # ---- IDs / categorical dims
    customer_id = rng.integers(1, n_customers + 1, size=n, dtype=np.int32)
    product_id = rng.integers(1, n_products + 1, size=n, dtype=np.int32)
    sales_rep_id = rng.integers(1, n_reps + 1, size=n, dtype=np.int32)

    regions = np.array([f"R{r:02d}" for r in range(1, n_regions + 1)], dtype=object)
    # Slight skew: a few large regions
    region_probs = np.linspace(1.8, 0.6, n_regions)
    region = _weighted_choice(rng, regions, region_probs, n)

    channel_items = np.array(["inbound", "outbound", "partner", "self_serve"], dtype=object)
    # Weekend effect later will slightly shift channels; base here:
    channel = _weighted_choice(rng, channel_items, np.array([0.34, 0.30, 0.16, 0.20]), n)

    lead_items = np.array(["ads", "referral", "events", "content", "cold"], dtype=object)
    lead_source = _weighted_choice(rng, lead_items, np.array([0.30, 0.18, 0.12, 0.22, 0.18]), n)

    seg_items = np.array(["SMB", "MM", "ENT"], dtype=object)
    segment = _weighted_choice(rng, seg_items, np.array([0.72, 0.22, 0.06]), n)

    # ---- Dates with seasonality
    order_date = _date_range_sample(
        rng=rng,
        n=n,
        start=start_date,
        end=end_date,
        monthly_strength=monthly_seasonality_strength,
        weekend_uplift=weekend_uplift,
    )
    dow = order_date.dayofweek.to_numpy()
    is_weekend = dow >= 5

    # ---- Adjust channel mix on weekends: more self_serve, less outbound
    # (Do a small probabilistic "flip" for weekend rows)
    flip = is_weekend & (rng.random(n) < 0.18)
    if flip.any():
        # For flipped weekend rows, re-draw with weekend-specific weights
        channel[flip] = _weighted_choice(rng, channel_items, np.array([0.30, 0.18, 0.15, 0.37]), flip.sum())

    # ---- Pipeline / conversion modeling
    # Base win probability by channel & segment (ENT slightly higher ACV but lower win rate)
    # Values are synthetic, deterministic constants (not "real facts").
    base_win_by_channel = {"inbound": 0.26, "outbound": 0.16, "partner": 0.22, "self_serve": 0.20}
    seg_mult = {"SMB": 1.05, "MM": 1.00, "ENT": 0.85}

    win_p = np.empty(n, dtype=np.float64)
    for ch in base_win_by_channel:
        m = channel == ch
        if m.any():
            win_p[m] = base_win_by_channel[ch]
    for s in seg_mult:
        m = segment == s
        if m.any():
            win_p[m] *= seg_mult[s]

    # Lead source influence (referral slightly better, cold slightly worse)
    ls_adj = {"ads": 1.00, "referral": 1.18, "events": 1.08, "content": 1.04, "cold": 0.88}
    for ls in ls_adj:
        m = lead_source == ls
        if m.any():
            win_p[m] *= ls_adj[ls]

    # Keep in bounds
    win_p = np.clip(win_p, 0.02, 0.60)
    is_won = rng.random(n) < win_p

    stage_items = np.array(["lead", "qualified", "proposal", "negotiation", "won", "lost"], dtype=object)
    # Stage distribution conditioned on won/lost with small noise
    pipeline_stage = np.empty(n, dtype=object)

    won_idx = np.where(is_won)[0]
    lost_idx = np.where(~is_won)[0]

    if won_idx.size:
        # mostly "won", some "negotiation/proposal" noise
        pipeline_stage[won_idx] = _weighted_choice(
            rng,
            stage_items,
            np.array([0.02, 0.06, 0.14, 0.18, 0.58, 0.02]),
            won_idx.size,
        )
    if lost_idx.size:
        pipeline_stage[lost_idx] = _weighted_choice(
            rng,
            stage_items,
            np.array([0.18, 0.20, 0.20, 0.15, 0.02, 0.25]),
            lost_idx.size,
        )

    # enforce stronger coherence with small correction
    # If won but stage is "lost" (rare), flip stage to "won" for most of them
    incoh = is_won & (pipeline_stage == "lost")
    if incoh.any():
        fix = incoh & (rng.random(n) < 0.85)
        pipeline_stage[fix] = "won"
    incoh2 = (~is_won) & (pipeline_stage == "won")
    if incoh2.any():
        fix = incoh2 & (rng.random(n) < 0.85)
        pipeline_stage[fix] = "lost"

    # ---- Quantity (heavy tail) + outliers
    qty = _heavy_tail_qty(rng, n)

    # controlled outliers: rare very large quantities
    if outlier_rate > 0:
        om = rng.random(n) < outlier_rate
        if om.any():
            qty[om] = np.maximum(qty[om], np.rint((rng.pareto(a=2.0, size=om.sum()) + 1) * 250)).astype(np.int32)

    # ---- Product base price levels (generated once, reused by product_id)
    # Mixture over product tiers -> not normal, heavy tail
    tier_weights = np.array([0.76, 0.20, 0.04])
    tier_means = np.array([2.6, 3.4, 4.2])   # log-space means
    tier_sigmas = np.array([0.45, 0.55, 0.65])
    product_base_price = _lognormal_mixture(rng, n_products, tier_means, tier_sigmas, tier_weights)

    # Segment multipliers (ENT higher)
    seg_price_mult = np.where(segment == "SMB", 0.95, np.where(segment == "MM", 1.10, 1.45))

    # Per-row unit_price: product_base * seg_mult * noise (lognormal noise to avoid normality)
    base = product_base_price[product_id - 1]
    # noise: mean 0 in log-space; sigma controlled by price_noise
    noise = rng.lognormal(mean=0.0, sigma=price_noise, size=n)
    unit_price = base * seg_price_mult * noise

    # Additional outliers: rare very high unit_price
    if outlier_rate > 0:
        om2 = rng.random(n) < (outlier_rate * 0.7)
        if om2.any():
            unit_price[om2] *= rng.lognormal(mean=1.2, sigma=0.6, size=om2.sum())

    # keep positive; round to cents later (after discount logic)
    unit_price = np.clip(unit_price, 0.5, None)

    # ---- Discount: base spike + stage/segment/size effects + promos
    discount_pct = _beta_spike_discount(rng, n, base_rate=discount_rate)

    is_promo = rng.random(n) < promo_rate
    # promo adds small extra discount, mostly for SMB/MM
    promo_add = np.zeros(n, dtype=np.float64)
    promo_add[is_promo] = 0.03 + 0.10 * rng.beta(a=1.8, b=6.5, size=is_promo.sum())
    promo_add *= np.where(segment == "ENT", 0.55, 1.0)
    discount_pct = np.clip(discount_pct + promo_add, 0.0, 0.70)

    # Negotiation stage -> higher discount; ENT -> higher discount; high gross amount/qty -> higher discount
    stage_mult = np.where(pipeline_stage == "negotiation", 1.35, np.where(pipeline_stage == "proposal", 1.12, 1.0))
    seg_disc_add = np.where(segment == "ENT", 0.03, np.where(segment == "MM", 0.01, 0.0))

    gross_amount = qty.astype(np.float64) * unit_price
    # scale effect (log1p) to avoid linear explosion
    size_add = 0.02 * np.tanh(np.log1p(gross_amount) / 10.0) + 0.015 * np.tanh(np.log1p(qty) / 4.0)
    discount_pct = np.clip(discount_pct * stage_mult + seg_disc_add + size_add, 0.0, 0.70)
    # Round discount to 4 decimals before revenue calc to keep contract stable
    discount_pct = np.round(discount_pct, 4)

    # ---- Payment terms
    terms_items = np.array([0, 7, 14, 30, 60, 90], dtype=np.int16)
    # Segment shifts: ENT more net60/90, self_serve more immediate
    base_terms_probs = np.array([0.20, 0.12, 0.18, 0.35, 0.10, 0.05], dtype=float)

    payment_terms_days = np.empty(n, dtype=np.int16)
    # draw per segment with adjusted probs (vectorized by masks)
    for s in ("SMB", "MM", "ENT"):
        m = segment == s
        if not m.any():
            continue
        p = base_terms_probs.copy()
        if s == "SMB":
            p *= np.array([1.25, 1.10, 1.05, 0.95, 0.70, 0.55])
        elif s == "MM":
            p *= np.array([0.95, 1.00, 1.05, 1.05, 1.15, 1.10])
        else:  # ENT
            p *= np.array([0.55, 0.70, 0.85, 1.15, 1.50, 1.65])
        payment_terms_days[m] = _weighted_choice(rng, terms_items, p, m.sum()).astype(np.int16)

    # Additional channel influence: self_serve -> more 0/7; partner -> more 30/60
    m_ss = channel == "self_serve"
    if m_ss.any():
        tweak = m_ss & (rng.random(n) < 0.35)
        if tweak.any():
            payment_terms_days[tweak] = _weighted_choice(
                rng, terms_items, np.array([0.55, 0.20, 0.12, 0.10, 0.02, 0.01]), tweak.sum()
            ).astype(np.int16)

    # ---- Days to close (skewed), depends on stage/channel/segment
    # Use gamma for skew + add deterministic offsets; for self_serve often small
    base_close = rng.gamma(shape=2.2, scale=7.5, size=n)  # positive skew
    ch_add = np.zeros(n, dtype=np.float64)
    ch_add[channel == "inbound"] += 4
    ch_add[channel == "outbound"] += 10
    ch_add[channel == "partner"] += 8
    ch_add[channel == "self_serve"] += 1

    seg_add = np.where(segment == "SMB", 0.0, np.where(segment == "MM", 6.0, 18.0))
    stage_add = np.where(
        pipeline_stage == "lead",
        0.0,
        np.where(
            pipeline_stage == "qualified",
            4.0,
            np.where(pipeline_stage == "proposal", 10.0, np.where(pipeline_stage == "negotiation", 16.0, 12.0)),
        ),
    )
    # Some deals close fast even in higher stages (noise)
    fast = rng.random(n) < 0.06
    base_close[fast] *= 0.35

    days_to_close = np.maximum(0, np.rint(base_close + ch_add + seg_add + stage_add)).astype(np.int32)

    # ---- Revenue + COGS + refunds
    # Round unit_price before revenue calc to keep contract stable with stored values
    unit_price = np.round(unit_price, 2)
    net_revenue = qty.astype(np.float64) * unit_price * (1.0 - discount_pct)

    # Round to cents with minor "non-ideal" rounding noise that doesn't break the contract:
    # Do rounding, then add a tiny epsilon and re-round.
    net_revenue = np.round(net_revenue, 2)
    eps = (rng.random(n) - 0.5) * 0.02  # +/- 1 cent
    net_revenue = np.round(np.maximum(net_revenue + eps, 0.0), 2)

    # Refunds: rule = if is_refund True, net_revenue becomes negative (refund amount equals prior net)
    is_refund = rng.random(n) < refund_rate
    # refunds more likely on won deals (returns/cancellations); push a bit:
    if is_refund.any():
        bump = is_refund & (~is_won) & (rng.random(n) < 0.55)
        # convert some non-won refunds to non-refund
        is_refund[bump] = False

    # Apply refund: negative or zero; choose negative to satisfy rule
    net_revenue_ref = net_revenue.copy()
    net_revenue_ref[is_refund] = -net_revenue_ref[is_refund]
    net_revenue = net_revenue_ref

    # COGS: correlated with unit_price and qty, with variability; usually <= net on won deals.
    # Use a cost ratio around 0.55-0.80 with heavy-ish tails.
    base_ratio = 0.52 + 0.30 * rng.beta(a=2.0, b=2.8, size=n)  # bounded, not normal
    # ENT tends to have slightly better margins sometimes, but with variance
    base_ratio *= np.where(segment == "ENT", 0.97, 1.0)
    # self_serve tends to have better margins (less sales cost proxy), partner slightly worse
    base_ratio *= np.where(channel == "self_serve", 0.96, np.where(channel == "partner", 1.03, 1.0))
    base_ratio = np.clip(base_ratio, 0.35, 0.95)

    # For refunds, set cogs to 0 or small negative? Keep it 0 to avoid confusing accounting.
    cogs = (np.maximum(net_revenue, 0.0) * base_ratio)
    cogs = np.round(cogs, 2)

    # Inject rare "bad deals" where cogs > net_revenue for won=True (controlled ~0.5% of won)
    bad = is_won & (~is_refund) & (rng.random(n) < 0.005)
    if bad.any():
        cogs[bad] = np.round(np.maximum(net_revenue[bad], 0.0) * (1.02 + 0.25 * rng.random(bad.sum())), 2)

    gross_margin = np.round(net_revenue - cogs, 2)

    # ---- Notes (templates + missing)
    notes_templates = np.array(
        [
            "Requested demo follow-up",
            "Pricing clarification needed",
            "Waiting on legal review",
            "Budget approval pending",
            "Competitor mentioned",
            "Trial activated",
            "Procurement involved",
            "Discount requested",
            "Stakeholder alignment call",
            "Contract sent",
        ],
        dtype=object,
    )
    notes = _weighted_choice(rng, notes_templates, np.array([0.12, 0.10, 0.08, 0.10, 0.06, 0.12, 0.08, 0.12, 0.10, 0.12]), n)

    # Make notes sparse + missing
    notes_empty = rng.random(n) < 0.35
    notes = notes.astype(object)
    notes[notes_empty] = ""
    notes_missing = rng.random(n) < missing_rate
    notes[notes_missing] = None  # becomes NaN in pandas

    # Some additional missing on lead_source or sales_rep_id (small; not breaking)
    lead_source = lead_source.astype(object)
    lead_miss = rng.random(n) < (missing_rate * 0.25)
    lead_source[lead_miss] = None

    # ---- Cohort month (YYYY-MM)
    cohort_month = order_date.to_period("M").astype(str)

    # ---- Build DataFrame with categoricals for memory
    df = pd.DataFrame(
        {
            "order_id": _make_order_ids(rng, n),
            "order_date": order_date,
            "customer_id": customer_id,
            "product_id": product_id,
            "sales_rep_id": sales_rep_id,
            "region": pd.Categorical(region, categories=regions),
            "channel": pd.Categorical(channel, categories=channel_items),
            "lead_source": pd.Categorical(lead_source, categories=lead_items),
            "pipeline_stage": pd.Categorical(pipeline_stage, categories=stage_items),
            "is_won": is_won.astype(bool),
            "customer_segment": pd.Categorical(segment, categories=seg_items),
            "qty": qty.astype(np.int32),
            "unit_price": unit_price,
            "discount_pct": discount_pct,
            "net_revenue": net_revenue.astype(np.float64),
            "cogs": cogs.astype(np.float64),
            "gross_margin": gross_margin.astype(np.float64),
            "payment_terms_days": payment_terms_days.astype(np.int16),
            "days_to_close": days_to_close.astype(np.int32),
            "is_refund": is_refund.astype(bool),
            "notes": notes,
            "cohort_month": pd.Categorical(cohort_month),
            "currency": pd.Categorical(np.full(n, currency, dtype=object)),
        }
    )

    # ---- Persist if requested
    if out_path is not None:
        if out_format == "parquet":
            _safe_to_parquet(df, out_path)
        elif out_format == "csv":
            _write_csv_in_chunks(df, out_path, csv_chunk_size)
        else:
            raise ValueError("out_format must be 'parquet' or 'csv'")

    return df


def validate_sales_dataset(df: pd.DataFrame) -> None:
    """
    Minimal integrity checks (raises AssertionError on failure).
    Designed to be fast enough for 300k rows.
    """
    required = {
        "order_id",
        "order_date",
        "customer_id",
        "product_id",
        "sales_rep_id",
        "region",
        "channel",
        "lead_source",
        "pipeline_stage",
        "is_won",
        "qty",
        "unit_price",
        "discount_pct",
        "net_revenue",
        "cogs",
        "gross_margin",
        "payment_terms_days",
        "days_to_close",
        "is_refund",
        "notes",
        "customer_segment",
        "cohort_month",
    }
    missing_cols = required - set(df.columns)
    assert not missing_cols, f"Missing columns: {missing_cols}"

    assert df["qty"].min() >= 1
    assert (df["unit_price"] > 0).all()
    assert (df["discount_pct"].between(0, 0.70)).all()
    assert (df["days_to_close"] >= 0).all()

    # Revenue contract: net_revenue equals rounded qty*unit_price*(1-discount), except refunds (negative).
    calc = np.round(df["qty"].to_numpy(dtype=float) * df["unit_price"].to_numpy(dtype=float) * (1.0 - df["discount_pct"].to_numpy(dtype=float)), 2)
    nr = df["net_revenue"].to_numpy(dtype=float)
    is_ref = df["is_refund"].to_numpy(dtype=bool)

    # For non-refunds: close match within 1 cent
    diff_non_ref = np.abs(nr[~is_ref] - calc[~is_ref])
    assert np.quantile(diff_non_ref, 0.999) <= 0.02, f"net_revenue mismatch too large (non-refunds): q99.9={np.quantile(diff_non_ref,0.999)}"

    # For refunds: net_revenue should be <= 0 and close to -calc within 1 cent (allow tiny noise)
    assert (nr[is_ref] <= 0).all()
    diff_ref = np.abs(nr[is_ref] + calc[is_ref])
    if diff_ref.size:
        assert np.quantile(diff_ref, 0.999) <= 0.02, f"net_revenue mismatch too large (refunds): q99.9={np.quantile(diff_ref,0.999)}"

    # gross_margin = net_revenue - cogs
    gm = df["gross_margin"].to_numpy(dtype=float)
    cogs = df["cogs"].to_numpy(dtype=float)
    diff_gm = np.abs(gm - np.round(nr - cogs, 2))
    assert np.quantile(diff_gm, 0.999) <= 0.02, f"gross_margin mismatch too large: q99.9={np.quantile(diff_gm,0.999)}"

    # Coherence: won implies stage is often 'won' (not always)
    is_won = df["is_won"].to_numpy(dtype=bool)
    stage = df["pipeline_stage"].astype(str).to_numpy()
    if is_won.any():
        share_won_stage = (stage[is_won] == "won").mean()
        assert share_won_stage >= 0.45, f"Too few won-stage among won deals: {share_won_stage:.3f}"
    if (~is_won).any():
        share_lost_stage = (stage[~is_won] == "lost").mean()
        assert share_lost_stage >= 0.18, f"Too few lost-stage among non-won deals: {share_lost_stage:.3f}"


if __name__ == "__main__":
    df = generate_sales_dataset(
        n_rows=300_000,
        seed=42,
        out_path="synthetic_fact_data.csv",  # e.g. "sales.parquet"
        out_format="csv", # parquet / csv
    )
    validate_sales_dataset(df)
    print(df.head())
    print(df.dtypes)
