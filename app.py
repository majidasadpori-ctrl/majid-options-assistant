import math
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

try:
    from tseopt import get_all_options_data
except ImportError:
    get_all_options_data = None


# =========================================================
# Brand / App Settings
# =========================================================

APP_BRAND = "دستیار معاملات آپشن مجید اسدپوری"
APP_NAME = "دستیار تصمیم‌گیری اختیار معامله ایران"
APP_VERSION = "MVP 1.1"
APP_OWNER = "مجید اسدپوری"

ALL_UNDERLYINGS_LABEL = "همه دارایی‌های پایه"

SCENARIO_MAP = {
    "خودکار بر اساس حرکت مورد انتظار": "auto",
    "صعودی": "bullish",
    "نزولی": "bearish",
    "خنثی": "neutral",
    "نامشخص": "unknown",
}

OPTION_TYPE_MAP = {
    "همه": "all",
    "فقط Call": "call",
    "فقط Put": "put",
}

VALID_POSITIVE_SIGNALS = ["خرید قابل بررسی", "خرید پرریسک", "فقط برای رصد"]


# =========================================================
# Streamlit Config / CSS
# =========================================================

st.set_page_config(
    page_title=f"{APP_BRAND} | {APP_NAME}",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
    .main-title {
        direction: rtl;
        text-align: right;
        font-size: 22px;
        font-weight: 800;
        line-height: 2.4;
        margin: 0 0 0.15rem 0;
        padding: 0;
        overflow: visible;
    }
    .subtle {
        direction: rtl;
        text-align: right;
        color:#666;
        font-size:13px;
        line-height:2;
        margin-bottom: 0.8rem;
    }
    .top-header {
        direction: rtl;
        text-align: right;
        padding: 1.0rem 0 0.9rem 0;
        margin-bottom: 0.4rem;
        overflow: visible;
    }
    .rtl-box {direction: rtl; text-align: right; padding: 14px; border-radius: 14px; border: 1px solid #e6e6e6; background-color: #fbfbfb; line-height: 2; margin: 10px 0;}
    .tip-box {direction: rtl; text-align: right; padding: 14px; border-radius: 14px; border: 1px solid #d9ecff; background-color: #f7fbff; line-height: 2; margin: 10px 0;}
    .rec-card {direction: rtl; text-align: right; padding: 14px; border-radius: 16px; background-color: #ffffff; border: 1px solid #e5e5e5; min-height: 150px; box-shadow: 0 1px 6px rgba(0,0,0,0.04); margin-bottom: 10px;}
    .rec-card b {font-size: 15px;}
    .buy {border-right: 6px solid #2e7d32;}
    .watch {border-right: 6px solid #f9a825;}
    .avoid {border-right: 6px solid #c62828;}
    .neutral {border-right: 6px solid #546e7a;}
    .decision-note {direction: rtl; text-align: right; line-height: 2;}
    [data-testid="stDataFrame"] {direction: ltr; text-align: left;}
    [data-testid="stDataFrame"] div {font-family: Tahoma, Arial, sans-serif;}
    .small-muted {direction: rtl; text-align:right; color:#777; font-size:12px; line-height:1.9;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Helpers
# =========================================================

def to_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(str(value).replace(",", "").strip())
    except Exception:
        return default


def clamp(x, low=0, high=100):
    try:
        if pd.isna(x):
            return low
        return max(low, min(high, float(x)))
    except Exception:
        return low


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame):
    required_cols = [
        "ua_ticker", "ua_close_price", "ua_last_price", "ua_yesterday_price",
        "strike_price", "end_date", "days_to_maturity", "last_price", "close_price",
        "trades_value", "trades_volume", "trades_num", "open_positions",
        "bid_price", "bid_volume", "ask_price", "ask_volume", "option_type", "ticker", "name",
    ]
    return [c for c in required_cols if c not in df.columns]


def format_number(x, decimals=0):
    try:
        if pd.isna(x):
            return "-"
        if decimals == 0:
            return f"{float(x):,.0f}"
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return str(x)


def resolve_scenario(scenario_choice, expected_move_pct):
    if scenario_choice != "خودکار بر اساس حرکت مورد انتظار":
        return SCENARIO_MAP[scenario_choice], scenario_choice
    if expected_move_pct > 0:
        return "bullish", "صعودی"
    if expected_move_pct < 0:
        return "bearish", "نزولی"
    return "neutral", "خنثی"


def estimate_working_days(calendar_days, working_ratio=5 / 7, holiday_buffer=0):
    calendar_days = to_float(calendar_days)
    if pd.isna(calendar_days) or calendar_days < 0:
        return np.nan
    working_days = math.floor(calendar_days * working_ratio) - int(holiday_buffer)
    return max(0, working_days)


# =========================================================
# Data Fetching
# =========================================================

@st.cache_data(ttl=60)
def load_options_data() -> pd.DataFrame:
    if get_all_options_data is None:
        raise RuntimeError("پکیج tseopt نصب نیست. ابتدا این دستور را اجرا کن: pip install --upgrade tseopt")
    try:
        data = get_all_options_data()
    except Exception as e:
        raise RuntimeError(
            "خطا در دریافت داده از tseopt / TSETMC. احتمالاً مشکل از اینترنت، DNS، VPN یا در دسترس نبودن cdn.tsetmc.com است. "
            f"جزئیات خطا: {e}"
        )
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if df.empty:
        raise RuntimeError("داده دریافت شد، اما خروجی خالی بود.")
    return normalize_columns(df)


def filter_underlying(df, selected_underlying):
    if selected_underlying == ALL_UNDERLYINGS_LABEL:
        return df.copy()
    return df[df["ua_ticker"].astype(str).str.contains(selected_underlying, na=False)].copy()


def filter_option_type(df, option_type_filter):
    if option_type_filter == "all":
        return df.copy()
    return df[df["option_type"].astype(str).str.lower().eq(option_type_filter)].copy()


# =========================================================
# Core Option Logic
# =========================================================

def is_put_option(row):
    return str(row.get("option_type", "")).strip().lower() == "put"


def is_call_option(row):
    return str(row.get("option_type", "")).strip().lower() == "call"


def get_market_price(row):
    last_price = to_float(row.get("last_price"))
    close_price = to_float(row.get("close_price"))
    if pd.notna(last_price) and last_price > 0:
        return last_price
    if pd.notna(close_price) and close_price > 0:
        return close_price
    return np.nan


def get_entry_price(row):
    ask_price = to_float(row.get("ask_price"))
    market_price = get_market_price(row)
    if pd.notna(ask_price) and ask_price > 0:
        return ask_price
    return market_price


def calc_intrinsic(row):
    s = to_float(row.get("ua_close_price"))
    k = to_float(row.get("strike_price"))
    if pd.isna(s) or pd.isna(k):
        return np.nan
    if is_put_option(row):
        return max(k - s, 0)
    return max(s - k, 0)


def calc_breakeven(row):
    k = to_float(row.get("strike_price"))
    entry_price = get_entry_price(row)
    if pd.isna(k) or pd.isna(entry_price):
        return np.nan
    if is_put_option(row):
        return k - entry_price
    return k + entry_price


def required_move_to_strike_pct(row):
    s = to_float(row.get("ua_close_price"))
    k = to_float(row.get("strike_price"))
    if pd.isna(s) or pd.isna(k) or s <= 0:
        return np.nan
    if is_put_option(row):
        if k >= s:
            return 0
        return round((s - k) / s * 100, 2)
    if k <= s:
        return 0
    return round((k / s - 1) * 100, 2)


def required_move_to_breakeven_pct(row):
    s = to_float(row.get("ua_close_price"))
    breakeven = to_float(row.get("breakeven"))
    if pd.isna(s) or pd.isna(breakeven) or s <= 0:
        return np.nan
    if is_put_option(row):
        if breakeven >= s:
            return 0
        return round((s - breakeven) / s * 100, 2)
    if breakeven <= s:
        return 0
    return round((breakeven / s - 1) * 100, 2)


def days_needed_to_breakeven(row, daily_limit=0.03):
    s = to_float(row.get("ua_close_price"))
    breakeven = to_float(row.get("breakeven"))
    if pd.isna(s) or pd.isna(breakeven) or s <= 0 or breakeven <= 0:
        return np.nan
    if is_put_option(row):
        if breakeven >= s:
            return 0
        return math.ceil(math.log(breakeven / s) / math.log(1 - daily_limit))
    if breakeven <= s:
        return 0
    return math.ceil(math.log(breakeven / s) / math.log(1 + daily_limit))


# =========================================================
# Underlying Analysis
# =========================================================

def calc_underlying_day_change_pct(row):
    close_price = to_float(row.get("ua_close_price"))
    yesterday = to_float(row.get("ua_yesterday_price"))
    if pd.isna(close_price) or pd.isna(yesterday) or yesterday <= 0:
        return np.nan
    return round((close_price / yesterday - 1) * 100, 2)


def calc_underlying_intraday_pressure_pct(row):
    last_price = to_float(row.get("ua_last_price"))
    close_price = to_float(row.get("ua_close_price"))
    if pd.isna(last_price) or pd.isna(close_price) or close_price <= 0:
        return np.nan
    return round((last_price / close_price - 1) * 100, 2)


def calc_expected_underlying_price(row, expected_move_pct):
    s = to_float(row.get("ua_close_price"))
    if pd.isna(s) or s <= 0:
        return np.nan
    return round(s * (1 + expected_move_pct / 100), 2)


def calc_underlying_scenario_score(row, underlying_scenario):
    day = to_float(row.get("underlying_day_change_pct"), 0)
    pressure = to_float(row.get("underlying_intraday_pressure_pct"), 0)
    score = 50

    if underlying_scenario == "bullish":
        score += 22 if day > 0 else -18 if day < 0 else 0
        score += 22 if pressure > 0 else -18 if pressure < 0 else 0
    elif underlying_scenario == "bearish":
        score += 22 if day < 0 else -18 if day > 0 else 0
        score += 22 if pressure < 0 else -18 if pressure > 0 else 0
    elif underlying_scenario == "neutral":
        score = 85 if abs(day) <= 1 and abs(pressure) <= 0.5 else 45
    else:
        score = 50

    return round(clamp(score), 1)


def calc_underlying_status(row, underlying_scenario):
    score = to_float(row.get("underlying_scenario_score"), 50)
    day = to_float(row.get("underlying_day_change_pct"))
    pressure = to_float(row.get("underlying_intraday_pressure_pct"))

    if score >= 75:
        return "دارایی پایه با سناریو هم‌راستاست"
    if score >= 55:
        return "دارایی پایه نسبتاً قابل قبول است"
    if score >= 40:
        return "دارایی پایه ابهام دارد"
    if pd.notna(day) and pd.notna(pressure):
        return "دارایی پایه خلاف سناریو حرکت می‌کند"
    return "داده کافی برای تحلیل دارایی پایه نیست"


# =========================================================
# Option Order Book / Factor Engineering
# =========================================================

def calc_bid_ask_spread_pct(row):
    bid = to_float(row.get("bid_price"))
    ask = to_float(row.get("ask_price"))
    if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0 or ask < bid:
        return np.nan
    mid = (bid + ask) / 2
    if mid <= 0:
        return np.nan
    return round((ask - bid) / mid * 100, 2)


def calc_spread_score(row):
    spread = to_float(row.get("bid_ask_spread_pct"))
    if pd.isna(spread):
        return 20
    if spread <= 3:
        return 100
    if spread <= 7:
        return 80
    if spread <= 15:
        return 55
    if spread <= 30:
        return 30
    return 10


def calc_entry_depth_score(row, config):
    ask_volume = to_float(row.get("ask_volume"), 0)
    min_depth = config["min_depth"]
    return round(min(100, 100 * ask_volume / min_depth), 1) if min_depth > 0 else 0


def calc_exit_depth_score(row, config):
    bid_volume = to_float(row.get("bid_volume"), 0)
    min_depth = config["min_depth"]
    return round(min(100, 100 * bid_volume / min_depth), 1) if min_depth > 0 else 0


def calc_depth_balance_score(row):
    bid_volume = to_float(row.get("bid_volume"), 0)
    ask_volume = to_float(row.get("ask_volume"), 0)
    if bid_volume <= 0 or ask_volume <= 0:
        return 0
    return round(100 * min(bid_volume, ask_volume) / max(bid_volume, ask_volume), 1)


def calc_orderbook_quality_score(row):
    """Decision-oriented order book score.

    In Iran options, a contract can look attractive on paper but be hard to enter or exit.
    Therefore missing ask or bid receives a hard cap, not just a small penalty.
    """
    bid = to_float(row.get("bid_price"))
    ask = to_float(row.get("ask_price"))
    bid_volume = to_float(row.get("bid_volume"), 0)
    ask_volume = to_float(row.get("ask_volume"), 0)

    score = (
        0.24 * to_float(row.get("entry_depth_score"), 0) +
        0.30 * to_float(row.get("exit_depth_score"), 0) +
        0.16 * to_float(row.get("depth_balance_score"), 0) +
        0.30 * to_float(row.get("spread_score"), 0)
    )

    if pd.isna(ask) or ask <= 0 or ask_volume <= 0:
        score = min(score, 25)
    if pd.isna(bid) or bid <= 0 or bid_volume <= 0:
        score = min(score, 22)
    if (pd.isna(bid) or bid <= 0 or bid_volume <= 0) and (pd.isna(ask) or ask <= 0 or ask_volume <= 0):
        score = min(score, 10)
    return round(clamp(score), 1)


def calc_orderbook_label(row):
    score = to_float(row.get("orderbook_quality_score"), 0)
    if score >= 75:
        return "اردربوک مناسب"
    if score >= 55:
        return "اردربوک قابل قبول"
    if score >= 35:
        return "اردربوک ضعیف"
    return "اردربوک نامناسب"


def calc_exit_capability_score(row):
    """How easy it is to exit a long option position using the current bid side."""
    bid = to_float(row.get("bid_price"))
    bid_volume = to_float(row.get("bid_volume"), 0)
    spread_score = to_float(row.get("spread_score"), 0)
    exit_depth_score = to_float(row.get("exit_depth_score"), 0)
    liquidity = to_float(row.get("liquidity_score"), 0)
    if pd.isna(bid) or bid <= 0 or bid_volume <= 0:
        return 0
    return round(clamp(0.45 * exit_depth_score + 0.30 * spread_score + 0.25 * liquidity), 1)


def calc_exit_capability_label(row):
    score = to_float(row.get("exit_capability_score"), 0)
    if score >= 75:
        return "خروج نسبتاً راحت"
    if score >= 55:
        return "خروج قابل قبول"
    if score >= 35:
        return "خروج پرریسک"
    return "خروج دشوار"


def calc_moneyness_pct(row):
    s = to_float(row.get("ua_close_price"))
    k = to_float(row.get("strike_price"))
    if pd.isna(s) or pd.isna(k) or s <= 0:
        return np.nan
    if is_put_option(row):
        return round((k - s) / s * 100, 2)
    return round((s - k) / s * 100, 2)


def calc_moneyness_label(row, atm_threshold=3, deep_threshold=15):
    m = to_float(row.get("moneyness_pct"))
    if pd.isna(m):
        return "نامشخص"
    if abs(m) <= atm_threshold:
        return "لب مرز اعمال"
    if m > atm_threshold:
        return "دارای ارزش ذاتی"
    if m < -deep_threshold:
        return "خیلی دور از ارزش ذاتی"
    return "بدون ارزش ذاتی"


def calc_moneyness_score(row):
    m = abs(to_float(row.get("moneyness_pct")))
    if pd.isna(m):
        return 0
    if m <= 3:
        return 100
    if m <= 7:
        return 85
    if m <= 12:
        return 65
    if m <= 20:
        return 35
    return 15


def calc_simple_leverage(row):
    s = to_float(row.get("ua_close_price"))
    entry = to_float(row.get("entry_price"))
    if pd.isna(s) or pd.isna(entry) or entry <= 0:
        return np.nan
    return round(s / entry, 2)


def calc_risk_bucket(row):
    leverage = to_float(row.get("simple_leverage"))
    move_be = to_float(row.get("required_move_to_breakeven_pct"))
    if pd.isna(leverage) or pd.isna(move_be):
        return "نامشخص"
    if leverage <= 4 and move_be <= 5:
        return "ریسک کمتر"
    if leverage <= 10 and move_be <= 12:
        return "ریسک متعادل"
    return "ریسک بالا / اهرمی"


def calc_premium_ratio_pct(row):
    entry = to_float(row.get("entry_price"))
    s = to_float(row.get("ua_close_price"))
    if pd.isna(entry) or pd.isna(s) or s <= 0:
        return np.nan
    return round(entry / s * 100, 2)


def calc_time_value_ratio_pct(row):
    time_value = to_float(row.get("time_value"))
    entry = to_float(row.get("entry_price"))
    if pd.isna(time_value) or pd.isna(entry) or entry <= 0:
        return np.nan
    return round(time_value / entry * 100, 2)


def calc_premium_risk_score(row):
    ratio = to_float(row.get("premium_ratio_pct"))
    if pd.isna(ratio) or ratio <= 0:
        return 0
    if 2 <= ratio <= 8:
        return 100
    if 1 <= ratio < 2:
        return 70
    if 8 < ratio <= 15:
        return 75
    if 15 < ratio <= 25:
        return 45
    return 25


def calc_volume_to_oi_ratio_pct(row):
    volume = to_float(row.get("trades_volume"))
    oi = to_float(row.get("open_positions"))
    if pd.isna(volume) or pd.isna(oi) or oi <= 0:
        return np.nan
    return round(volume / oi * 100, 2)


def calc_price_reliability_score(row, config):
    score = 0
    bid = to_float(row.get("bid_price"))
    ask = to_float(row.get("ask_price"))
    last = to_float(row.get("last_price"))
    close = to_float(row.get("close_price"))
    trades_num = to_float(row.get("trades_num"), 0)

    if pd.notna(last) and last > 0:
        score += 20
    if pd.notna(close) and close > 0:
        score += 15
    if pd.notna(bid) and bid > 0:
        score += 20
    if pd.notna(ask) and ask > 0:
        score += 20
    if trades_num >= config["min_trades_num"]:
        score += 15
    if pd.notna(last) and pd.notna(close) and last > 0 and close > 0:
        diff = abs(last / close - 1)
        if diff <= 0.05:
            score += 10
        elif diff <= 0.10:
            score += 5
    return min(100, score)


def calc_expiry_quality_score(row, config):
    wd = to_float(row.get("working_days_to_maturity"))
    min_days = config["min_days_to_maturity"]
    if pd.isna(wd):
        return 0
    if wd < min_days:
        return 0
    if 5 <= wd <= 30:
        return 100
    if 31 <= wd <= 60:
        return 80
    if 61 <= wd <= 90:
        return 60
    if wd > 90:
        return 40
    return 50


def calc_alignment_score(row, underlying_scenario):
    option_type = str(row.get("option_type", "")).lower()
    if underlying_scenario == "bullish":
        return 100 if option_type == "call" else 0
    if underlying_scenario == "bearish":
        return 100 if option_type == "put" else 0
    if underlying_scenario == "neutral":
        return 50
    return 50


# =========================================================
# Scenario / Scores / Signals
# =========================================================

def calc_scenario_payoff(row):
    target = to_float(row.get("expected_underlying_price"))
    k = to_float(row.get("strike_price"))
    if pd.isna(target) or pd.isna(k):
        return np.nan
    if is_put_option(row):
        return max(k - target, 0)
    return max(target - k, 0)


def calc_scenario_pnl(row):
    payoff = to_float(row.get("scenario_payoff"))
    entry = to_float(row.get("entry_price"))
    if pd.isna(payoff) or pd.isna(entry):
        return np.nan
    return payoff - entry


def calc_scenario_return_pct(row):
    pnl = to_float(row.get("scenario_pnl"))
    entry = to_float(row.get("entry_price"))
    if pd.isna(pnl) or pd.isna(entry) or entry <= 0:
        return np.nan
    return round((pnl / entry) * 100, 2)


def calc_scenario_value_score(row):
    r = to_float(row.get("scenario_return_pct"))
    if pd.isna(r):
        return 0
    if r <= 0:
        return 0
    if r < 15:
        return 30
    if r < 40:
        return 60
    if r < 80:
        return 80
    return 100


def calc_path_score(row):
    working_days = to_float(row.get("working_days_to_maturity"))
    needed = to_float(row.get("days_needed_to_breakeven"))
    if pd.isna(needed) or pd.isna(working_days):
        return 0, "نامشخص"
    if working_days <= 0:
        return 0, "بدون روز کاری مؤثر"
    if needed > working_days:
        return 0, "مسیر تقریباً غیرممکن"
    ratio = needed / max(working_days, 1)
    if ratio <= 0.35:
        return 100, "مسیر قابل تحقق"
    if ratio <= 0.65:
        return 70, "مسیر متوسط"
    if ratio <= 0.90:
        return 40, "مسیر سخت"
    return 20, "مسیر بسیار فشرده"


def calc_liquidity_score(row, config):
    trades_value = to_float(row.get("trades_value"), 0)
    trades_volume = to_float(row.get("trades_volume"), 0)
    trades_num = to_float(row.get("trades_num"), 0)
    open_positions = to_float(row.get("open_positions"), 0)
    bid_volume = to_float(row.get("bid_volume"), 0)
    ask_volume = to_float(row.get("ask_volume"), 0)

    min_value = config["min_trades_value"]
    min_volume = config["min_trades_volume"]
    min_trades = config["min_trades_num"]
    min_oi = config["min_open_positions"]
    min_depth = config["min_depth"]

    score = 0
    score += min(30, 30 * trades_value / min_value) if min_value > 0 else 0
    score += min(20, 20 * trades_volume / min_volume) if min_volume > 0 else 0
    score += min(15, 15 * trades_num / min_trades) if min_trades > 0 else 0
    score += min(15, 15 * open_positions / min_oi) if min_oi > 0 else 0
    score += min(10, 10 * bid_volume / min_depth) if min_depth > 0 else 0
    score += min(10, 10 * ask_volume / min_depth) if min_depth > 0 else 0
    return round(score, 1)


def calc_execution_score(row):
    liquidity = to_float(row.get("liquidity_score"), 0)
    orderbook = to_float(row.get("orderbook_quality_score"), 0)
    reliability = to_float(row.get("price_reliability_score"), 0)
    score = 0.45 * liquidity + 0.40 * orderbook + 0.15 * reliability
    return round(score, 1)


def execution_risk_label(row):
    execution = to_float(row.get("execution_score"), 0)
    if execution >= 80:
        return "پایین"
    if execution >= 60:
        return "متوسط"
    if execution >= 35:
        return "بالا"
    return "بسیار بالا"


def calc_opportunity_score(row):
    """Final opportunity score, tuned for trading decisions rather than pure pricing."""
    score = (
        0.14 * to_float(row.get("path_score"), 0) +
        0.16 * to_float(row.get("scenario_value_score"), 0) +
        0.11 * to_float(row.get("underlying_scenario_score"), 0) +
        0.13 * to_float(row.get("orderbook_quality_score"), 0) +
        0.09 * to_float(row.get("liquidity_score"), 0) +
        0.07 * to_float(row.get("execution_score"), 0) +
        0.07 * to_float(row.get("exit_capability_score"), 0) +
        0.06 * to_float(row.get("moneyness_score"), 0) +
        0.04 * to_float(row.get("price_reliability_score"), 0) +
        0.04 * to_float(row.get("expiry_quality_score"), 0) +
        0.02 * to_float(row.get("premium_risk_score"), 0) +
        0.07 * to_float(row.get("alignment_score"), 0)
    )
    return round(clamp(score), 1)


def final_signal(row, underlying_scenario, config):
    option_type = str(row.get("option_type", "")).lower()
    working_days = to_float(row.get("working_days_to_maturity"))
    path_score = to_float(row.get("path_score"), 0)
    liquidity_score = to_float(row.get("liquidity_score"), 0)
    scenario_value_score = to_float(row.get("scenario_value_score"), 0)
    opportunity_score = to_float(row.get("opportunity_score"), 0)
    move_to_breakeven = to_float(row.get("required_move_to_breakeven_pct"))
    orderbook_score = to_float(row.get("orderbook_quality_score"), 0)
    underlying_score = to_float(row.get("underlying_scenario_score"), 0)
    exit_score = to_float(row.get("exit_capability_score"), 0)
    bid = to_float(row.get("bid_price"))
    ask = to_float(row.get("ask_price"))
    bid_volume = to_float(row.get("bid_volume"), 0)
    ask_volume = to_float(row.get("ask_volume"), 0)

    if pd.isna(working_days) or working_days <= 0:
        return "عدم معامله: روز کاری مؤثر باقی نمانده"
    if working_days < config["min_days_to_maturity"]:
        return "عدم معامله: زمان کاری تا سررسید بسیار کم است"
    if pd.isna(ask) or ask <= 0 or ask_volume <= 0:
        return "عدم معامله: Ask فعال برای ورود ندارد"
    if pd.isna(bid) or bid <= 0 or bid_volume <= 0:
        return "عدم معامله: Bid فعال برای خروج ندارد"
    if underlying_scenario == "bullish" and option_type == "put":
        return "عدم معامله: سناریوی صعودی با Put هم‌راستا نیست"
    if underlying_scenario == "bearish" and option_type == "call":
        return "عدم معامله: سناریوی نزولی با Call هم‌راستا نیست"
    if underlying_scenario == "neutral":
        if path_score >= 70 and liquidity_score >= 60 and orderbook_score >= 55:
            return "کاندید بررسی دستی: سناریو خنثی است"
        return "عدم معامله: سناریوی دارایی پایه خنثی است"
    if underlying_scenario == "unknown":
        return "کاندید بررسی دستی: سناریو نامشخص است"
    if underlying_score < 35:
        return "عدم معامله: دارایی پایه خلاف سناریو است"
    if path_score == 0:
        return "عدم معامله: مسیر دارایی پایه توجیه ندارد"
    if pd.notna(move_to_breakeven) and move_to_breakeven > config["max_breakeven_move_pct"]:
        return "فقط برای رصد: فاصله تا Breakeven زیاد است"
    if exit_score < 25:
        return "عدم معامله: قابلیت خروج ضعیف است"
    if liquidity_score < 30 or orderbook_score < 30:
        return "جذاب اما غیرقابل اجرا"
    if scenario_value_score == 0:
        return "ارزان اما بی‌سناریو"
    if opportunity_score >= 74 and path_score >= 60 and liquidity_score >= 45 and orderbook_score >= 50 and exit_score >= 45:
        return "خرید قابل بررسی"
    if opportunity_score >= 58 and liquidity_score >= 30 and orderbook_score >= 35 and exit_score >= 30:
        return "خرید پرریسک"
    if opportunity_score >= 45:
        return "فقط برای رصد"
    return "نیازمند بررسی دستی"

def calc_trade_action(row):
    signal = str(row.get("final_signal", ""))
    score = to_float(row.get("opportunity_score"), 0)

    if signal == "خرید قابل بررسی":
        return "قابل خرید با کنترل ریسک"

    if signal == "خرید پرریسک":
        return "خرید فقط کم‌حجم و پله‌ای"

    if signal.startswith("فقط برای رصد"):
        return "فعلاً خرید نکن؛ فقط زیر نظر بگیر"

    if signal in ["جذاب اما غیرقابل اجرا"]:
        return "عدم خرید تا بهتر شدن اردربوک"

    if "Ask فعال" in signal or "Bid فعال" in signal or "قابلیت خروج" in signal:
        return "عدم خرید تا بهتر شدن اردربوک"

    if signal == "ارزان اما بی‌سناریو":
        return "عدم خرید؛ سناریو سود نمی‌دهد"

    if signal.startswith("عدم معامله"):
        return "عدم معامله"

    if signal.startswith("کاندید بررسی دستی") or score >= 50:
        return "کاندید بررسی دستی"

    return "عدم معامله"


def calc_holder_action(row):
    action = str(row.get("trade_action", ""))
    path = to_float(row.get("path_score"), 0)
    ret = to_float(row.get("scenario_return_pct"), 0)
    exit_score = to_float(row.get("exit_capability_score"), 0)
    if action == "قابل خرید با کنترل ریسک":
        return "نگهداری؛ افزایش فقط با کنترل ریسک"
    if action == "خرید فقط کم‌حجم و پله‌ای":
        return "نگهداری پرریسک / عدم افزایش سنگین"
    if action == "فعلاً خرید نکن؛ فقط زیر نظر بگیر":
        return "نگهداری مشروط / آماده کاهش موقعیت"
    if exit_score < 25:
        return "کاهش موقعیت در اولین فرصت مناسب"
    if path == 0 or ret <= 0:
        return "خروج یا عدم افزایش"
    return "کاهش ریسک یا بررسی دستی"


def score_explanation(row):
    return (
        f"امتیاز دارایی پایه: {row.get('underlying_scenario_score')} از ۱۰۰؛ یعنی خود دارایی پایه چقدر با سناریوی انتخابی هم‌راستاست. | "
        f"امتیاز مسیر: {row.get('path_score')} از ۱۰۰؛ یعنی رسیدن به Breakeven در روزهای کاری باقی‌مانده چقدر عملی است. | "
        f"امتیاز اردربوک اختیار: {row.get('orderbook_quality_score')} از ۱۰۰؛ یعنی کیفیت bid/ask، عمق ورود، عمق خروج و اسپرد. | "
        f"امتیاز نقدشوندگی: {row.get('liquidity_score')} از ۱۰۰؛ یعنی حجم، ارزش معاملات، تعداد معاملات و موقعیت باز. | "
        f"امتیاز نهایی فرصت: {row.get('opportunity_score')} از ۱۰۰."
    )


def signal_reason(row, scenario_label):
    reasons = [
        f"توصیه معامله: {row.get('trade_action')}",
        f"اقدام برای دارنده اختیار: {row.get('holder_action')}",
        f"سناریوی دارایی پایه: {scenario_label}",
        f"وضعیت دارایی پایه: {row.get('underlying_status')}",
        f"نوع اختیار: {str(row.get('option_type', '')).upper()}",
        f"وضعیت اختیار نسبت به قیمت پایه: {row.get('moneyness_label')}",
        f"فاصله تا Breakeven: {row.get('required_move_to_breakeven_pct')}٪",
        f"روز لازم تا Breakeven: {row.get('days_needed_to_breakeven')}",
        f"روز کاری تا سررسید: {row.get('working_days_to_maturity')}",
        f"وضعیت مسیر: {row.get('path_status')}",
        f"وضعیت اردربوک اختیار: {row.get('orderbook_label')}",
        f"امتیاز نهایی فرصت: {row.get('opportunity_score')} از ۱۰۰",
        f"سیگنال مدل: {row.get('final_signal')}",
    ]
    return " | ".join([str(x) for x in reasons])


def calc_warning_flags(row, config):
    warnings = []
    if to_float(row.get("alignment_score"), 0) < 90:
        warnings.append("سناریو با نوع اختیار هم‌راستا نیست")
    if to_float(row.get("scenario_return_pct"), 0) <= 0:
        warnings.append("بازده سناریویی مثبت نیست")
    if to_float(row.get("path_score"), 0) == 0:
        warnings.append("روز کاری کافی برای Breakeven نیست")
    move_be = to_float(row.get("required_move_to_breakeven_pct"))
    if pd.notna(move_be) and move_be > config["max_breakeven_move_pct"]:
        warnings.append("فاصله Breakeven زیاد است")
    if to_float(row.get("orderbook_quality_score"), 0) < 35:
        warnings.append("اردربوک ضعیف است")
    if to_float(row.get("exit_capability_score"), 0) < 35:
        warnings.append("قابلیت خروج ضعیف است")
    if to_float(row.get("liquidity_score"), 0) < 35:
        warnings.append("نقدشوندگی ضعیف است")
    if to_float(row.get("premium_risk_score"), 0) < 40:
        warnings.append("پرمیوم سنگین یا نامطمئن است")
    return " | ".join(warnings) if warnings else "هشدار جدی ندارد"


def calc_ranking_reason(row):
    parts = []
    if to_float(row.get("alignment_score"), 0) >= 90:
        parts.append("هم‌جهت با سناریو")
    if to_float(row.get("scenario_return_pct"), 0) > 0:
        parts.append(f"بازده سناریویی {row.get('scenario_return_pct')}٪")
    if to_float(row.get("path_score"), 0) >= 70:
        parts.append("مسیر Breakeven قابل قبول")
    if to_float(row.get("orderbook_quality_score"), 0) >= 55:
        parts.append("اردربوک قابل قبول")
    if to_float(row.get("exit_capability_score"), 0) >= 55:
        parts.append("قابلیت خروج مناسب")
    if to_float(row.get("liquidity_score"), 0) >= 55:
        parts.append("نقدشوندگی قابل قبول")
    if not parts:
        parts.append("نیازمند بررسی دستی به دلیل ضعف در مسیر، سناریو یا اردربوک")
    return "؛ ".join(parts)


# =========================================================
# Enrichment
# =========================================================

def enrich_options(df, daily_limit, underlying_scenario, scenario_label, expected_move_pct, config):
    df = df.copy()

    df["calendar_days_to_maturity"] = pd.to_numeric(df["days_to_maturity"], errors="coerce")
    df["working_days_to_maturity"] = df["calendar_days_to_maturity"].apply(
        lambda x: estimate_working_days(x, working_ratio=config["working_day_ratio"], holiday_buffer=config["holiday_buffer"])
    )

    df["underlying_day_change_pct"] = df.apply(calc_underlying_day_change_pct, axis=1)
    df["underlying_intraday_pressure_pct"] = df.apply(calc_underlying_intraday_pressure_pct, axis=1)
    df["expected_underlying_price"] = df.apply(lambda r: calc_expected_underlying_price(r, expected_move_pct), axis=1)
    df["underlying_scenario_score"] = df.apply(lambda r: calc_underlying_scenario_score(r, underlying_scenario), axis=1)
    df["underlying_status"] = df.apply(lambda r: calc_underlying_status(r, underlying_scenario), axis=1)

    df["market_price"] = df.apply(get_market_price, axis=1)
    df["entry_price"] = df.apply(get_entry_price, axis=1)
    df["intrinsic_value"] = df.apply(calc_intrinsic, axis=1)
    df["time_value"] = df["entry_price"] - df["intrinsic_value"]
    df["breakeven"] = df.apply(calc_breakeven, axis=1)
    df["required_move_to_strike_pct"] = df.apply(required_move_to_strike_pct, axis=1)
    df["required_move_to_breakeven_pct"] = df.apply(required_move_to_breakeven_pct, axis=1)
    df["days_needed_to_breakeven"] = df.apply(lambda r: days_needed_to_breakeven(r, daily_limit), axis=1)

    df["bid_ask_spread_pct"] = df.apply(calc_bid_ask_spread_pct, axis=1)
    df["spread_score"] = df.apply(calc_spread_score, axis=1)
    df["entry_depth_score"] = df.apply(lambda r: calc_entry_depth_score(r, config), axis=1)
    df["exit_depth_score"] = df.apply(lambda r: calc_exit_depth_score(r, config), axis=1)
    df["depth_balance_score"] = df.apply(calc_depth_balance_score, axis=1)
    df["orderbook_quality_score"] = df.apply(calc_orderbook_quality_score, axis=1)
    df["orderbook_label"] = df.apply(calc_orderbook_label, axis=1)

    df["moneyness_pct"] = df.apply(calc_moneyness_pct, axis=1)
    df["moneyness_label"] = df.apply(
        lambda r: calc_moneyness_label(r, atm_threshold=config["atm_threshold_pct"], deep_threshold=config["deep_otm_threshold_pct"]), axis=1
    )
    df["moneyness_score"] = df.apply(calc_moneyness_score, axis=1)
    df["premium_ratio_pct"] = df.apply(calc_premium_ratio_pct, axis=1)
    df["time_value_ratio_pct"] = df.apply(calc_time_value_ratio_pct, axis=1)
    df["premium_risk_score"] = df.apply(calc_premium_risk_score, axis=1)
    df["volume_to_oi_ratio_pct"] = df.apply(calc_volume_to_oi_ratio_pct, axis=1)

    path_results = df.apply(calc_path_score, axis=1)
    df["path_score"] = [x[0] for x in path_results]
    df["path_status"] = [x[1] for x in path_results]

    df["liquidity_score"] = df.apply(lambda r: calc_liquidity_score(r, config), axis=1)
    df["price_reliability_score"] = df.apply(lambda r: calc_price_reliability_score(r, config), axis=1)
    df["expiry_quality_score"] = df.apply(lambda r: calc_expiry_quality_score(r, config), axis=1)
    df["alignment_score"] = df.apply(lambda r: calc_alignment_score(r, underlying_scenario), axis=1)
    df["exit_capability_score"] = df.apply(calc_exit_capability_score, axis=1)
    df["exit_capability_label"] = df.apply(calc_exit_capability_label, axis=1)

    df["scenario_payoff"] = df.apply(calc_scenario_payoff, axis=1)
    df["scenario_pnl"] = df.apply(calc_scenario_pnl, axis=1)
    df["scenario_return_pct"] = df.apply(calc_scenario_return_pct, axis=1)
    df["scenario_value_score"] = df.apply(calc_scenario_value_score, axis=1)

    df["simple_leverage"] = df.apply(calc_simple_leverage, axis=1)
    df["risk_bucket"] = df.apply(calc_risk_bucket, axis=1)
    df["execution_score"] = df.apply(calc_execution_score, axis=1)
    df["execution_risk"] = df.apply(execution_risk_label, axis=1)
    df["opportunity_score"] = df.apply(calc_opportunity_score, axis=1)
    df["final_signal"] = df.apply(lambda r: final_signal(r, underlying_scenario, config), axis=1)
    df["trade_action"] = df.apply(calc_trade_action, axis=1)
    df["holder_action"] = df.apply(calc_holder_action, axis=1)
    df["warning_flags"] = df.apply(lambda r: calc_warning_flags(r, config), axis=1)
    df["ranking_reason"] = df.apply(calc_ranking_reason, axis=1)
    df["score_explanation"] = df.apply(score_explanation, axis=1)
    df["signal_reason"] = df.apply(lambda r: signal_reason(r, scenario_label), axis=1)

    df["app_brand"] = APP_BRAND
    df["app_version"] = APP_VERSION
    df["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df


# =========================================================
# Recommendations / Display
# =========================================================

def tradable_subset(df):
    return df[df["trade_action"].isin([
        "قابل خرید با کنترل ریسک",
        "خرید فقط کم‌حجم و پله‌ای",
        "فعلاً خرید نکن؛ فقط زیر نظر بگیر",
        "کاندید بررسی دستی",
    ])].copy()


def aligned_subset(df):
    """Rows aligned with the selected scenario. This prevents bearish scenarios from hiding Put candidates."""
    x = df.copy()
    if "alignment_score" in x.columns:
        x = x[x["alignment_score"] >= 90]
    return x


def top_candidates(df, n=5):
    x = tradable_subset(df)

    aligned = aligned_subset(x)
    if not aligned.empty:
        x = aligned

    # Fallback: if no tradable aligned row exists, still show best aligned rows.
    # This is essential when expected move is negative: Put candidates must surface.
    if x.empty:
        x = aligned_subset(df)

    if x.empty:
        return x

    return x.sort_values(
        ["alignment_score", "opportunity_score", "scenario_return_pct", "orderbook_quality_score", "exit_capability_score", "liquidity_score"],
        ascending=[False, False, False, False, False, False],
    ).head(n)


def categorized_recommendations(df):
    """Five practical recommendation cards by trader risk profile."""
    valid = tradable_subset(df)
    aligned_valid = aligned_subset(valid)
    if not aligned_valid.empty:
        valid = aligned_valid
    if valid.empty:
        valid = aligned_subset(df)
    rows = []
    if valid.empty:
        return pd.DataFrame()

    def pick(label, subset, sort_columns, ascending=None, used=None):
        if subset.empty:
            return None
        subset = subset.copy()
        sort_columns = [c for c in sort_columns if c in subset.columns]
        if ascending is None:
            ascending = [False] * len(sort_columns)
        if sort_columns:
            subset = subset.sort_values(sort_columns, ascending=ascending[:len(sort_columns)])
        for _, candidate in subset.iterrows():
            ticker = str(candidate.get("ticker"))
            if used is None or ticker not in used:
                row = candidate.copy()
                row["recommendation_type"] = label
                if used is not None:
                    used.add(ticker)
                return row
        return None

    used = set()
    categories = [
        (
            "گزینه محافظه‌کارانه",
            valid[valid["risk_bucket"].isin(["ریسک کمتر", "ریسک متعادل"])],
            ["opportunity_score", "orderbook_quality_score", "exit_capability_score", "liquidity_score"],
            [False, False, False, False],
        ),
        (
            "گزینه متعادل",
            valid[valid["trade_action"].isin(["قابل خرید با کنترل ریسک", "خرید فقط کم‌حجم و پله‌ای", "کاندید بررسی دستی"])],
            ["opportunity_score", "scenario_return_pct", "orderbook_quality_score"],
            [False, False, False],
        ),
        (
            "گزینه اهرمی",
            valid[valid["risk_bucket"].eq("ریسک بالا / اهرمی")],
            ["scenario_return_pct", "opportunity_score", "exit_capability_score"],
            [False, False, False],
        ),
        (
            "نقدشونده‌ترین گزینه",
            valid,
            ["liquidity_score", "exit_capability_score", "orderbook_quality_score", "opportunity_score"],
            [False, False, False, False],
        ),
        (
            "پربازده‌ترین گزینه سناریویی",
            valid[valid["scenario_return_pct"] > 0],
            ["scenario_return_pct", "opportunity_score", "orderbook_quality_score"],
            [False, False, False],
        ),
    ]

    for label, subset, sort_columns, ascending in categories:
        row = pick(label, subset, sort_columns, ascending, used)
        if row is not None:
            rows.append(row)

    # If a category could not be filled, complete the cards with the best remaining aligned opportunities.
    while len(rows) < min(5, len(valid)):
        row = pick("کاندید تکمیلی", valid, ["opportunity_score", "scenario_return_pct", "orderbook_quality_score"], [False, False, False], used)
        if row is None:
            break
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()

def top_by_underlying(df, n=5):
    valid = tradable_subset(df)
    aligned = aligned_subset(valid)
    if not aligned.empty:
        valid = aligned
    if valid.empty:
        valid = aligned_subset(df)
    if valid.empty:
        return valid
    valid = valid.sort_values(["ua_ticker", "alignment_score", "opportunity_score", "scenario_return_pct"], ascending=[True, False, False, False])
    return valid.groupby("ua_ticker", group_keys=False).head(n)


def add_row_number(df):
    x = df.copy().reset_index(drop=True)
    x.insert(0, "ردیف", range(1, len(x) + 1))
    return x


def style_decision_table(df):
    def row_style(row):
        label = str(row.get("وضعیت اختیار نسبت به قیمت پایه", ""))
        action = str(row.get("توصیه معامله", ""))

        if "دارای ارزش ذاتی" in label:
            color = "background-color: #e8f5e9"
        elif "لب مرز" in label:
            color = "background-color: #fffde7"
        elif "خیلی دور" in label:
            color = "background-color: #fff3e0"
        elif "بدون ارزش" in label:
            color = "background-color: #e3f2fd"
        else:
            color = ""

        if action == "عدم معامله":
            color = "background-color: #ffebee"
        return [color for _ in row]
    return df.style.apply(row_style, axis=1)


def prepare_view(df, display_cols, persian_names, sort_cols=None):
    available_cols = [c for c in display_cols if c in df.columns]
    x = df.copy()
    if sort_cols:
        sort_cols = [c for c in sort_cols if c in x.columns]
        if sort_cols:
            x = x.sort_values(sort_cols)
    view = x[available_cols].copy().rename(columns=persian_names)
    return add_row_number(view)


def card_class(action):
    if action == "قابل خرید با کنترل ریسک":
        return "buy"
    if action in ["خرید فقط کم‌حجم و پله‌ای", "فعلاً خرید نکن؛ فقط زیر نظر بگیر", "کاندید بررسی دستی"]:
        return "watch"
    if "عدم" in action or "خروج" in action:
        return "avoid"
    return "neutral"


def make_recommendation_cards(df):
    if df.empty:
        st.info("در شرایط فعلی پیشنهادی که شروط اولیه را پاس کند پیدا نشد.")
        return

    cols = st.columns(min(3, len(df)))
    for i, (_, row) in enumerate(df.iterrows()):
        action = row.get("trade_action", "-")
        cls = card_class(action)
        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div class="rec-card {cls}">
                    <b>{row.get('recommendation_type', 'پیشنهاد')}</b><br>
                    نماد: <b>{row.get('ticker', '-')}</b><br>
                    دارایی پایه: {row.get('ua_ticker', '-')}<br>
                    توصیه معامله: <b>{action}</b><br>
                    اقدام برای دارنده: {row.get('holder_action', '-')}<br>
                    امتیاز فرصت: <b>{row.get('opportunity_score', '-')}</b> از ۱۰۰<br>
                    بازده سناریویی: {row.get('scenario_return_pct', '-')}٪<br>
                    اردربوک: {row.get('orderbook_label', '-')}<br>
                    قابلیت خروج: {row.get('exit_capability_label', '-')}<br>
                    وضعیت اختیار: {row.get('moneyness_label', '-')}<br>
                    دلیل رتبه‌بندی: {row.get('ranking_reason', '-')}
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# Charts
# =========================================================

def no_plotly_warning():
    st.warning("برای نمایش نمودارها، plotly باید نصب باشد: pip install plotly")


_PLOTLY_COUNTER = 0


def render_plotly_chart(fig, base_key):
    """Render Plotly charts with a unique Streamlit key."""
    global _PLOTLY_COUNTER
    _PLOTLY_COUNTER += 1
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{base_key}_{_PLOTLY_COUNTER}",
    )


def chart_action_composition(df):
    if px is None:
        no_plotly_warning(); return
    counts = df["trade_action"].value_counts().reset_index()
    counts.columns = ["توصیه معامله", "تعداد"]
    fig = px.bar(
        counts, x="تعداد", y="توصیه معامله", orientation="h", text="تعداد",
        title="ترکیب توصیه‌های معامله"
    )
    fig.update_layout(height=420)
    render_plotly_chart(fig, "chart_action_composition")


def chart_top_recommendations(df):
    if px is None:
        no_plotly_warning(); return
    top = top_candidates(df, n=10)
    if top.empty:
        st.info("داده کافی برای نمودار پیشنهادهای برتر وجود ندارد."); return
    top = top.sort_values("opportunity_score", ascending=True)
    fig = px.bar(
        top,
        x="opportunity_score", y="ticker", orientation="h", color="trade_action", text="opportunity_score",
        hover_data=["ua_ticker", "scenario_return_pct", "orderbook_quality_score", "required_move_to_breakeven_pct", "option_type"],
        labels={"opportunity_score": "امتیاز فرصت", "ticker": "نماد", "trade_action": "توصیه معامله"},
        title="۱۰ گزینه اول بر اساس امتیاز فرصت"
    )
    fig.update_layout(height=540)
    render_plotly_chart(fig, "chart_top_recommendations")


def chart_return_bar(df):
    if px is None:
        no_plotly_warning(); return
    top = top_candidates(df, n=12)
    if top.empty:
        st.info("داده کافی برای نمودار بازده سناریویی وجود ندارد."); return
    top = top.sort_values("scenario_return_pct", ascending=True)
    fig = px.bar(
        top,
        x="scenario_return_pct", y="ticker", orientation="h", color="trade_action", text="scenario_return_pct",
        hover_data=["ua_ticker", "opportunity_score", "required_move_to_breakeven_pct", "orderbook_quality_score", "option_type"],
        labels={"scenario_return_pct": "بازده سناریویی (%)", "ticker": "نماد", "trade_action": "توصیه معامله"},
        title="بازده سناریویی گزینه‌های برتر"
    )
    fig.update_layout(height=540)
    render_plotly_chart(fig, "chart_return_bar")


def chart_breakeven_bar(df):
    if px is None:
        no_plotly_warning(); return
    top = top_candidates(df, n=12)
    if top.empty:
        st.info("داده کافی برای نمودار فاصله تا Breakeven وجود ندارد."); return
    top = top.sort_values("required_move_to_breakeven_pct", ascending=True)
    fig = px.bar(
        top,
        x="required_move_to_breakeven_pct", y="ticker", orientation="h", color="trade_action", text="required_move_to_breakeven_pct",
        hover_data=["ua_ticker", "opportunity_score", "scenario_return_pct", "orderbook_quality_score", "option_type"],
        labels={"required_move_to_breakeven_pct": "فاصله تا Breakeven (%)", "ticker": "نماد", "trade_action": "توصیه معامله"},
        title="فاصله تا Breakeven در گزینه‌های برتر؛ کمتر معمولاً بهتر است"
    )
    fig.update_layout(height=540)
    render_plotly_chart(fig, "chart_breakeven_bar")


def chart_path_feasibility(df):
    if px is None:
        no_plotly_warning(); return
    top = top_candidates(df, n=12)
    if top.empty:
        st.info("داده کافی برای نمودار مسیر وجود ندارد."); return
    cols = ["ticker", "working_days_to_maturity", "days_needed_to_breakeven"]
    x = top[cols].copy()
    x = x.rename(columns={
        "working_days_to_maturity": "روز کاری تا سررسید",
        "days_needed_to_breakeven": "روز لازم تا Breakeven",
    })
    melted = x.melt(id_vars="ticker", var_name="نوع روز", value_name="تعداد روز")
    fig = px.bar(
        melted, x="ticker", y="تعداد روز", color="نوع روز", barmode="group",
        title="مقایسه روز لازم تا Breakeven با روز کاری باقی‌مانده"
    )
    fig.update_layout(height=520, xaxis_title="نماد")
    render_plotly_chart(fig, "chart_path_feasibility")


def chart_orderbook_bar(df):
    if px is None:
        no_plotly_warning(); return
    top = top_candidates(df, n=12)
    if top.empty:
        st.info("داده کافی برای نمودار کیفیت اردربوک وجود ندارد."); return
    cols = ["ticker", "orderbook_quality_score", "entry_depth_score", "exit_depth_score", "exit_capability_score"]
    x = top[cols].copy().rename(columns={
        "orderbook_quality_score": "امتیاز اردربوک",
        "entry_depth_score": "عمق ورود",
        "exit_depth_score": "عمق خروج",
        "exit_capability_score": "قابلیت خروج",
    })
    melted = x.melt(id_vars="ticker", var_name="عامل", value_name="امتیاز")
    fig = px.bar(
        melted,
        x="ticker",
        y="امتیاز",
        color="عامل",
        barmode="group",
        title="کیفیت اردربوک، عمق ورود/خروج و قابلیت خروج گزینه‌های برتر",
    )
    fig.update_layout(height=520, xaxis_title="نماد")
    render_plotly_chart(fig, "chart_orderbook_bar")


def chart_orderbook_tree(df):
    if px is None:
        no_plotly_warning(); return
    top = top_candidates(df, n=20)
    if top.empty:
        st.info("داده کافی برای نمودار درختی اردربوک وجود ندارد."); return
    top = top.copy()
    top["اندازه"] = top["orderbook_quality_score"].clip(lower=1)
    fig = px.treemap(
        top,
        path=["trade_action", "orderbook_label", "ticker"],
        values="اندازه",
        color="opportunity_score",
        hover_data=["ua_ticker", "option_type", "liquidity_score", "scenario_return_pct", "required_move_to_breakeven_pct"],
        title="نمودار درختی کیفیت اردربوک و امتیاز فرصت"
    )
    fig.update_layout(height=560)
    render_plotly_chart(fig, "chart_orderbook_tree")


def chart_score_breakdown_tree(df):
    if px is None:
        no_plotly_warning(); return
    top = top_candidates(df, n=25)
    if top.empty:
        st.info("داده کافی برای نمودار درختی پیشنهادها وجود ندارد."); return
    top = top.copy()
    top["اندازه"] = top["opportunity_score"].clip(lower=1)
    fig = px.treemap(
        top,
        path=["ua_ticker", "trade_action", "ticker"],
        values="اندازه",
        color="scenario_return_pct",
        hover_data=["option_type", "orderbook_quality_score", "liquidity_score", "required_move_to_breakeven_pct"],
        title="نمودار درختی پیشنهادها؛ اندازه بر اساس امتیاز فرصت، رنگ بر اساس بازده سناریویی"
    )
    fig.update_layout(height=560)
    render_plotly_chart(fig, "chart_score_breakdown_tree")


def chart_heatmap(df):
    if go is None:
        no_plotly_warning(); return
    x = df.copy()
    x = x[pd.notna(x["strike_price"]) & pd.notna(x["opportunity_score"])]
    if x.empty:
        st.info("داده کافی برای نقشه حرارتی وجود ندارد."); return
    if x["ua_ticker"].nunique() > 1:
        best_underlying = x.groupby("ua_ticker")["opportunity_score"].mean().sort_values(ascending=False).index[0]
        x = x[x["ua_ticker"].eq(best_underlying)]
        st.caption(f"برای خوانایی، نقشه حرارتی فقط برای دارایی پایه {best_underlying} نمایش داده شده است.")
    pivot = x.pivot_table(index="strike_price", columns="end_date", values="opportunity_score", aggfunc="mean")
    if pivot.empty:
        st.info("داده کافی برای نقشه حرارتی وجود ندارد."); return
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index], colorbar=dict(title="امتیاز فرصت")))
    fig.update_layout(title="نقشه حرارتی امتیاز فرصت بر اساس سررسید و قیمت اعمال", xaxis_title="سررسید", yaxis_title="قیمت اعمال", height=560)
    render_plotly_chart(fig, "chart_heatmap")



# =========================================================
# Strategy Dashboard Model
# =========================================================

STRATEGY_MODE_OPTIONS = [
    "هوشمند بر اساس سناریو",
    "فقط استراتژی‌های خریدی با ریسک محدود",
    "اسپردهای عمودی",
    "نوسان شدید / بی‌جهت",
    "هج و سهام‌داری",
    "درآمدی / تعهد فروش فقط برای بررسی",
]

DEBIT_STRATEGIES = {
    "خرید Call تکی", "خرید Put تکی", "Bull Call Spread", "Bear Put Spread",
    "Long Straddle", "Long Strangle", "Protective Put", "Collar"
}

OBLIGATION_STRATEGIES = {
    "Covered Call", "Bull Put Spread", "Bear Call Spread", "Iron Condor"
}


def safe_pct(numerator, denominator, decimals=2):
    try:
        numerator = to_float(numerator)
        denominator = to_float(denominator)
        if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
            return np.nan
        return round(numerator / denominator * 100, decimals)
    except Exception:
        return np.nan


def option_leg_price(row, side):
    """Price used for a leg: buy at ask, sell/receive at bid."""
    if side == "buy":
        return to_float(row.get("ask_price"))
    return to_float(row.get("bid_price"))


def has_buy_side(row):
    price = to_float(row.get("ask_price"))
    volume = to_float(row.get("ask_volume"), 0)
    return pd.notna(price) and price > 0 and volume > 0


def has_sell_side(row):
    price = to_float(row.get("bid_price"))
    volume = to_float(row.get("bid_volume"), 0)
    return pd.notna(price) and price > 0 and volume > 0


def option_payoff(row, target_price):
    k = to_float(row.get("strike_price"))
    target_price = to_float(target_price)
    if pd.isna(k) or pd.isna(target_price):
        return np.nan
    if is_put_option(row):
        return max(k - target_price, 0)
    return max(target_price - k, 0)


def strategy_return_score(ret_pct):
    r = to_float(ret_pct)
    if pd.isna(r):
        return 0
    if r <= -20:
        return 0
    if r <= 0:
        return 20
    if r < 15:
        return 40
    if r < 40:
        return 65
    if r < 80:
        return 85
    return 100


def strategy_alignment_score(strategy_view, underlying_scenario, expected_move_pct, expected_volatility_pct):
    move = abs(to_float(expected_move_pct, 0))
    vol = abs(to_float(expected_volatility_pct, move))

    if strategy_view == "bullish":
        if underlying_scenario == "bullish":
            return 100
        if underlying_scenario == "neutral":
            return 45
        if underlying_scenario == "unknown":
            return 55
        return 10

    if strategy_view == "bearish":
        if underlying_scenario == "bearish":
            return 100
        if underlying_scenario == "neutral":
            return 45
        if underlying_scenario == "unknown":
            return 55
        return 10

    if strategy_view == "volatility":
        if vol >= 8:
            return 100
        if vol >= 5:
            return 75
        if underlying_scenario in ["unknown", "neutral"]:
            return 65
        return 45

    if strategy_view == "neutral_income":
        if underlying_scenario == "neutral" and move <= 4:
            return 100
        if move <= 3:
            return 85
        if move <= 6:
            return 55
        return 25

    if strategy_view == "hedge":
        if underlying_scenario == "bearish":
            return 95
        if underlying_scenario in ["neutral", "unknown"]:
            return 75
        return 55

    return 50


def strategy_time_score(days):
    d = to_float(days)
    if pd.isna(d) or d <= 0:
        return 0
    if 5 <= d <= 45:
        return 100
    if 46 <= d <= 75:
        return 75
    if 76 <= d <= 120:
        return 55
    if d < 5:
        return 25
    return 40


def aggregate_leg_score(legs, column_name, default=0):
    values = []
    for leg in legs:
        row = leg.get("row")
        if row is not None:
            values.append(to_float(row.get(column_name), default))
    if not values:
        return default
    return round(float(np.nanmean(values)), 1)


def aggregate_execution_score(legs):
    if not legs:
        return 0
    ob = aggregate_leg_score(legs, "orderbook_quality_score")
    liq = aggregate_leg_score(legs, "liquidity_score")
    exe = aggregate_leg_score(legs, "execution_score")
    exit_score = aggregate_leg_score(legs, "exit_capability_score")
    return round(clamp(0.32 * ob + 0.28 * liq + 0.25 * exe + 0.15 * exit_score), 1)


def leg_text(row, side):
    side_label = "خرید" if side == "buy" else "فروش"
    opt = "Put" if is_put_option(row) else "Call"
    ticker = row.get("ticker", "-")
    strike = format_number(row.get("strike_price"), 0)
    price = option_leg_price(row, side)
    return f"{side_label} {opt} {ticker} با اعمال {strike} @ {format_number(price, 0)}"


def build_leg_payload(legs):
    """Machine-readable legs for Strategy Lab payoff and scenario tables."""
    payload = []
    for leg in legs:
        row = leg.get("row")
        side = leg.get("side", "buy")
        if row is None:
            continue
        payload.append({
            "side": side,
            "side_fa": "خرید" if side == "buy" else "فروش",
            "ticker": str(row.get("ticker", "-")),
            "name": str(row.get("name", "-")),
            "option_type": str(row.get("option_type", "")).lower(),
            "option_type_fa": "Put" if is_put_option(row) else "Call",
            "strike_price": to_float(row.get("strike_price")),
            "entry_price": option_leg_price(row, side),
            "bid_price": to_float(row.get("bid_price")),
            "ask_price": to_float(row.get("ask_price")),
            "bid_volume": to_float(row.get("bid_volume"), 0),
            "ask_volume": to_float(row.get("ask_volume"), 0),
            "ua_ticker": str(row.get("ua_ticker", "-")),
            "ua_close_price": to_float(row.get("ua_close_price")),
            "end_date": str(row.get("end_date", "-")),
            "orderbook_quality_score": to_float(row.get("orderbook_quality_score"), 0),
            "liquidity_score": to_float(row.get("liquidity_score"), 0),
            "exit_capability_score": to_float(row.get("exit_capability_score"), 0),
            "working_days_to_maturity": to_float(row.get("working_days_to_maturity"), 0),
        })
    return payload


def legs_payload_to_json(legs):
    try:
        return json.dumps(build_leg_payload(legs), ensure_ascii=False)
    except Exception:
        return "[]"


def parse_strategy_legs(strategy_row):
    payload = strategy_row.get("legs_payload", "[]") if hasattr(strategy_row, "get") else "[]"
    if isinstance(payload, list):
        return payload
    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def leg_payload_payoff(leg, underlying_price):
    k = to_float(leg.get("strike_price"))
    s = to_float(underlying_price)
    if pd.isna(k) or pd.isna(s):
        return np.nan
    opt_type = str(leg.get("option_type", "")).lower()
    if opt_type == "put":
        return max(k - s, 0)
    return max(s - k, 0)


def strategy_expiry_pnl(strategy_row, underlying_price):
    """Expiry P/L from stored legs. Uses buy-at-ask / sell-at-bid cashflows and optional stock exposure."""
    pnl = 0.0
    for leg in parse_strategy_legs(strategy_row):
        payoff = leg_payload_payoff(leg, underlying_price)
        entry = to_float(leg.get("entry_price"))
        if pd.isna(payoff) or pd.isna(entry):
            return np.nan
        if str(leg.get("side", "buy")) == "buy":
            pnl += payoff - entry
        else:
            pnl += entry - payoff

    stock_units = to_float(strategy_row.get("underlying_units", 0), 0)
    stock_entry = to_float(strategy_row.get("underlying_entry_price"))
    if stock_units and pd.notna(stock_entry):
        pnl += stock_units * (to_float(underlying_price) - stock_entry)
    return round(pnl, 2)


def strategy_capital_base(strategy_row):
    base = to_float(strategy_row.get("capital_base"))
    if pd.notna(base) and base > 0:
        return base
    for key in ["entry_cost", "max_loss"]:
        value = abs(to_float(strategy_row.get(key), 0))
        if value > 0:
            return value
    stock_entry = to_float(strategy_row.get("underlying_entry_price"))
    stock_units = abs(to_float(strategy_row.get("underlying_units", 0), 0))
    if pd.notna(stock_entry) and stock_entry > 0 and stock_units > 0:
        return stock_entry * stock_units
    return np.nan


def strategy_underlying_reference(strategy_row):
    s0 = to_float(strategy_row.get("underlying_entry_price"))
    if pd.notna(s0) and s0 > 0:
        return s0
    legs = parse_strategy_legs(strategy_row)
    for leg in legs:
        s0 = to_float(leg.get("ua_close_price"))
        if pd.notna(s0) and s0 > 0:
            return s0
    target = to_float(strategy_row.get("target_price"))
    if pd.notna(target) and target > 0:
        return target
    return np.nan


def build_strategy_payoff_curve(strategy_row, range_pct=25, steps=61):
    s0 = strategy_underlying_reference(strategy_row)
    if pd.isna(s0) or s0 <= 0:
        return pd.DataFrame()
    range_pct = max(5.0, min(80.0, abs(to_float(range_pct, 25)))) / 100
    prices = np.linspace(max(1, s0 * (1 - range_pct)), s0 * (1 + range_pct), int(steps))
    capital = strategy_capital_base(strategy_row)
    rows = []
    for price in prices:
        pnl = strategy_expiry_pnl(strategy_row, price)
        rows.append({
            "قیمت پایه": round(float(price), 2),
            "تغییر پایه (%)": round((price / s0 - 1) * 100, 2),
            "سود/زیان در سررسید": pnl,
            "بازده نسبت به سرمایه (%)": safe_pct(pnl, capital),
        })
    return pd.DataFrame(rows)


def build_strategy_scenario_table(strategy_row, expected_move_pct=0, expected_volatility_pct=0):
    s0 = strategy_underlying_reference(strategy_row)
    if pd.isna(s0) or s0 <= 0:
        return pd.DataFrame()
    moves = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    for m in [to_float(expected_move_pct, 0), abs(to_float(expected_volatility_pct, 0)), -abs(to_float(expected_volatility_pct, 0))]:
        if pd.notna(m):
            rounded = int(round(m))
            if rounded not in moves and -50 <= rounded <= 50:
                moves.append(rounded)
    moves = sorted(set(moves))
    capital = strategy_capital_base(strategy_row)
    rows = []
    for move in moves:
        price = s0 * (1 + move / 100)
        pnl = strategy_expiry_pnl(strategy_row, price)
        rows.append({
            "سناریو تغییر پایه": f"{move:+.0f}%",
            "قیمت پایه در سناریو": round(price, 2),
            "سود/زیان استراتژی": pnl,
            "بازده نسبت به سرمایه (%)": safe_pct(pnl, capital),
            "نتیجه": "سود" if to_float(pnl, 0) > 0 else "زیان" if to_float(pnl, 0) < 0 else "سر به سر",
        })
    return pd.DataFrame(rows)


def strategy_leg_details_table(strategy_row):
    legs = parse_strategy_legs(strategy_row)
    if not legs:
        return pd.DataFrame()
    rows = []
    for leg in legs:
        rows.append({
            "عملیات": leg.get("side_fa", "-"),
            "نوع": leg.get("option_type_fa", "-"),
            "نماد": leg.get("ticker", "-"),
            "قیمت اعمال": leg.get("strike_price", np.nan),
            "قیمت ورود پایه": leg.get("entry_price", np.nan),
            "Bid": leg.get("bid_price", np.nan),
            "حجم Bid": leg.get("bid_volume", np.nan),
            "Ask": leg.get("ask_price", np.nan),
            "حجم Ask": leg.get("ask_volume", np.nan),
            "امتیاز اردربوک": leg.get("orderbook_quality_score", np.nan),
            "امتیاز نقدشوندگی": leg.get("liquidity_score", np.nan),
            "امتیاز خروج": leg.get("exit_capability_score", np.nan),
        })
    return add_row_number(pd.DataFrame(rows))


def chart_strategy_payoff(strategy_row, range_pct=25):
    if px is None:
        no_plotly_warning(); return
    curve = build_strategy_payoff_curve(strategy_row, range_pct=range_pct)
    if curve.empty:
        st.info("داده کافی برای نمودار سود/زیان استراتژی وجود ندارد."); return
    fig = px.line(
        curve, x="قیمت پایه", y="سود/زیان در سررسید",
        hover_data=["تغییر پایه (%)", "بازده نسبت به سرمایه (%)"],
        title=f"نمودار سود/زیان در سررسید — {strategy_row.get('strategy_name', '-')}"
    )
    try:
        fig.add_hline(y=0, line_dash="dash", annotation_text="سر به سر")
        s0 = strategy_underlying_reference(strategy_row)
        target = to_float(strategy_row.get("target_price"))
        if pd.notna(s0):
            fig.add_vline(x=s0, line_dash="dot", annotation_text="قیمت فعلی پایه")
        if pd.notna(target):
            fig.add_vline(x=target, line_dash="dot", annotation_text="قیمت سناریویی")
    except Exception:
        pass
    fig.update_layout(height=560, xaxis_title="قیمت دارایی پایه", yaxis_title="سود/زیان تقریبی")
    render_plotly_chart(fig, "chart_strategy_payoff_lab")


def strategy_warnings(name, legs, obligations_required, allow_obligation_strategies, max_loss, scenario_return_pct, execution_score, alignment_score):
    warnings = []
    if obligations_required and not allow_obligation_strategies:
        warnings.append("این ساختار شامل فروش اختیار/تعهد است و فقط برای بررسی نمایش داده شده")
    if obligations_required:
        warnings.append("قبل از اجرا، وجه تضمین، امکان فروش تعهدی و مقررات کارگزاری بررسی شود")
    if pd.isna(to_float(max_loss)):
        warnings.append("حداکثر زیان عددی قابل اتکا نیست")
    if to_float(scenario_return_pct, 0) <= 0:
        warnings.append("در سناریوی فعلی بازده مثبت نمی‌دهد")
    if execution_score < 45:
        warnings.append("کیفیت اجرا/اردربوک برای ترکیب چندپایه ضعیف است")
    if alignment_score < 50:
        warnings.append("سناریوی بازار با این استراتژی هم‌راستا نیست")
    for leg in legs:
        row = leg.get("row")
        side = leg.get("side")
        if row is None:
            continue
        if side == "buy" and not has_buy_side(row):
            warnings.append(f"برای ورود به {row.get('ticker', '-')} Ask فعال کافی نیست")
        if side == "sell" and not has_sell_side(row):
            warnings.append(f"برای فروش/دریافت پرمیوم {row.get('ticker', '-')} Bid فعال کافی نیست")
        if to_float(row.get("exit_capability_score"), 0) < 30:
            warnings.append(f"قابلیت خروج {row.get('ticker', '-')} ضعیف است")
    # keep the message readable
    clean = []
    for item in warnings:
        if item not in clean:
            clean.append(item)
    return " | ".join(clean[:5]) if clean else "هشدار جدی ندارد"


def strategy_action_label(score, obligations_required, allow_obligation_strategies, scenario_return_pct, execution_score, alignment_score):
    score = to_float(score, 0)
    ret = to_float(scenario_return_pct, 0)
    if obligations_required and not allow_obligation_strategies:
        return "فقط بررسی؛ نیازمند مجوز/کنترل تعهد فروش"
    if alignment_score < 35:
        return "عدم اجرا؛ با سناریو هم‌راستا نیست"
    if execution_score < 35:
        return "عدم اجرا تا بهتر شدن اردربوک"
    if ret <= 0 and not obligations_required:
        return "عدم اجرا؛ سناریو سود نمی‌دهد"
    if score >= 76:
        return "استراتژی قابل اجرا با کنترل ریسک"
    if score >= 62:
        return "قابل بررسی کم‌حجم و پله‌ای"
    if score >= 50:
        return "فقط زیر نظر / بررسی دستی"
    return "عدم اجرای استراتژی"


def make_strategy_row(strategy_name, family, market_view, legs, underlying_scenario, expected_move_pct,
                      expected_volatility_pct, ua_ticker, end_date, target_price, entry_cost=0,
                      net_credit=0, max_profit=np.nan, max_loss=np.nan, breakeven_1=np.nan,
                      breakeven_2=np.nan, scenario_payoff=np.nan, scenario_pnl=np.nan,
                      capital_base=np.nan, obligations_required=False, risk_level="متوسط",
                      best_use="", management_note="", risk_shape="ریسک محدود",
                      underlying_units=0, underlying_entry_price=np.nan):
    entry_cost = to_float(entry_cost, 0)
    net_credit = to_float(net_credit, 0)
    scenario_pnl = to_float(scenario_pnl)
    if pd.isna(capital_base) or to_float(capital_base, 0) <= 0:
        capital_base = entry_cost if entry_cost > 0 else abs(to_float(max_loss, 0))
    scenario_return_pct = safe_pct(scenario_pnl, capital_base)

    execution_score = aggregate_execution_score(legs)
    liquidity_score = aggregate_leg_score(legs, "liquidity_score")
    orderbook_score = aggregate_leg_score(legs, "orderbook_quality_score")
    exit_score = aggregate_leg_score(legs, "exit_capability_score")
    alignment_score = strategy_alignment_score(market_view, underlying_scenario, expected_move_pct, expected_volatility_pct)
    ret_score = strategy_return_score(scenario_return_pct)
    time_score = strategy_time_score(aggregate_leg_score(legs, "working_days_to_maturity"))

    if risk_shape == "ریسک محدود":
        risk_control_score = 88
    elif risk_shape == "هج / بیمه":
        risk_control_score = 82
    elif risk_shape == "ریسک سهام‌داری":
        risk_control_score = 55
    else:
        risk_control_score = 35

    obligation_penalty = 12 if obligations_required else 0
    score = (
        0.26 * ret_score +
        0.23 * execution_score +
        0.20 * alignment_score +
        0.12 * risk_control_score +
        0.10 * liquidity_score +
        0.06 * time_score +
        0.03 * exit_score -
        obligation_penalty
    )
    score = round(clamp(score), 1)

    allow_obligation_strategies = bool(st.session_state.get("allow_obligation_strategies", False))
    action = strategy_action_label(score, obligations_required, allow_obligation_strategies, scenario_return_pct, execution_score, alignment_score)
    warnings = strategy_warnings(strategy_name, legs, obligations_required, allow_obligation_strategies, max_loss, scenario_return_pct, execution_score, alignment_score)
    legs_label = " | ".join([leg_text(leg["row"], leg["side"]) for leg in legs])
    tickers = " + ".join([str(leg["row"].get("ticker", "-")) for leg in legs])
    if pd.isna(to_float(underlying_entry_price)):
        for leg in legs:
            row = leg.get("row")
            if row is not None:
                candidate_s0 = to_float(row.get("ua_close_price"))
                if pd.notna(candidate_s0) and candidate_s0 > 0:
                    underlying_entry_price = candidate_s0
                    break
    legs_payload = legs_payload_to_json(legs)

    explanation_parts = []
    explanation_parts.append(f"هم‌جهتی استراتژی با سناریو: {alignment_score} از ۱۰۰")
    explanation_parts.append(f"کیفیت اجرای ترکیب: {execution_score} از ۱۰۰")
    explanation_parts.append(f"بازده سناریویی ترکیب: {scenario_return_pct}٪")
    if obligations_required:
        explanation_parts.append("دارای پایه فروش اختیار/تعهد؛ پیشنهاد خرید عادی نیست")
    if best_use:
        explanation_parts.append(best_use)

    return {
        "strategy_name": strategy_name,
        "strategy_family": family,
        "market_view": market_view,
        "ua_ticker": ua_ticker,
        "end_date": end_date,
        "strategy_action": action,
        "strategy_score": score,
        "risk_level": risk_level,
        "risk_shape": risk_shape,
        "obligations_required": "بله" if obligations_required else "خیر",
        "legs_count": len(legs),
        "legs": legs_label,
        "contracts": tickers,
        "legs_payload": legs_payload,
        "underlying_units": round(to_float(underlying_units, 0), 4),
        "underlying_entry_price": round(to_float(underlying_entry_price), 2) if pd.notna(to_float(underlying_entry_price)) else np.nan,
        "capital_base": round(to_float(capital_base), 2) if pd.notna(to_float(capital_base)) else np.nan,
        "entry_cost": round(entry_cost, 2) if pd.notna(entry_cost) else np.nan,
        "net_credit": round(net_credit, 2) if pd.notna(net_credit) else np.nan,
        "max_profit": round(max_profit, 2) if pd.notna(to_float(max_profit)) else np.nan,
        "max_loss": round(max_loss, 2) if pd.notna(to_float(max_loss)) else np.nan,
        "breakeven_1": round(breakeven_1, 2) if pd.notna(to_float(breakeven_1)) else np.nan,
        "breakeven_2": round(breakeven_2, 2) if pd.notna(to_float(breakeven_2)) else np.nan,
        "target_price": round(to_float(target_price), 2) if pd.notna(to_float(target_price)) else np.nan,
        "scenario_payoff": round(scenario_payoff, 2) if pd.notna(to_float(scenario_payoff)) else np.nan,
        "scenario_pnl": round(scenario_pnl, 2) if pd.notna(to_float(scenario_pnl)) else np.nan,
        "scenario_return_pct": scenario_return_pct,
        "strategy_execution_score": execution_score,
        "strategy_liquidity_score": liquidity_score,
        "strategy_orderbook_score": orderbook_score,
        "strategy_exit_score": exit_score,
        "alignment_score_strategy": alignment_score,
        "working_days_to_maturity": round(aggregate_leg_score(legs, "working_days_to_maturity"), 0),
        "best_use": best_use,
        "management_note": management_note,
        "strategy_warnings": warnings,
        "strategy_reason": "؛ ".join(explanation_parts),
        "app_brand": APP_BRAND,
        "app_version": APP_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def select_option_pool(g, option_type, max_rows=12):
    x = g[g["option_type"].astype(str).str.lower().eq(option_type)].copy()
    if x.empty:
        return x
    x["_pool_rank"] = (
        0.30 * x["orderbook_quality_score"].fillna(0) +
        0.25 * x["liquidity_score"].fillna(0) +
        0.20 * x["exit_capability_score"].fillna(0) +
        0.15 * x["price_reliability_score"].fillna(0) +
        0.10 * x["moneyness_score"].fillna(0)
    )
    x = x.sort_values(["_pool_rank", "opportunity_score"], ascending=[False, False]).head(max_rows)
    return x.drop(columns=["_pool_rank"], errors="ignore")


def add_single_option_strategies(rows, g, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price):
    # keep the classic one-leg model inside the strategy dashboard for comparison
    top = top_candidates(g, n=6)
    for _, r in top.iterrows():
        if not has_buy_side(r):
            continue
        entry = option_leg_price(r, "buy")
        payoff = option_payoff(r, target_price)
        pnl = payoff - entry
        name = "خرید Put تکی" if is_put_option(r) else "خرید Call تکی"
        view = "bearish" if is_put_option(r) else "bullish"
        rows.append(make_strategy_row(
            strategy_name=name,
            family="اختیار تکی",
            market_view=view,
            legs=[{"row": r, "side": "buy"}],
            underlying_scenario=underlying_scenario,
            expected_move_pct=expected_move_pct,
            expected_volatility_pct=expected_volatility_pct,
            ua_ticker=r.get("ua_ticker"),
            end_date=r.get("end_date"),
            target_price=target_price,
            entry_cost=entry,
            net_credit=0,
            max_profit=np.nan,
            max_loss=entry,
            breakeven_1=calc_breakeven(r),
            scenario_payoff=payoff,
            scenario_pnl=pnl,
            capital_base=entry,
            obligations_required=False,
            risk_level=str(r.get("risk_bucket", "متوسط")),
            best_use="مقایسه با استراتژی‌های چندپایه؛ ریسک محدود ولی حساس به قیمت ورود و زمان",
            management_note="حد ضرر و اندازه موقعیت را مستقل از امتیاز مدل کنترل کن.",
            risk_shape="ریسک محدود",
        ))


def add_bull_call_spreads(rows, calls, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price):
    if calls.empty:
        return
    calls = calls.sort_values("strike_price")
    for _, low in calls.iterrows():
        for _, high in calls[calls["strike_price"] > low["strike_price"]].head(5).iterrows():
            if not has_buy_side(low) or not has_sell_side(high):
                continue
            k1 = to_float(low.get("strike_price")); k2 = to_float(high.get("strike_price"))
            debit = option_leg_price(low, "buy") - option_leg_price(high, "sell")
            width = k2 - k1
            if pd.isna(debit) or debit <= 0 or width <= 0 or debit >= width:
                continue
            gross = min(max(to_float(target_price) - k1, 0), width)
            pnl = gross - debit
            rows.append(make_strategy_row(
                "Bull Call Spread", "اسپرد عمودی", "bullish",
                [{"row": low, "side": "buy"}, {"row": high, "side": "sell"}], underlying_scenario,
                expected_move_pct, expected_volatility_pct, low.get("ua_ticker"), low.get("end_date"), target_price,
                entry_cost=debit, net_credit=0, max_profit=width - debit, max_loss=debit,
                breakeven_1=k1 + debit, scenario_payoff=gross, scenario_pnl=pnl, capital_base=debit,
                obligations_required=False, risk_level="ریسک محدود", risk_shape="ریسک محدود",
                best_use="سناریوی صعودی با ریسک کمتر از خرید Call تکی و سقف سود مشخص",
                management_note="اگر قیمت پایه نزدیک Strike بالایی رفت، شناسایی سود زودتر از سررسید را بررسی کن.",
            ))


def add_bear_put_spreads(rows, puts, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price):
    if puts.empty:
        return
    puts = puts.sort_values("strike_price")
    for _, high in puts.iterrows():
        lower_puts = puts[puts["strike_price"] < high["strike_price"]].sort_values("strike_price", ascending=False).head(5)
        for _, low in lower_puts.iterrows():
            if not has_buy_side(high) or not has_sell_side(low):
                continue
            k_high = to_float(high.get("strike_price")); k_low = to_float(low.get("strike_price"))
            debit = option_leg_price(high, "buy") - option_leg_price(low, "sell")
            width = k_high - k_low
            if pd.isna(debit) or debit <= 0 or width <= 0 or debit >= width:
                continue
            gross = min(max(k_high - to_float(target_price), 0), width)
            pnl = gross - debit
            rows.append(make_strategy_row(
                "Bear Put Spread", "اسپرد عمودی", "bearish",
                [{"row": high, "side": "buy"}, {"row": low, "side": "sell"}], underlying_scenario,
                expected_move_pct, expected_volatility_pct, high.get("ua_ticker"), high.get("end_date"), target_price,
                entry_cost=debit, net_credit=0, max_profit=width - debit, max_loss=debit,
                breakeven_1=k_high - debit, scenario_payoff=gross, scenario_pnl=pnl, capital_base=debit,
                obligations_required=False, risk_level="ریسک محدود", risk_shape="ریسک محدود",
                best_use="سناریوی نزولی با زیان محدود؛ جایگزین کنترل‌شده‌تر خرید Put تکی",
                management_note="برای بازار ایران، اول Bid خروجی هر دو پایه را کنترل کن.",
            ))


def add_long_straddles(rows, calls, puts, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price):
    if calls.empty or puts.empty:
        return
    merged_strikes = sorted(set(calls["strike_price"].dropna()).intersection(set(puts["strike_price"].dropna())))
    if not merged_strikes:
        return
    s0 = to_float(calls["ua_close_price"].dropna().iloc[0]) if not calls["ua_close_price"].dropna().empty else np.nan
    up = s0 * (1 + abs(expected_volatility_pct) / 100) if pd.notna(s0) else target_price
    down = s0 * (1 - abs(expected_volatility_pct) / 100) if pd.notna(s0) else target_price
    for k in merged_strikes[:20]:
        call = calls[calls["strike_price"].eq(k)].sort_values("orderbook_quality_score", ascending=False).iloc[0]
        put = puts[puts["strike_price"].eq(k)].sort_values("orderbook_quality_score", ascending=False).iloc[0]
        if not has_buy_side(call) or not has_buy_side(put):
            continue
        debit = option_leg_price(call, "buy") + option_leg_price(put, "buy")
        gross_directional = option_payoff(call, target_price) + option_payoff(put, target_price)
        gross_vol = max(option_payoff(call, up) + option_payoff(put, up), option_payoff(call, down) + option_payoff(put, down))
        gross = max(gross_directional, gross_vol)
        pnl = gross - debit
        rows.append(make_strategy_row(
            "Long Straddle", "نوسان شدید", "volatility",
            [{"row": call, "side": "buy"}, {"row": put, "side": "buy"}], underlying_scenario,
            expected_move_pct, expected_volatility_pct, call.get("ua_ticker"), call.get("end_date"), target_price,
            entry_cost=debit, max_profit=np.nan, max_loss=debit, breakeven_1=k - debit, breakeven_2=k + debit,
            scenario_payoff=gross, scenario_pnl=pnl, capital_base=debit, obligations_required=False,
            risk_level="ریسک محدود / پرمیوم سنگین", risk_shape="ریسک محدود",
            best_use="وقتی جهت حرکت نامطمئن است اما انتظار حرکت بزرگ یا شوک قیمتی داری",
            management_note="اگر حرکت بزرگ زودتر رخ داد، افست زودتر از سررسید را بررسی کن.",
        ))


def add_long_strangles(rows, calls, puts, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price):
    if calls.empty or puts.empty:
        return
    s0 = to_float(calls["ua_close_price"].dropna().iloc[0]) if not calls["ua_close_price"].dropna().empty else np.nan
    if pd.isna(s0):
        return
    otm_calls = calls[calls["strike_price"] > s0].sort_values(["moneyness_score", "orderbook_quality_score"], ascending=[False, False]).head(7)
    otm_puts = puts[puts["strike_price"] < s0].sort_values(["moneyness_score", "orderbook_quality_score"], ascending=[False, False]).head(7)
    up = s0 * (1 + abs(expected_volatility_pct) / 100)
    down = s0 * (1 - abs(expected_volatility_pct) / 100)
    for _, call in otm_calls.iterrows():
        for _, put in otm_puts.iterrows():
            if not has_buy_side(call) or not has_buy_side(put):
                continue
            kc = to_float(call.get("strike_price")); kp = to_float(put.get("strike_price"))
            debit = option_leg_price(call, "buy") + option_leg_price(put, "buy")
            gross_directional = option_payoff(call, target_price) + option_payoff(put, target_price)
            gross_vol = max(option_payoff(call, up) + option_payoff(put, up), option_payoff(call, down) + option_payoff(put, down))
            gross = max(gross_directional, gross_vol)
            pnl = gross - debit
            rows.append(make_strategy_row(
                "Long Strangle", "نوسان شدید", "volatility",
                [{"row": call, "side": "buy"}, {"row": put, "side": "buy"}], underlying_scenario,
                expected_move_pct, expected_volatility_pct, call.get("ua_ticker"), call.get("end_date"), target_price,
                entry_cost=debit, max_profit=np.nan, max_loss=debit, breakeven_1=kp - debit, breakeven_2=kc + debit,
                scenario_payoff=gross, scenario_pnl=pnl, capital_base=debit, obligations_required=False,
                risk_level="ریسک محدود / احتمال موفقیت کمتر از استرادل", risk_shape="ریسک محدود",
                best_use="انتظار نوسان شدید با هزینه کمتر از استرادل؛ نیازمند حرکت بزرگ‌تر",
                management_note="اگر هیچ‌کدام از دو سمت وارد ارزش نشد، فرسایش زمانی را جدی بگیر.",
            ))


def add_hedge_and_stockholder_strategies(rows, calls, puts, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price):
    if puts.empty and calls.empty:
        return
    s0 = np.nan
    for pool in [calls, puts]:
        if not pool.empty and not pool["ua_close_price"].dropna().empty:
            s0 = to_float(pool["ua_close_price"].dropna().iloc[0]); break
    if pd.isna(s0) or s0 <= 0:
        return

    # Protective Put: stock + long put
    candidate_puts = puts.sort_values(["moneyness_score", "orderbook_quality_score"], ascending=[False, False]).head(8)
    for _, put in candidate_puts.iterrows():
        if not has_buy_side(put):
            continue
        k = to_float(put.get("strike_price")); debit = option_leg_price(put, "buy")
        terminal_value = to_float(target_price) + option_payoff(put, target_price)
        cost = s0 + debit
        pnl = terminal_value - cost
        rows.append(make_strategy_row(
            "Protective Put", "هج و سهام‌داری", "hedge",
            [{"row": put, "side": "buy"}], underlying_scenario,
            expected_move_pct, expected_volatility_pct, put.get("ua_ticker"), put.get("end_date"), target_price,
            entry_cost=debit, max_profit=np.nan, max_loss=max(s0 + debit - k, 0), breakeven_1=s0 + debit,
            scenario_payoff=terminal_value, scenario_pnl=pnl, capital_base=cost, obligations_required=False,
            risk_level="بیمه نزول", risk_shape="هج / بیمه",
            best_use="برای کسی که دارایی پایه را دارد و می‌خواهد ریسک افت قیمت را محدود کند",
            management_note="این توصیه خرید اختیار خالی نیست؛ برای دارنده سهم پایه معنی دارد.",
            underlying_units=1, underlying_entry_price=s0,
        ))

    # Covered Call: stock + short call
    candidate_calls = calls[calls["strike_price"] >= s0].sort_values(["moneyness_score", "orderbook_quality_score"], ascending=[False, False]).head(8)
    for _, call in candidate_calls.iterrows():
        if not has_sell_side(call):
            continue
        k = to_float(call.get("strike_price")); credit = option_leg_price(call, "sell")
        terminal_value = min(to_float(target_price), k) + credit
        pnl = terminal_value - s0
        max_profit = max(k - s0 + credit, credit)
        rows.append(make_strategy_row(
            "Covered Call", "درآمدی / سهام‌داری", "neutral_income",
            [{"row": call, "side": "sell"}], underlying_scenario,
            expected_move_pct, expected_volatility_pct, call.get("ua_ticker"), call.get("end_date"), target_price,
            entry_cost=0, net_credit=credit, max_profit=max_profit, max_loss=np.nan, breakeven_1=s0 - credit,
            scenario_payoff=terminal_value, scenario_pnl=pnl, capital_base=s0, obligations_required=True,
            risk_level="ریسک سهام‌داری + سقف سود", risk_shape="ریسک سهام‌داری",
            best_use="برای دارنده سهم پایه در بازار خنثی تا کمی صعودی؛ دریافت پرمیوم در برابر سقف کردن سود",
            management_note="اگر سهم به محدوده اعمال نزدیک شد، ریسک واگذاری/تعهد را بررسی کن.",
            underlying_units=1, underlying_entry_price=s0,
        ))

    # Collar: stock + long put + short call
    otm_puts = puts[puts["strike_price"] <= s0].sort_values(["moneyness_score", "orderbook_quality_score"], ascending=[False, False]).head(6)
    otm_calls = calls[calls["strike_price"] >= s0].sort_values(["moneyness_score", "orderbook_quality_score"], ascending=[False, False]).head(6)
    for _, put in otm_puts.iterrows():
        for _, call in otm_calls.iterrows():
            if not has_buy_side(put) or not has_sell_side(call):
                continue
            kp = to_float(put.get("strike_price")); kc = to_float(call.get("strike_price"))
            if kp >= kc:
                continue
            net_cost = option_leg_price(put, "buy") - option_leg_price(call, "sell")
            terminal_stock = min(max(to_float(target_price), kp), kc)
            pnl = terminal_stock - s0 - net_cost
            max_loss = max(s0 + net_cost - kp, 0)
            max_profit = kc - s0 - net_cost
            rows.append(make_strategy_row(
                "Collar", "هج و سهام‌داری", "hedge",
                [{"row": put, "side": "buy"}, {"row": call, "side": "sell"}], underlying_scenario,
                expected_move_pct, expected_volatility_pct, put.get("ua_ticker"), put.get("end_date"), target_price,
                entry_cost=max(net_cost, 0), net_credit=max(-net_cost, 0), max_profit=max_profit, max_loss=max_loss,
                breakeven_1=s0 + net_cost, scenario_payoff=terminal_stock, scenario_pnl=pnl,
                capital_base=s0 + max(net_cost, 0), obligations_required=True,
                risk_level="هج با سقف سود", risk_shape="هج / بیمه",
                best_use="برای دارنده سهم که می‌خواهد زیان را محدود کند و حاضر است سقف سود بپذیرد",
                management_note="اگر Call فروخته‌شده نقدشوندگی ضعیف دارد، خروج از کولار سخت می‌شود.",
                underlying_units=1, underlying_entry_price=s0,
            ))


def add_credit_spreads_and_condor(rows, calls, puts, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price):
    if calls.empty and puts.empty:
        return
    s0 = np.nan
    for pool in [calls, puts]:
        if not pool.empty and not pool["ua_close_price"].dropna().empty:
            s0 = to_float(pool["ua_close_price"].dropna().iloc[0]); break

    # Bull Put Spread: sell higher put, buy lower put
    if not puts.empty:
        puts_sorted = puts.sort_values("strike_price")
        for _, short_put in puts_sorted.iterrows():
            lower_puts = puts_sorted[puts_sorted["strike_price"] < short_put["strike_price"]].sort_values("strike_price", ascending=False).head(4)
            for _, long_put in lower_puts.iterrows():
                if not has_sell_side(short_put) or not has_buy_side(long_put):
                    continue
                kh = to_float(short_put.get("strike_price")); kl = to_float(long_put.get("strike_price"))
                credit = option_leg_price(short_put, "sell") - option_leg_price(long_put, "buy")
                width = kh - kl
                if pd.isna(credit) or credit <= 0 or width <= 0 or credit >= width:
                    continue
                pnl = credit - max(kh - to_float(target_price), 0) + max(kl - to_float(target_price), 0)
                rows.append(make_strategy_row(
                    "Bull Put Spread", "اعتباری / تعهد فروش", "bullish",
                    [{"row": short_put, "side": "sell"}, {"row": long_put, "side": "buy"}], underlying_scenario,
                    expected_move_pct, expected_volatility_pct, short_put.get("ua_ticker"), short_put.get("end_date"), target_price,
                    entry_cost=0, net_credit=credit, max_profit=credit, max_loss=width - credit, breakeven_1=kh - credit,
                    scenario_payoff=np.nan, scenario_pnl=pnl, capital_base=width - credit, obligations_required=True,
                    risk_level="تعهد فروش با زیان محدود", risk_shape="تعهد فروش",
                    best_use="بازار خنثی تا صعودی؛ دریافت پرمیوم با زیان محدود نسبت به Naked Put",
                    management_note="فقط در صورت امکان فروش اختیار و کنترل وجه تضمین بررسی شود.",
                ))

    # Bear Call Spread: sell lower call, buy higher call
    if not calls.empty:
        calls_sorted = calls.sort_values("strike_price")
        for _, short_call in calls_sorted.iterrows():
            higher_calls = calls_sorted[calls_sorted["strike_price"] > short_call["strike_price"]].head(4)
            for _, long_call in higher_calls.iterrows():
                if not has_sell_side(short_call) or not has_buy_side(long_call):
                    continue
                kl = to_float(short_call.get("strike_price")); kh = to_float(long_call.get("strike_price"))
                credit = option_leg_price(short_call, "sell") - option_leg_price(long_call, "buy")
                width = kh - kl
                if pd.isna(credit) or credit <= 0 or width <= 0 or credit >= width:
                    continue
                pnl = credit - max(to_float(target_price) - kl, 0) + max(to_float(target_price) - kh, 0)
                rows.append(make_strategy_row(
                    "Bear Call Spread", "اعتباری / تعهد فروش", "bearish",
                    [{"row": short_call, "side": "sell"}, {"row": long_call, "side": "buy"}], underlying_scenario,
                    expected_move_pct, expected_volatility_pct, short_call.get("ua_ticker"), short_call.get("end_date"), target_price,
                    entry_cost=0, net_credit=credit, max_profit=credit, max_loss=width - credit, breakeven_1=kl + credit,
                    scenario_payoff=np.nan, scenario_pnl=pnl, capital_base=width - credit, obligations_required=True,
                    risk_level="تعهد فروش با زیان محدود", risk_shape="تعهد فروش",
                    best_use="بازار خنثی تا نزولی؛ دریافت پرمیوم با سقف زیان مشخص",
                    management_note="فقط در صورت امکان فروش اختیار و کنترل وجه تضمین بررسی شود.",
                ))

    # Iron Condor: combine OTM short put spread + OTM short call spread
    if calls.empty or puts.empty or pd.isna(s0):
        return
    short_puts = puts[puts["strike_price"] < s0].sort_values(["moneyness_score", "orderbook_quality_score"], ascending=[False, False]).head(4)
    long_puts = puts[puts["strike_price"] < s0].sort_values("strike_price", ascending=True).head(6)
    short_calls = calls[calls["strike_price"] > s0].sort_values(["moneyness_score", "orderbook_quality_score"], ascending=[False, False]).head(4)
    long_calls = calls[calls["strike_price"] > s0].sort_values("strike_price", ascending=False).head(6)
    for _, sp in short_puts.iterrows():
        lp_candidates = long_puts[long_puts["strike_price"] < sp["strike_price"]]
        for _, sc in short_calls.iterrows():
            lc_candidates = long_calls[long_calls["strike_price"] > sc["strike_price"]]
            if lp_candidates.empty or lc_candidates.empty:
                continue
            lp = lp_candidates.sort_values("strike_price", ascending=False).iloc[0]
            lc = lc_candidates.sort_values("strike_price").iloc[0]
            if not (has_sell_side(sp) and has_buy_side(lp) and has_sell_side(sc) and has_buy_side(lc)):
                continue
            ksp = to_float(sp.get("strike_price")); klp = to_float(lp.get("strike_price")); ksc = to_float(sc.get("strike_price")); klc = to_float(lc.get("strike_price"))
            put_width = ksp - klp; call_width = klc - ksc
            credit = option_leg_price(sp, "sell") + option_leg_price(sc, "sell") - option_leg_price(lp, "buy") - option_leg_price(lc, "buy")
            max_loss = max(put_width, call_width) - credit
            if credit <= 0 or max_loss <= 0:
                continue
            pnl = credit - max(ksp - to_float(target_price), 0) + max(klp - to_float(target_price), 0) - max(to_float(target_price) - ksc, 0) + max(to_float(target_price) - klc, 0)
            rows.append(make_strategy_row(
                "Iron Condor", "خنثی / اعتباری", "neutral_income",
                [{"row": sp, "side": "sell"}, {"row": lp, "side": "buy"}, {"row": sc, "side": "sell"}, {"row": lc, "side": "buy"}], underlying_scenario,
                expected_move_pct, expected_volatility_pct, sp.get("ua_ticker"), sp.get("end_date"), target_price,
                entry_cost=0, net_credit=credit, max_profit=credit, max_loss=max_loss, breakeven_1=ksp - credit, breakeven_2=ksc + credit,
                scenario_payoff=np.nan, scenario_pnl=pnl, capital_base=max_loss, obligations_required=True,
                risk_level="خنثی با تعهد کنترل‌شده", risk_shape="تعهد فروش",
                best_use="بازار رنج و کم‌نوسان؛ دریافت پرمیوم از دو طرف با ریسک محدود",
                management_note="اگر قیمت پایه به یکی از بال‌ها نزدیک شد، تعدیل یا خروج را بررسی کن.",
            ))


def filter_strategy_mode(df_strategy, strategy_mode):
    if df_strategy.empty:
        return df_strategy
    x = df_strategy.copy()
    if strategy_mode == "فقط استراتژی‌های خریدی با ریسک محدود":
        return x[(x["obligations_required"].eq("خیر")) & (x["risk_shape"].isin(["ریسک محدود", "هج / بیمه"]))]
    if strategy_mode == "اسپردهای عمودی":
        return x[x["strategy_family"].str.contains("اسپرد عمودی|اعتباری", na=False)]
    if strategy_mode == "نوسان شدید / بی‌جهت":
        return x[x["strategy_family"].str.contains("نوسان شدید", na=False)]
    if strategy_mode == "هج و سهام‌داری":
        return x[x["strategy_family"].str.contains("هج|سهام", na=False)]
    if strategy_mode == "درآمدی / تعهد فروش فقط برای بررسی":
        return x[x["obligations_required"].eq("بله")]
    return x


def build_strategy_dashboard(df, underlying_scenario, expected_move_pct, expected_volatility_pct, config, strategy_mode="هوشمند بر اساس سناریو"):
    rows = []
    if df.empty:
        return pd.DataFrame()
    max_per_type = int(config.get("max_strategy_pool_per_type", 12))
    # To keep Streamlit responsive, limit groups and build candidates from liquid/visible contracts only.
    grouped = df.groupby(["ua_ticker", "end_date"], dropna=False)
    for (ua, end), g in grouped:
        if g.empty:
            continue
        g = g.copy()
        g["strike_price"] = pd.to_numeric(g["strike_price"], errors="coerce")
        g = g[pd.notna(g["strike_price"])]
        if g.empty:
            continue
        s0 = to_float(g["ua_close_price"].dropna().iloc[0]) if not g["ua_close_price"].dropna().empty else np.nan
        if pd.isna(s0) or s0 <= 0:
            continue
        target_price = s0 * (1 + to_float(expected_move_pct, 0) / 100)
        calls = select_option_pool(g, "call", max_per_type)
        puts = select_option_pool(g, "put", max_per_type)

        add_single_option_strategies(rows, g, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price)
        add_bull_call_spreads(rows, calls, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price)
        add_bear_put_spreads(rows, puts, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price)
        add_long_straddles(rows, calls, puts, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price)
        add_long_strangles(rows, calls, puts, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price)
        add_hedge_and_stockholder_strategies(rows, calls, puts, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price)
        add_credit_spreads_and_condor(rows, calls, puts, underlying_scenario, expected_move_pct, expected_volatility_pct, target_price)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = filter_strategy_mode(out, strategy_mode)
    if out.empty:
        return out
    out = out.sort_values(
        ["strategy_score", "scenario_return_pct", "strategy_execution_score", "strategy_liquidity_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    # Drop near duplicate rows by exact strategy + contracts to keep the dashboard readable.
    out = out.drop_duplicates(subset=["strategy_name", "ua_ticker", "end_date", "contracts"], keep="first")
    return out.head(int(config.get("max_strategy_candidates", 250)))


STRATEGY_DISPLAY_COLS = [
    "strategy_action", "strategy_score", "strategy_name", "strategy_family", "ua_ticker", "end_date",
    "market_view", "risk_level", "risk_shape", "obligations_required", "contracts", "legs",
    "entry_cost", "net_credit", "capital_base", "max_profit", "max_loss", "breakeven_1", "breakeven_2",
    "target_price", "scenario_pnl", "scenario_return_pct", "strategy_execution_score",
    "strategy_liquidity_score", "strategy_orderbook_score", "strategy_exit_score", "alignment_score_strategy",
    "working_days_to_maturity", "best_use", "management_note", "strategy_warnings", "strategy_reason",
]

STRATEGY_PERSIAN_NAMES = {
    "strategy_action": "توصیه استراتژی",
    "strategy_score": "امتیاز استراتژی",
    "strategy_name": "نام استراتژی",
    "strategy_family": "خانواده استراتژی",
    "ua_ticker": "دارایی پایه",
    "end_date": "سررسید",
    "market_view": "دید بازار",
    "risk_level": "سطح ریسک",
    "risk_shape": "شکل ریسک",
    "obligations_required": "نیازمند تعهد فروش؟",
    "contracts": "قراردادها",
    "legs": "پایه‌های استراتژی",
    "entry_cost": "هزینه ورود خالص",
    "net_credit": "پرمیوم دریافتی خالص",
    "capital_base": "سرمایه مبنا برای بازده",
    "max_profit": "حداکثر سود تقریبی",
    "max_loss": "حداکثر زیان تقریبی",
    "breakeven_1": "Breakeven اول",
    "breakeven_2": "Breakeven دوم",
    "target_price": "قیمت سناریویی پایه",
    "scenario_pnl": "سود/زیان سناریویی",
    "scenario_return_pct": "بازده سناریویی (%)",
    "strategy_execution_score": "امتیاز اجرای استراتژی",
    "strategy_liquidity_score": "امتیاز نقدشوندگی ترکیب",
    "strategy_orderbook_score": "امتیاز اردربوک ترکیب",
    "strategy_exit_score": "امتیاز خروج ترکیب",
    "alignment_score_strategy": "امتیاز هم‌جهتی استراتژی",
    "working_days_to_maturity": "روز کاری تا سررسید",
    "best_use": "کاربرد مناسب",
    "management_note": "نکته مدیریت موقعیت",
    "strategy_warnings": "هشدارهای استراتژی",
    "strategy_reason": "دلیل رتبه‌بندی استراتژی",
}


def prepare_strategy_view(df_strategy, n=None):
    if df_strategy.empty:
        return pd.DataFrame()
    x = df_strategy.copy()
    if n is not None:
        x = x.head(n)
    cols = [c for c in STRATEGY_DISPLAY_COLS if c in x.columns]
    view = x[cols].rename(columns=STRATEGY_PERSIAN_NAMES)
    return add_row_number(view)


def style_strategy_table(df_view):
    def row_style(row):
        action = str(row.get("توصیه استراتژی", ""))
        obligation = str(row.get("نیازمند تعهد فروش؟", ""))
        risk_shape = str(row.get("شکل ریسک", ""))
        if "قابل اجرا" in action:
            color = "background-color: #e8f5e9"
        elif "کم‌حجم" in action or "زیر نظر" in action:
            color = "background-color: #fffde7"
        elif "فقط بررسی" in action or obligation == "بله":
            color = "background-color: #fff3e0"
        elif "عدم" in action:
            color = "background-color: #ffebee"
        elif "هج" in risk_shape:
            color = "background-color: #e3f2fd"
        else:
            color = ""
        return [color for _ in row]
    return df_view.style.apply(row_style, axis=1)


def top_strategy_cards(df_strategy, n=5):
    if df_strategy.empty:
        st.info("برای فیلترهای فعلی، استراتژی قابل رتبه‌بندی پیدا نشد.")
        return
    top = df_strategy.head(n)
    cols = st.columns(min(3, len(top)))
    for i, (_, row) in enumerate(top.iterrows()):
        action = str(row.get("strategy_action", ""))
        cls = "buy" if "قابل اجرا" in action else "watch" if "کم‌حجم" in action or "زیر نظر" in action or "فقط بررسی" in action else "avoid" if "عدم" in action else "neutral"
        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div class="rec-card {cls}">
                    <b>{row.get('strategy_name', '-')}</b><br>
                    دارایی پایه: {row.get('ua_ticker', '-')} | سررسید: {row.get('end_date', '-')}<br>
                    توصیه: <b>{row.get('strategy_action', '-')}</b><br>
                    امتیاز استراتژی: <b>{row.get('strategy_score', '-')}</b> از ۱۰۰<br>
                    بازده سناریویی: {row.get('scenario_return_pct', '-')}٪<br>
                    هزینه/اعتبار خالص: {format_number(row.get('entry_cost', 0), 0)} / {format_number(row.get('net_credit', 0), 0)}<br>
                    ریسک: {row.get('risk_level', '-')}<br>
                    قراردادها: {row.get('contracts', '-')}
                </div>
                """,
                unsafe_allow_html=True,
            )


def chart_strategy_scores(df_strategy):
    if px is None:
        no_plotly_warning(); return
    if df_strategy.empty:
        st.info("داده کافی برای نمودار امتیاز استراتژی وجود ندارد."); return
    top = df_strategy.head(12).sort_values("strategy_score", ascending=True)
    fig = px.bar(
        top, x="strategy_score", y="strategy_name", orientation="h", color="strategy_action",
        text="strategy_score", hover_data=["ua_ticker", "end_date", "scenario_return_pct", "strategy_execution_score", "obligations_required"],
        labels={"strategy_score": "امتیاز استراتژی", "strategy_name": "استراتژی", "strategy_action": "توصیه"},
        title="۱۲ استراتژی برتر بر اساس امتیاز نهایی استراتژی",
    )
    fig.update_layout(height=560)
    render_plotly_chart(fig, "chart_strategy_scores")


def chart_strategy_returns(df_strategy):
    if px is None:
        no_plotly_warning(); return
    if df_strategy.empty:
        st.info("داده کافی برای نمودار بازده استراتژی وجود ندارد."); return
    top = df_strategy.head(12).sort_values("scenario_return_pct", ascending=True)
    fig = px.bar(
        top, x="scenario_return_pct", y="strategy_name", orientation="h", color="strategy_family",
        text="scenario_return_pct", hover_data=["ua_ticker", "end_date", "strategy_score", "contracts"],
        labels={"scenario_return_pct": "بازده سناریویی (%)", "strategy_name": "استراتژی", "strategy_family": "خانواده"},
        title="بازده سناریویی استراتژی‌های برتر",
    )
    fig.update_layout(height=560)
    render_plotly_chart(fig, "chart_strategy_returns")


def chart_strategy_family(df_strategy):
    if px is None:
        no_plotly_warning(); return
    if df_strategy.empty:
        st.info("داده کافی برای نمودار خانواده استراتژی وجود ندارد."); return
    counts = df_strategy.groupby(["strategy_family", "strategy_action"]).size().reset_index(name="تعداد")
    fig = px.bar(
        counts, x="تعداد", y="strategy_family", color="strategy_action", orientation="h", text="تعداد",
        labels={"strategy_family": "خانواده استراتژی", "strategy_action": "توصیه"},
        title="ترکیب استراتژی‌ها بر اساس خانواده و توصیه",
    )
    fig.update_layout(height=520)
    render_plotly_chart(fig, "chart_strategy_family")


def chart_strategy_heatmap(df_strategy):
    if go is None:
        no_plotly_warning(); return
    if df_strategy.empty:
        st.info("داده کافی برای نقشه حرارتی استراتژی وجود ندارد."); return
    pivot = df_strategy.pivot_table(index="strategy_name", columns="risk_shape", values="strategy_score", aggfunc="mean")
    if pivot.empty:
        st.info("داده کافی برای نقشه حرارتی استراتژی وجود ندارد."); return
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index], colorbar=dict(title="امتیاز")))
    fig.update_layout(title="نقشه حرارتی امتیاز استراتژی بر اساس نوع ریسک", xaxis_title="شکل ریسک", yaxis_title="استراتژی", height=600)
    render_plotly_chart(fig, "chart_strategy_heatmap")


# =========================================================
# Data Load
# =========================================================

try:
    raw = load_options_data()
except Exception as e:
    st.error("دریافت داده آنلاین ناموفق بود.")
    st.write(str(e))
    st.stop()

missing_cols = validate_columns(raw)
if missing_cols:
    st.error("برخی ستون‌های ضروری در خروجی tseopt پیدا نشدند.")
    st.write("ستون‌های ناقص:", missing_cols)
    st.write("ستون‌های موجود:", raw.columns.tolist())
    st.stop()

available_underlyings = sorted([x for x in raw["ua_ticker"].dropna().astype(str).unique().tolist() if x.strip()])
underlying_options = [ALL_UNDERLYINGS_LABEL] + available_underlyings


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.header("تنظیمات اصلی")
    default_index = underlying_options.index("وبملت") if "وبملت" in underlying_options else (1 if len(underlying_options) > 1 else 0)
    selected_underlying = st.selectbox("دارایی پایه", underlying_options, index=default_index)

    option_type_label = st.selectbox("نوع اختیار برای جدول زنجیره", list(OPTION_TYPE_MAP.keys()), index=0)
    option_type_filter = OPTION_TYPE_MAP[option_type_label]

    expected_move_pct = st.number_input(
        "حرکت مورد انتظار دارایی پایه (%)",
        min_value=-50.0, max_value=50.0, value=8.0, step=0.5,
        help="عدد مثبت یعنی انتظار رشد؛ عدد منفی یعنی انتظار ریزش."
    )

    scenario_choice = st.selectbox(
        "سناریوی دارایی پایه",
        list(SCENARIO_MAP.keys()), index=0,
        help="در حالت خودکار، مثبت بودن حرکت مورد انتظار یعنی سناریوی صعودی و منفی بودن آن یعنی سناریوی نزولی."
    )
    underlying_scenario, scenario_label = resolve_scenario(scenario_choice, expected_move_pct)

    expected_volatility_pct = st.number_input(
        "انتظار نوسان مطلق برای استراتژی‌های بی‌جهت (%)",
        min_value=0.0, max_value=80.0, value=float(max(abs(expected_move_pct), 6.0)), step=0.5,
        help="برای استرادل، استرانگل و آیرون کاندور مهم است. این عدد جهت ندارد و فقط بزرگی حرکت احتمالی را نشان می‌دهد."
    )

    strategy_mode = st.selectbox(
        "حالت داشبورد استراتژی",
        STRATEGY_MODE_OPTIONS,
        index=0,
        help="در حالت هوشمند، مدل همه ساختارهای قابل محاسبه را می‌سازد و بر اساس سناریو، ریسک، نقدشوندگی و اجرا رتبه‌بندی می‌کند."
    )

    allow_obligation_strategies = st.checkbox(
        "نمایش استراتژی‌های دارای فروش اختیار به‌عنوان قابل اجرا",
        value=False,
        help="اگر فعال نباشد، استراتژی‌هایی مثل کاوردکال، بول پوت اسپرد، بیر کال اسپرد و آیرون کاندور فقط برای بررسی نمایش داده می‌شوند."
    )
    st.session_state["allow_obligation_strategies"] = allow_obligation_strategies

    daily_limit = st.number_input("دامنه نوسان روزانه", min_value=0.01, max_value=0.10, value=0.03, step=0.005, format="%.3f")

    st.divider()
    st.header("آستانه‌های تصمیم")
    min_days_to_maturity = st.number_input("حداقل روز کاری تا سررسید", min_value=0, max_value=90, value=3, step=1)
    max_breakeven_move_pct = st.number_input("حداکثر فاصله مجاز تا Breakeven (%)", min_value=1, max_value=100, value=20, step=1)

    st.divider()
    st.header("آستانه‌های نقدشوندگی")
    min_trades_value = st.number_input("حداقل ارزش معاملات اختیار", min_value=1_000_000, value=10_000_000_000, step=1_000_000)
    min_trades_volume = st.number_input("حداقل حجم معاملات اختیار", min_value=1, value=10_000, step=100)
    min_trades_num = st.number_input("حداقل تعداد معاملات", min_value=1, value=20, step=1)
    min_open_positions = st.number_input("حداقل موقعیت باز", min_value=1, value=1_000, step=100)
    min_depth = st.number_input("حداقل عمق مظنه خرید/فروش", min_value=1, value=1_000, step=100)

    with st.expander("تنظیمات پیشرفته زمان"):
        st.markdown(
            """
            نسبت روز کاری به تقویمی برای این است که مدل بداند همه روزهای تقویمی قابل معامله نیستند.
            حالت پیش‌فرض ۵/۷ است، یعنی تقریباً پنج روز کاری در هر هفت روز تقویمی.
            معمولاً لازم نیست آن را تغییر بدهی؛ فقط اگر هفته تعطیل خاصی داریم از کسر تعطیلات استفاده کن.
            """
        )
        working_day_ratio = st.number_input("نسبت روز کاری به تقویمی", min_value=0.1, max_value=1.0, value=5 / 7, step=0.01, format="%.2f")
        holiday_buffer = st.number_input("کسر روز کاری بابت تعطیلات احتمالی", min_value=0, max_value=30, value=0, step=1)

    with st.expander("تنظیمات پیشرفته وضعیت اختیار"):
        atm_threshold_pct = st.number_input("آستانه لب مرز اعمال (%)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
        deep_otm_threshold_pct = st.number_input("آستانه خیلی دور از ارزش ذاتی (%)", min_value=5.0, max_value=50.0, value=15.0, step=1.0)

    with st.expander("تنظیمات پیشرفته داشبورد استراتژی"):
        max_strategy_pool_per_type = st.number_input(
            "حداکثر قرارداد منتخب از هر نوع برای ساخت استراتژی",
            min_value=4, max_value=30, value=12, step=1,
            help="عدد بالاتر ترکیب‌های بیشتری می‌سازد ولی ممکن است داشبورد کندتر شود."
        )
        max_strategy_candidates = st.number_input(
            "حداکثر استراتژی قابل نمایش/خروجی",
            min_value=50, max_value=1000, value=250, step=50,
        )
        payoff_chart_range_pct = st.number_input(
            "دامنه نمودار سود/زیان استراتژی (%)",
            min_value=5, max_value=80, value=25, step=5,
            help="برای Strategy Lab مشخص می‌کند نمودار سود/زیان چند درصد اطراف قیمت فعلی پایه رسم شود."
        )

    refresh = st.button("به‌روزرسانی داده‌ها")

if refresh:
    st.cache_data.clear()

config = {
    "min_days_to_maturity": min_days_to_maturity,
    "max_breakeven_move_pct": max_breakeven_move_pct,
    "min_trades_value": min_trades_value,
    "min_trades_volume": min_trades_volume,
    "min_trades_num": min_trades_num,
    "min_open_positions": min_open_positions,
    "min_depth": min_depth,
    "working_day_ratio": working_day_ratio,
    "holiday_buffer": holiday_buffer,
    "atm_threshold_pct": atm_threshold_pct,
    "deep_otm_threshold_pct": deep_otm_threshold_pct,
    "max_strategy_pool_per_type": max_strategy_pool_per_type,
    "max_strategy_candidates": max_strategy_candidates,
    "payoff_chart_range_pct": payoff_chart_range_pct,
}


# =========================================================
# Process Data
# =========================================================

df_underlying = filter_underlying(raw, selected_underlying)

if df_underlying.empty:
    st.warning("برای دارایی پایه انتخاب‌شده داده‌ای پیدا نشد.")
    st.stop()

# Recommendations are always built from all Call/Put contracts of the selected underlying.
# The Call/Put selector is applied only to the option-chain table so a bearish scenario can still surface Put ideas.
df = enrich_options(
    df_underlying,
    daily_limit=daily_limit,
    underlying_scenario=underlying_scenario,
    scenario_label=scenario_label,
    expected_move_pct=expected_move_pct,
    config=config,
)

df_chain = filter_option_type(df, option_type_filter)
sort_cols = [c for c in ["ua_ticker", "end_date", "strike_price", "option_type"] if c in df_chain.columns]

df_strategy = build_strategy_dashboard(
    df,
    underlying_scenario=underlying_scenario,
    expected_move_pct=expected_move_pct,
    expected_volatility_pct=expected_volatility_pct,
    config=config,
    strategy_mode=strategy_mode,
)


# =========================================================
# Columns
# =========================================================

display_cols = [
    "trade_action", "holder_action", "ticker", "name", "option_type", "ua_ticker",
    "underlying_status", "underlying_scenario_score", "ua_close_price", "ua_last_price", "ua_yesterday_price",
    "underlying_day_change_pct", "underlying_intraday_pressure_pct", "expected_underlying_price",
    "strike_price", "end_date", "calendar_days_to_maturity", "working_days_to_maturity",
    "market_price", "entry_price", "intrinsic_value", "time_value", "breakeven",
    "required_move_to_strike_pct", "required_move_to_breakeven_pct", "days_needed_to_breakeven", "path_status", "path_score",
    "moneyness_pct", "moneyness_label", "moneyness_score", "premium_ratio_pct", "time_value_ratio_pct", "premium_risk_score",
    "bid_price", "bid_volume", "ask_price", "ask_volume", "bid_ask_spread_pct", "spread_score",
    "entry_depth_score", "exit_depth_score", "depth_balance_score", "orderbook_quality_score", "orderbook_label",
    "trades_value", "trades_volume", "trades_num", "open_positions", "volume_to_oi_ratio_pct", "liquidity_score",
    "scenario_payoff", "scenario_pnl", "scenario_return_pct", "scenario_value_score",
    "simple_leverage", "risk_bucket", "price_reliability_score", "expiry_quality_score", "alignment_score",
    "exit_capability_score", "exit_capability_label", "execution_score", "execution_risk", "opportunity_score",
    "warning_flags", "ranking_reason", "final_signal", "score_explanation", "signal_reason",
]

persian_names = {
    "trade_action": "توصیه معامله",
    "holder_action": "اقدام برای دارنده اختیار",
    "ticker": "نماد",
    "name": "نام قرارداد",
    "option_type": "نوع اختیار",
    "ua_ticker": "دارایی پایه",
    "underlying_status": "وضعیت دارایی پایه",
    "underlying_scenario_score": "امتیاز دارایی پایه",
    "ua_close_price": "قیمت پایانی پایه",
    "ua_last_price": "آخرین قیمت پایه",
    "ua_yesterday_price": "قیمت دیروز پایه",
    "underlying_day_change_pct": "تغییر پایانی پایه (%)",
    "underlying_intraday_pressure_pct": "فشار آخرین نسبت به پایانی (%)",
    "expected_underlying_price": "قیمت مورد انتظار پایه",
    "strike_price": "قیمت اعمال",
    "end_date": "سررسید",
    "calendar_days_to_maturity": "روز تقویمی تا سررسید",
    "working_days_to_maturity": "روز کاری تخمینی تا سررسید",
    "market_price": "قیمت بازار",
    "entry_price": "قیمت ورود تخمینی",
    "intrinsic_value": "ارزش ذاتی",
    "time_value": "ارزش زمانی",
    "breakeven": "Breakeven",
    "required_move_to_strike_pct": "حرکت لازم تا Strike (%)",
    "required_move_to_breakeven_pct": "فاصله تا Breakeven (%)",
    "days_needed_to_breakeven": "روز لازم تا Breakeven",
    "path_status": "وضعیت مسیر",
    "path_score": "امتیاز مسیر",
    "moneyness_pct": "درصد فاصله از ارزش ذاتی",
    "moneyness_label": "وضعیت اختیار نسبت به قیمت پایه",
    "moneyness_score": "امتیاز تعادل Strike",
    "premium_ratio_pct": "نسبت پرمیوم به پایه (%)",
    "time_value_ratio_pct": "نسبت ارزش زمانی به پرمیوم (%)",
    "premium_risk_score": "امتیاز ریسک پرمیوم",
    "bid_price": "بهترین خرید",
    "bid_volume": "حجم بهترین خرید",
    "ask_price": "بهترین فروش",
    "ask_volume": "حجم بهترین فروش",
    "bid_ask_spread_pct": "اسپرد مظنه (%)",
    "spread_score": "امتیاز اسپرد",
    "entry_depth_score": "امتیاز عمق ورود",
    "exit_depth_score": "امتیاز عمق خروج",
    "depth_balance_score": "امتیاز تعادل عمق",
    "orderbook_quality_score": "امتیاز اردربوک اختیار",
    "orderbook_label": "وضعیت اردربوک اختیار",
    "trades_value": "ارزش معاملات",
    "trades_volume": "حجم معاملات",
    "trades_num": "تعداد معاملات",
    "open_positions": "موقعیت باز",
    "volume_to_oi_ratio_pct": "نسبت حجم به موقعیت باز (%)",
    "liquidity_score": "امتیاز نقدشوندگی",
    "scenario_payoff": "Payoff سناریویی",
    "scenario_pnl": "سود/زیان سناریویی",
    "scenario_return_pct": "بازده سناریویی (%)",
    "scenario_value_score": "امتیاز بازده سناریویی",
    "simple_leverage": "اهرم ساده",
    "risk_bucket": "دسته ریسک",
    "price_reliability_score": "امتیاز اعتبار قیمت",
    "expiry_quality_score": "امتیاز کیفیت سررسید",
    "alignment_score": "امتیاز هم‌جهتی",
    "exit_capability_score": "امتیاز قابلیت خروج",
    "exit_capability_label": "وضعیت قابلیت خروج",
    "execution_score": "امتیاز اجرا",
    "execution_risk": "ریسک اجرا",
    "opportunity_score": "امتیاز نهایی فرصت",
    "warning_flags": "هشدارها",
    "ranking_reason": "دلیل رتبه‌بندی",
    "final_signal": "سیگنال مدل",
    "score_explanation": "توضیح امتیازها",
    "signal_reason": "توضیح سیگنال",
}


# =========================================================
# UI Header - compact, no large brand box
# =========================================================

st.markdown("### داشبورد تصمیم و استراتژی آپشن — نسخه استراتژی‌محور")
st.caption(f"{APP_BRAND} — {APP_VERSION}")


# =========================================================
# Main Navigation
# =========================================================

PAGE_OPTIONS = [
    "داشبورد استراتژی",
    "توصیه خرید/فروش",
    "تحلیل دارایی پایه و اردربوک",
    "زنجیره آپشن",
    "نمودارهای تصمیم",
    "بررسی توضیحی قرارداد",
    "داده خام",
]

st.markdown(
    """
    <div class="tip-box">
    <b>انتخاب بخش داشبورد:</b> اگر سربرگ‌های Streamlit در صفحه شما کامل دیده نمی‌شود، از نوار زیر استفاده کن.
    بخش <b>داشبورد استراتژی</b> عمداً اولین گزینه است تا همیشه قابل مشاهده و قابل انتخاب باشد.
    </div>
    """,
    unsafe_allow_html=True,
)

active_page = st.radio(
    "انتخاب بخش",
    PAGE_OPTIONS,
    index=0,
    horizontal=True,
    key="main_page_navigation",
    label_visibility="collapsed",
)

st.divider()


if active_page == "داشبورد استراتژی":
    st.subheader("داشبورد انتخاب استراتژی")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("استراتژی‌های ساخته‌شده", len(df_strategy))
    c2.metric("سناریوی پایه", scenario_label)
    c3.metric("حرکت مورد انتظار", f"{expected_move_pct}%")
    c4.metric("نوسان مطلق مورد انتظار", f"{expected_volatility_pct}%")

    st.markdown(
        """
        <div class="tip-box">
        این بخش دیگر فقط دنبال «خرید یا عدم خرید یک اختیار» نیست؛ مدل ترکیب‌های عملی مثل
        <b>Bull Call Spread</b>، <b>Bear Put Spread</b>، <b>Long Straddle</b>، <b>Long Strangle</b>،
        <b>Protective Put</b>، <b>Collar</b>، <b>Covered Call</b> و چند ساختار اعتباری را می‌سازد و
        با سناریو، بازده سناریویی، ریسک، اردربوک و قابلیت اجرا رتبه‌بندی می‌کند.
        <br><br>
        نکته مهم: ساختارهایی که پایه فروش اختیار دارند، در حالت پیش‌فرض فقط برای بررسی نمایش داده می‌شوند؛ چون در بازار ایران ممکن است نیازمند مجوز، وجه تضمین یا محدودیت کارگزاری باشند.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("منطق مدل استراتژی چیست؟", expanded=False):
        st.markdown(
            """
            <div class="rtl-box">
            مدل استراتژی ابتدا قراردادهای قابل اتکاتر هر سررسید را انتخاب می‌کند، سپس ترکیب‌هایی می‌سازد که با متن آموزشی استراتژی‌ها هم‌خوانی دارند:
            اسپردهای صعودی/نزولی برای حرکت جهت‌دار، استرادل و استرانگل برای نوسان شدید، کولار و پرتکتیو پوت برای هج، و کاوردکال/اسپردهای اعتباری برای حالت درآمدی یا خنثی.
            <br><br>
            قیمت ورود پایه خرید با <b>Ask</b> و قیمت فروش/دریافت پرمیوم با <b>Bid</b> محاسبه می‌شود تا مدل به قیمت قابل معامله نزدیک‌تر باشد.
            امتیاز نهایی استراتژی از ترکیب بازده سناریویی، کیفیت اجرای کل پایه‌ها، هم‌جهتی با سناریو، شکل ریسک، نقدشوندگی، زمان تا سررسید و قابلیت خروج ساخته می‌شود.
            </div>
            """,
            unsafe_allow_html=True,
        )

    top_strategy_cards(df_strategy, n=5)

    st.divider()
    st.subheader("جدول رتبه‌بندی استراتژی‌ها")
    if df_strategy.empty:
        st.info("برای تنظیمات فعلی استراتژی قابل رتبه‌بندی پیدا نشد. دارایی پایه، سررسید، انتظار نوسان یا حالت داشبورد استراتژی را تغییر بده.")
    else:
        strategy_view = prepare_strategy_view(df_strategy, n=60)
        st.dataframe(style_strategy_table(strategy_view), use_container_width=True, height=620)

        st.subheader("نمودارهای استراتژی")
        chart_strategy_scores(df_strategy)
        chart_strategy_returns(df_strategy)
        chart_strategy_family(df_strategy)
        chart_strategy_heatmap(df_strategy)

        selected_strategy_idx = st.selectbox(
            "بررسی توضیحی یک استراتژی",
            options=list(df_strategy.index[:min(len(df_strategy), 80)]),
            format_func=lambda i: f"{df_strategy.loc[i, 'strategy_name']} | {df_strategy.loc[i, 'ua_ticker']} | امتیاز {df_strategy.loc[i, 'strategy_score']} | {df_strategy.loc[i, 'contracts']}",
        )
        sr = df_strategy.loc[selected_strategy_idx]
        st.markdown(
            f"""
            <div class="rtl-box">
            <b>استراتژی:</b> {sr.get('strategy_name', '-')}<br>
            <b>پایه‌ها:</b> {sr.get('legs', '-')}<br><br>
            <b>توصیه:</b> {sr.get('strategy_action', '-')}<br>
            <b>دلیل رتبه‌بندی:</b> {sr.get('strategy_reason', '-')}<br>
            <b>کاربرد مناسب:</b> {sr.get('best_use', '-')}<br>
            <b>نکته مدیریت موقعیت:</b> {sr.get('management_note', '-')}<br>
            <b>هشدارها:</b> {sr.get('strategy_warnings', '-')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Strategy Lab: پرونده سود/زیان استراتژی منتخب")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("هزینه ورود خالص", format_number(sr.get("entry_cost"), 0))
        m2.metric("پرمیوم دریافتی خالص", format_number(sr.get("net_credit"), 0))
        m3.metric("حداکثر سود تقریبی", format_number(sr.get("max_profit"), 0))
        m4.metric("حداکثر زیان تقریبی", format_number(sr.get("max_loss"), 0))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Breakeven اول", format_number(sr.get("breakeven_1"), 0))
        m6.metric("Breakeven دوم", format_number(sr.get("breakeven_2"), 0))
        m7.metric("سرمایه مبنا", format_number(sr.get("capital_base"), 0))
        m8.metric("بازده سناریویی", f"{sr.get('scenario_return_pct', '-')}%")

        st.markdown(
            """
            <div class="tip-box">
            این بخش سود/زیان تقریبی استراتژی را در سررسید نشان می‌دهد. پایه خرید با Ask و پایه فروش با Bid محاسبه شده است.
            اگر استراتژی شامل سهم پایه باشد، مثل Protective Put، Covered Call یا Collar، سود/زیان خود سهم هم به نمودار اضافه می‌شود.
            </div>
            """,
            unsafe_allow_html=True,
        )

        leg_details = strategy_leg_details_table(sr)
        if not leg_details.empty:
            st.markdown("**پایه‌های قابل اجرای استراتژی**")
            st.dataframe(leg_details, use_container_width=True, height=240)

        chart_strategy_payoff(sr, range_pct=config.get("payoff_chart_range_pct", 25))

        scenario_table = build_strategy_scenario_table(sr, expected_move_pct=expected_move_pct, expected_volatility_pct=expected_volatility_pct)
        if not scenario_table.empty:
            st.markdown("**جدول سناریوهای قیمت پایه**")
            st.dataframe(add_row_number(scenario_table), use_container_width=True, height=360)


elif active_page == "توصیه خرید/فروش":
    st.subheader("خلاصه تصمیم")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("قراردادهای بررسی‌شده", len(df))
    col2.metric("میانگین امتیاز فرصت", round(df["opportunity_score"].mean(), 1))
    col3.metric("سناریوی پایه", scenario_label)
    col4.metric("حرکت مورد انتظار", f"{expected_move_pct}%")

    if underlying_scenario == "bearish":
        st.caption("در سناریوی نزولی، پیشنهادها به‌صورت پیش‌فرض Putهای هم‌جهت را جلوتر می‌آورند؛ حتی اگر فیلتر زنجیره روی Call باشد.")
    elif underlying_scenario == "bullish":
        st.caption("در سناریوی صعودی، پیشنهادها به‌صورت پیش‌فرض Callهای هم‌جهت را جلوتر می‌آورند؛ حتی اگر فیلتر زنجیره روی Put باشد.")

    st.markdown(
        """
        <div class="tip-box">
        تمرکز این بخش روی تصمیم عملی است: <b>قابل خرید با کنترل ریسک</b>، <b>خرید فقط کم‌حجم و پله‌ای</b>، <b>فعلاً خرید نکن؛ فقط زیر نظر بگیر</b> یا <b>عدم معامله</b>.
        اگر دارنده اختیار هستی، ستون «اقدام برای دارنده اختیار» کمک می‌کند درباره نگهداری، کاهش ریسک یا خروج فکر کنی.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("بهترین بازده سناریویی یعنی چه و چرا تغییر می‌کند؟", expanded=False):
        st.markdown(
            """
            <div class="rtl-box">
            <b>بازده سناریویی</b> یعنی اگر حرکت مورد انتظار تو برای دارایی پایه محقق شود،
            ارزش پایانی تقریبی اختیار نسبت به قیمت ورود تخمینی چند درصد سود یا زیان می‌دهد.
            <br><br>
            برای مثال اگر حرکت مورد انتظار را مثبت وارد کنی، مدل قیمت مورد انتظار پایه را بالاتر می‌برد
            و معمولاً Callها بازده سناریویی بهتری می‌گیرند. اگر عدد را منفی وارد کنی، قیمت مورد انتظار پایه پایین‌تر می‌آید
            و معمولاً Putها بهتر می‌شوند.
            <br><br>
            این عدد زیاد تغییر می‌کند چون به چند چیز حساس است: <b>حرکت مورد انتظار</b>، <b>قیمت ورود یا Ask</b>،
            <b>قیمت اعمال</b>، <b>وضعیت Call/Put</b>، و <b>قیمت آنلاین بازار</b>. قراردادهای خیلی ارزان یا خیلی دور از اعمال
            ممکن است بازده درصدی بسیار بزرگی نشان بدهند، اما الزاماً بهترین توصیه خرید نیستند؛ چون ممکن است مسیر تا Breakeven سخت،
            اردربوک ضعیف یا ریسک اجرا بالا داشته باشند.
            <br><br>
            بنابراین «بیشترین بازده سناریویی» فقط یک زاویه دید است؛ توصیه خرید/فروش نهایی همچنان بر اساس امتیاز فرصت، مسیر،
            نقدشوندگی، اردربوک و هم‌جهتی با سناریو تصمیم می‌گیرد.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("فاصله تا Breakeven یعنی چه؟", expanded=False):
        st.markdown(
            """
            <div class="rtl-box">
            <b>فاصله تا Breakeven</b> یعنی قیمت دارایی پایه چند درصد باید در جهت درست حرکت کند تا خریدار اختیار به نقطه سربه‌سر برسد.
            برای Call، نقطه سربه‌سر تقریباً برابر است با <b>Strike + قیمت ورود</b>؛ برای Put، تقریباً برابر است با <b>Strike - قیمت ورود</b>.
            اگر این فاصله زیاد باشد، حتی ارزان بودن اختیار کافی نیست؛ چون ممکن است تا سررسید روز کاری کافی برای رسیدن به آن سطح وجود نداشته باشد.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("منطق وزن‌دهی امتیاز نهایی فرصت", expanded=False):
        st.markdown(
            """
            <div class="rtl-box">
            وزن‌ها برای تصمیم عملی تنظیم شده‌اند: بازده سناریویی و امکان رسیدن به Breakeven وزن بالایی دارند،
            اما اردربوک، نقدشوندگی و قابلیت خروج هم وزن جدی گرفته‌اند تا قرارداد صرفاً روی کاغذ جذاب نباشد.
            ساختار تقریبی امتیاز: مسیر ۱۴٪، بازده سناریویی ۱۶٪، دارایی پایه ۱۱٪، اردربوک ۱۳٪، نقدشوندگی ۹٪، اجرا ۷٪، قابلیت خروج ۷٪، تعادل Strike شش درصد، اعتبار قیمت ۴٪، کیفیت سررسید ۴٪، ریسک پرمیوم ۲٪ و هم‌جهتی با سناریو ۷٪.
            علاوه بر وزن‌ها، چند شرط سخت هم وجود دارد: نبود Bid یا Ask، عدم هم‌جهتی Call/Put با سناریو، نبود روز کاری کافی، و بازده سناریویی غیرمثبت می‌تواند توصیه خرید را حذف کند.
            </div>
            """,
            unsafe_allow_html=True,
        )

    categorized = categorized_recommendations(df)
    make_recommendation_cards(categorized)

    st.divider()
    if selected_underlying == ALL_UNDERLYINGS_LABEL:
        st.subheader("۵ گزینه برتر برای هر دارایی پایه")
        top = top_by_underlying(df, n=5)
    else:
        st.subheader("۵ گزینه برتر برای دارایی پایه انتخاب‌شده")
        top = top_candidates(df, n=5)

    if top.empty:
        st.info("گزینه‌ای که شروط اولیه را پاس کند پیدا نشد.")
    else:
        # top_candidates و top_by_underlying از قبل بر اساس امتیاز فرصت نزولی مرتب شده‌اند.
        # اینجا دوباره sort صعودی انجام نمی‌دهیم تا بهترین گزینه‌ها در بالای جدول بمانند.
        top_view = prepare_view(top, display_cols, persian_names, sort_cols=None)
        st.dataframe(style_decision_table(top_view), use_container_width=True, height=420)
        chart_top_recommendations(df)


elif active_page == "تحلیل دارایی پایه و اردربوک":
    st.subheader("تحلیل دارایی پایه و اردربوک اختیار")
    st.markdown(
        """
        <div class="tip-box">
        در این نسخه، تحلیل دارایی پایه از قیمت پایانی، آخرین قیمت و قیمت دیروز پایه ساخته می‌شود.
        تحلیل اردربوک هم فعلاً روی اردربوک خود اختیار تمرکز دارد: بهترین خرید، بهترین فروش، عمق سفارش و اسپرد.
        اگر نسخه‌های بعدی داده کامل اردربوک دارایی پایه را مستقیم از tseopt بگیریم، این بخش دقیق‌تر می‌شود.
        </div>
        """,
        unsafe_allow_html=True,
    )
    underlying_summary_cols = [
        "ua_ticker", "ua_close_price", "ua_last_price", "ua_yesterday_price",
        "underlying_day_change_pct", "underlying_intraday_pressure_pct", "expected_underlying_price",
        "underlying_scenario_score", "underlying_status",
    ]
    u = df[underlying_summary_cols].drop_duplicates("ua_ticker").copy().rename(columns=persian_names)
    st.dataframe(add_row_number(u), use_container_width=True, height=300)

    ob_cols = [
        "ticker", "ua_ticker", "bid_price", "bid_volume", "ask_price", "ask_volume",
        "bid_ask_spread_pct", "entry_depth_score", "exit_depth_score", "depth_balance_score",
        "orderbook_quality_score", "orderbook_label", "exit_capability_score", "exit_capability_label",
        "trade_action", "opportunity_score",
    ]
    ob_view = prepare_view(df.sort_values("orderbook_quality_score", ascending=False).head(30), ob_cols, persian_names)
    st.dataframe(style_decision_table(ob_view), use_container_width=True, height=520)


elif active_page == "زنجیره آپشن":
    st.subheader("زنجیره آپشن")
    st.markdown(
        """
        <div class="tip-box">
        برای استفاده از زنجیره آپشن، اول ستون «توصیه معامله» و «امتیاز نهایی فرصت» را ببین.
        بعد «فاصله تا Breakeven»، «روز لازم تا Breakeven»، «روز کاری تا سررسید» و «امتیاز اردربوک اختیار» را کنترل کن.
        رنگ‌های کم‌رنگ فقط وضعیت اختیار نسبت به قیمت پایه را نشان می‌دهند: سبز یعنی دارای ارزش ذاتی، زرد یعنی لب مرز اعمال، آبی یعنی بدون ارزش ذاتی، نارنجی یعنی خیلی دور از ارزش ذاتی.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if df_chain.empty:
        st.warning("برای فیلتر نوع اختیار انتخاب‌شده در زنجیره، داده‌ای پیدا نشد.")
    else:
        view = prepare_view(df_chain, display_cols, persian_names, sort_cols=sort_cols)
        st.dataframe(style_decision_table(view), use_container_width=True, height=700)


elif active_page == "نمودارهای تصمیم":
    st.subheader("نمودارهای تصمیم‌محور")
    chart_action_composition(df)
    chart_top_recommendations(df)
    chart_return_bar(df)
    chart_breakeven_bar(df)
    chart_path_feasibility(df)
    chart_orderbook_bar(df)
    chart_orderbook_tree(df)
    chart_score_breakdown_tree(df)
    chart_heatmap(df)


elif active_page == "بررسی توضیحی قرارداد":
    st.subheader("بررسی توضیحی یک قرارداد")
    contract_list = df["ticker"].astype(str).tolist()
    selected_contract = st.selectbox("انتخاب قرارداد", contract_list)
    selected_row = df[df["ticker"].astype(str) == selected_contract].iloc[0]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("توصیه معامله", selected_row["trade_action"])
    d2.metric("امتیاز فرصت", selected_row["opportunity_score"])
    d3.metric("وضعیت اختیار", selected_row["moneyness_label"])
    d4.metric("اردربوک", selected_row["orderbook_label"])
    st.markdown(
        f"""
        <div class="rtl-box">
        <b>توضیح سیگنال:</b><br>{selected_row['signal_reason']}<br><br>
        <b>توضیح امتیازها:</b><br>{selected_row['score_explanation']}<br><br>
        <b>فاصله تا Breakeven یعنی چه؟</b><br>
        یعنی دارایی پایه چند درصد باید در جهت سناریوی تو حرکت کند تا خریدار اختیار به نقطه سربه‌سر برسد.
        هرچه این عدد کمتر باشد مسیر کوتاه‌تر است، اما باید نقدشوندگی، اردربوک و زمان کاری هم مناسب باشند.
        </div>
        """,
        unsafe_allow_html=True,
    )


export_cols = [c for c in display_cols + ["app_brand", "app_version", "generated_at"] if c in df.columns]
csv_data = df[export_cols].to_csv(index=False).encode("utf-8-sig")
strategy_export_cols = [c for c in STRATEGY_DISPLAY_COLS + ["app_brand", "app_version", "generated_at"] if c in df_strategy.columns]
csv_strategy_data = df_strategy[strategy_export_cols].to_csv(index=False).encode("utf-8-sig") if not df_strategy.empty else b""

if active_page == "داده خام":
    st.subheader("داده خام و خروجی")
    st.download_button(
        label="دانلود خروجی CSV قراردادها با برند",
        data=csv_data,
        file_name=f"{APP_BRAND}_{selected_underlying}_options_mvp_v101.csv",
        mime="text/csv",
    )
    st.download_button(
        label="دانلود خروجی CSV استراتژی‌ها با برند",
        data=csv_strategy_data,
        file_name=f"{APP_BRAND}_{selected_underlying}_strategies_mvp_v101.csv",
        mime="text/csv",
        disabled=df_strategy.empty,
    )
    with st.expander("نمایش ستون‌ها و داده خام دریافتی از tseopt"):
        st.write(raw.columns.tolist())
        st.dataframe(raw.head(30), use_container_width=True)


st.markdown(
    f"""
    <hr>
    <div style="text-align:center;color:#777;font-size:13px;line-height:1.8">
        {APP_BRAND} — {APP_VERSION}<br>
        مالک مدل و خروجی: {APP_OWNER}<br>
        استفاده از خروجی این ابزار بدون ذکر منبع و مالک مدل مجاز نیست.
    </div>
    """,
    unsafe_allow_html=True,
)
