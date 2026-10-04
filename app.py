import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Kiểm Định Chiến Lược SMA + OBV",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-train {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-test {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding-left: 18px;
        padding-right: 18px;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. XỬ LÝ & TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
def prepare_stock_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa dữ liệu theo cấu trúc chuẩn:
    Date (index), Open, High, Low, Close, Volume
    """
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.lower()

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing}. Cần có: Date, Open, High, Low, Close, Volume.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }
    df = df.rename(columns=rename_map)
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.set_index("date")
    return df


@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Tải dữ liệu mẫu mặc định (ACB.csv hoặc ACB (1).csv)."""
    import os
    candidates = ["ACB.csv", "ACB (1).csv", "./ACB.csv", "./ACB (1).csv"]
    for path in candidates:
        if os.path.exists(path):
            df_raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            return prepare_stock_data(df_raw)
    raise FileNotFoundError("Không tìm thấy tệp ACB.csv hoặc ACB (1).csv trong thư mục làm việc.")


# ==============================================================================
# 3. TÍNH TOÁN CHỈ BÁO & TÍN HIỆU CHIẾN LƯỢC
# ==============================================================================
def calculate_indicators(df: pd.DataFrame, ma_short: int, ma_long: int, obv_window: int) -> pd.DataFrame:
    """
    Tính các đường SMA, chỉ báo OBV và đường trung bình động OBV_MA.
    """
    data = df.copy()

    # 1. SMA Indicators
    data["SMA_Short"] = ta.trend.SMAIndicator(close=data["Close"], window=int(ma_short)).sma_indicator()
    data["SMA_Long"] = ta.trend.SMAIndicator(close=data["Close"], window=int(ma_long)).sma_indicator()

    # 2. OBV Indicators
    data["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=data["Close"], volume=data["Volume"]).on_balance_volume()
    data["OBV_MA"] = data["OBV"].rolling(window=int(obv_window)).mean()

    return data


def generate_strategy_signals(data: pd.DataFrame, ma_short: int, ma_long: int, obv_window: int) -> pd.DataFrame:
    """
    Sinh tín hiệu MUA (+1), BÁN (-1) cho từng chiến lược theo quy chuẩn notebook:
    - SMA Crossover
    - OBV vs OBV_MA Crossover
    - SMA + OBV (AND)
    - SMA + OBV (OR)
    """
    df = calculate_indicators(data, ma_short, ma_long, obv_window)

    # ------------------ 1. SMA SIGNALS ------------------
    pos_sma = pd.Series(0.0, index=df.index, name="pos_sma")
    if ma_short < ma_long:
        buy_sma = (df["SMA_Short"] > df["SMA_Long"]) & (df["SMA_Short"].shift(1) <= df["SMA_Long"].shift(1))
        sell_sma = (df["SMA_Short"] < df["SMA_Long"]) & (df["SMA_Short"].shift(1) >= df["SMA_Long"].shift(1))
        pos_sma.loc[buy_sma] = 1.0
        pos_sma.loc[sell_sma] = -1.0
    df["pos_sma"] = pos_sma

    # ------------------ 2. OBV SIGNALS ------------------
    pos_obv = pd.Series(0.0, index=df.index, name="pos_obv")
    buy_obv = (df["OBV"] > df["OBV_MA"]) & (df["OBV"].shift(1) <= df["OBV_MA"].shift(1))
    sell_obv = (df["OBV"] < df["OBV_MA"]) & (df["OBV"].shift(1) >= df["OBV_MA"].shift(1))
    pos_obv.loc[buy_obv] = 1.0
    pos_obv.loc[sell_obv] = -1.0
    df["pos_obv"] = pos_obv

    # ------------------ 3. COMBINED AND SIGNALS ------------------
    # Chuẩn Notebook: Cùng phiên phát tín hiệu BUY (hoặc SELL)
    pos_and = pd.Series(0.0, index=df.index, name="pos_and")
    buy_and = (pos_sma == 1.0) & (pos_obv == 1.0)
    sell_and = (pos_sma == -1.0) & (pos_obv == -1.0)
    pos_and.loc[buy_and] = 1.0
    pos_and.loc[sell_and] = -1.0
    df["pos_and"] = pos_and

    # ------------------ 4. COMBINED OR SIGNALS ------------------
    pos_or = pd.Series(0.0, index=df.index, name="pos_or")
    buy_or = (pos_sma == 1.0) | (pos_obv == 1.0)
    sell_or = (pos_sma == -1.0) | (pos_obv == -1.0)
    pos_or.loc[buy_or] = 1.0
    pos_or.loc[sell_or] = -1.0
    df["pos_or"] = pos_or

    return df


# ==============================================================================
# 4. BACKTEST ENGINE (MÔ PHỎNG CHI TIẾT & CHUẨN XÁC)
# ==============================================================================
def compute_sharpe(returns: pd.Series, window: int = 252) -> float:
    """Tính toán Sharpe Ratio hàng năm chuẩn hóa theo 252 phiên giao dịch."""
    valid_returns = returns.dropna()
    if valid_returns.empty or valid_returns.std() == 0:
        return 0.0
    return float(np.sqrt(window) * valid_returns.mean() / valid_returns.std())


def run_backtest_simulation(
    df_data: pd.DataFrame,
    signal_col: str,
    cash: float = 1_000_000.0,
    commission: float = 0.0
):
    """
    Backtest Engine mô phỏng logic tương đương backtesting.py với:
    - trade_on_close = True (Khớp lệnh tại giá Close của phiên phát tín hiệu)
    - exclusive_orders = True (Toàn bộ tài khoản mua hoặc bán ra tiền mặt)
    - Tính toán chi tiết: Return, Sharpe, Max Drawdown, Win Rate, Danh sách lệnh (Trades), Equity Curve.
    """
    dates = df_data.index
    closes = df_data["Close"].values
    signals = df_data[signal_col].values
    n = len(df_data)

    current_cash = cash
    shares = 0.0
    in_position = False
    entry_price = 0.0
    entry_date = None

    equity_curve = np.zeros(n)
    trades = []

    for i in range(n):
        sig = signals[i]
        price = closes[i]
        dt = dates[i]

        # Kiểm tra tín hiệu MUA
        if sig == 1.0 and not in_position:
            cost = current_cash * (1.0 - commission)
            shares = cost / price
            current_cash = 0.0
            in_position = True
            entry_price = price
            entry_date = dt

        # Kiểm tra tín hiệu BÁN
        elif sig == -1.0 and in_position:
            proceeds = shares * price * (1.0 - commission)
            trade_pnl = proceeds - (shares * entry_price)
            trade_return = ((price / entry_price) * (1.0 - commission)**2 - 1.0) * 100.0
            trades.append({
                "Entry Date": entry_date,
                "Exit Date": dt,
                "Entry Price": entry_price,
                "Exit Price": price,
                "Shares": shares,
                "PnL": trade_pnl,
                "Return [%]": trade_return,
                "Duration (days)": (dt - entry_date).days
            })
            current_cash = proceeds
            shares = 0.0
            in_position = False
            entry_price = 0.0
            entry_date = None

        # Cập nhật giá trị tài khoản tại cuối phiên
        if in_position:
            equity_curve[i] = shares * price
        else:
            equity_curve[i] = current_cash

    # Nếu cuối giai đoạn vẫn còn giữ vị thế, đóng vị thế ảo để tính PnL
    if in_position:
        final_price = closes[-1]
        proceeds = shares * final_price * (1.0 - commission)
        trade_pnl = proceeds - (shares * entry_price)
        trade_return = ((final_price / entry_price) * (1.0 - commission)**2 - 1.0) * 100.0
        trades.append({
            "Entry Date": entry_date,
            "Exit Date": dates[-1],
            "Entry Price": entry_price,
            "Exit Price": final_price,
            "Shares": shares,
            "PnL": trade_pnl,
            "Return [%]": trade_return,
            "Duration (days)": (dates[-1] - entry_date).days
        })

    # Chuyển đổi thành Series và DataFrame
    equity_series = pd.Series(equity_curve, index=dates, name="Equity")
    trades_df = pd.DataFrame(trades)

    # Tính toán các chỉ số hiệu quả
    final_equity = equity_series.iloc[-1] if n > 0 else cash
    total_return = ((final_equity - cash) / cash) * 100.0

    # Lợi nhuận Buy & Hold (Mua và nắm giữ)
    buy_hold_return = ((closes[-1] - closes[0]) / closes[0]) * 100.0 if n > 0 else 0.0

    # Drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax * 100.0
    max_drawdown = float(drawdown.min())

    # Daily returns & Sharpe ratio
    daily_returns = equity_series.pct_change()
    sharpe = compute_sharpe(daily_returns, window=252)

    # Thống kê giao dịch
    num_trades = len(trades_df)
    if num_trades > 0:
        winning_trades = trades_df[trades_df["Return [%]"] > 0]
        losing_trades = trades_df[trades_df["Return [%]"] <= 0]
        win_rate = (len(winning_trades) / num_trades) * 100.0

        gross_profit = winning_trades["PnL"].sum() if not winning_trades.empty else 0.0
        gross_loss = abs(losing_trades["PnL"].sum()) if not losing_trades.empty else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        best_trade = trades_df["Return [%]"].max()
        worst_trade = trades_df["Return [%]"].min()
        avg_trade = trades_df["Return [%]"].mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        avg_trade = 0.0

    stats = {
        "Duration (bars)": n,
        "Initial Capital": cash,
        "Final Equity": final_equity,
        "Return [%]": total_return,
        "Buy & Hold Return [%]": buy_hold_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_drawdown,
        "# Trades": num_trades,
        "Win Rate [%]": win_rate,
        "Profit Factor": profit_factor,
        "Best Trade [%]": best_trade,
        "Worst Trade [%]": worst_trade,
        "Avg. Trade [%]": avg_trade
    }

    return stats, equity_series, drawdown, trades_df


# ==============================================================================
# 5. TỐI ƯU HÓA THAM SỐ HYPEROPT
# ==============================================================================
def run_hyperopt_optimization(df_train: pd.DataFrame, cash: float, commission: float, max_evals: int = 40):
    """
    Thực hiện Bayesian Optimization bằng Hyperopt trên tập Train:
    1. Tối ưu SMA (ma_short, ma_long)
    2. Tối ưu OBV (obv_window)
    """
    from hyperopt import fmin, tpe, hp, Trials

    # --- Tối ưu SMA ---
    def objective_sma(params):
        ma_s = int(params["ma_short"])
        ma_l = int(params["ma_long"])
        if ma_s >= ma_l:
            return 999999.0
        
        df_temp = generate_strategy_signals(df_train, ma_s, ma_l, obv_window=20)
        stats, _, _, _ = run_backtest_simulation(df_temp, "pos_sma", cash=cash, commission=commission)
        return -stats["Return [%]"]

    space_sma = {
        "ma_short": hp.quniform("ma_short", 20, 120, 5),
        "ma_long": hp.quniform("ma_long", 150, 350, 5)
    }

    trials_sma = Trials()
    best_sma_raw = fmin(
        fn=objective_sma,
        space=space_sma,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials_sma,
        show_progressbar=False
    )
    best_sma = {
        "ma_short": int(best_sma_raw["ma_short"]),
        "ma_long": int(best_sma_raw["ma_long"])
    }

    # --- Tối ưu OBV ---
    def objective_obv(params):
        obv_win = int(params["obv_window"])
        df_temp = generate_strategy_signals(df_train, best_sma["ma_short"], best_sma["ma_long"], obv_window=obv_win)
        stats, _, _, _ = run_backtest_simulation(df_temp, "pos_obv", cash=cash, commission=commission)
        return -stats["Return [%]"]

    space_obv = {
        "obv_window": hp.quniform("obv_window", 5, 80, 5)
    }

    trials_obv = Trials()
    best_obv_raw = fmin(
        fn=objective_obv,
        space=space_obv,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials_obv,
        show_progressbar=False
    )
    best_obv = {
        "obv_window": int(best_obv_raw["obv_window"])
    }

    return best_sma, best_obv


# ==============================================================================
# 6. GIAO DIỆN CHÍNH & SIDEBAR
# ==============================================================================
def main():
    # Header Banner
    st.markdown('<div class="main-header">📈 Kiểm Định Hiệu Quả Chiến Lược SMA + OBV</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Hệ thống Quantitative Backtesting kết hợp chỉ báo xu hướng <b>SMA</b> (Simple Moving Average) '
        'và chỉ báo động lượng dòng tiền <b>OBV</b> (On-Balance Volume) theo điều kiện <b>AND / OR</b>. '
        'Áp dụng quy trình chuẩn khoa học <b>Train / Test Split</b> để tránh overfitting.'
        '</div>',
        unsafe_allow_html=True
    )

    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.header("⚙️ Cấu Hình & Tham Số")

    # 1. Nguồn dữ liệu
    st.sidebar.subheader("1. Dữ liệu giá cổ phiếu")
    data_source_mode = st.sidebar.radio(
        "Nguồn dữ liệu:",
        ["Dữ liệu mẫu ACB (2014 - 2023)", "Tải lên tệp CSV riêng"],
        index=0
    )

    df_full = None
    if data_source_mode == "Dữ liệu mẫu ACB (2014 - 2023)":
        try:
            df_full = load_default_data()
            st.sidebar.success(f" Đã nạp ACB ({len(df_full):,} phiên giao dịch)")
        except Exception as e:
            st.sidebar.error(f"Lỗi nạp dữ liệu mặc định: {e}")
            st.stop()
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Tải lên tệp CSV (Date, Open, High, Low, Close, Volume):",
            type=["csv"]
        )
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                df_full = prepare_stock_data(df_raw)
                st.sidebar.success(f" Đã nạp thành công: {len(df_full):,} phiên")
            except Exception as e:
                st.sidebar.error(f"Lỗi định dạng tệp: {e}")
                st.stop()
        else:
            st.info("Vui lòng tải lên tệp CSV hợp lệ hoặc chọn sử dụng dữ liệu mẫu ACB.")
            st.stop()

    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()

    # 2. Phân chia Train / Test
    st.sidebar.subheader("2. Phân chia Train / Test")
    col_t1, col_t2 = st.sidebar.columns(2)

    # Mặc định theo notebook: Train 2014-01-01 đến 2019-12-31, Test 2020-01-01 đến 2023-12-31
    default_train_start = min_date
    default_train_end = pd.to_datetime("2019-12-31").date() if pd.to_datetime("2019-12-31").date() <= max_date else min_date
    default_test_start = pd.to_datetime("2020-01-01").date() if pd.to_datetime("2020-01-01").date() <= max_date else default_train_end
    default_test_end = max_date

    with col_t1:
        train_start = st.date_input("Train bắt đầu", value=default_train_start, min_value=min_date, max_value=max_date)
        train_end = st.date_input("Train kết thúc", value=default_train_end, min_value=min_date, max_value=max_date)
    with col_t2:
        test_start = st.date_input("Test bắt đầu", value=default_test_start, min_value=min_date, max_value=max_date)
        test_end = st.date_input("Test kết thúc", value=default_test_end, min_value=min_date, max_value=max_date)

    # 3. Vốn & Phí
    st.sidebar.subheader("3. Thiết lập Vốn & Chi phí")
    col_c1, col_c2 = st.sidebar.columns(2)
    with col_c1:
        initial_cash = st.number_input("Vốn ban đầu (VND)", value=1_000_000, step=100_000, min_value=10_000)
    with col_c2:
        commission_pct = st.number_input("Phí GD (%)", value=0.0, step=0.05, min_value=0.0, max_value=2.0)
    commission = commission_pct / 100.0

    # 4. Tham số chiến lược
    st.sidebar.subheader("4. Tham số Chỉ Báo")
    
    # Session state lưu trữ tham số
    if "ma_short" not in st.session_state:
        st.session_state["ma_short"] = 35
    if "ma_long" not in st.session_state:
        st.session_state["ma_long"] = 205
    if "obv_window" not in st.session_state:
        st.session_state["obv_window"] = 20

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("📌 Mẫu Notebook", use_container_width=True):
        st.session_state["ma_short"] = 35
        st.session_state["ma_long"] = 205
        st.session_state["obv_window"] = 20
        st.rerun()

    if col_btn2.button("↺ Mặc định chuẩn", use_container_width=True):
        st.session_state["ma_short"] = 30
        st.session_state["ma_long"] = 200
        st.session_state["obv_window"] = 20
        st.rerun()

    ma_short = st.sidebar.slider("SMA Ngắn hạn (Fast Window)", min_value=5, max_value=120, value=st.session_state["ma_short"], step=5)
    ma_long = st.sidebar.slider("SMA Dài hạn (Slow Window)", min_value=50, max_value=350, value=st.session_state["ma_long"], step=5)
    obv_window = st.sidebar.slider("Chu kỳ MA của OBV (OBV Window)", min_value=5, max_value=80, value=st.session_state["obv_window"], step=5)

    st.session_state["ma_short"] = ma_short
    st.session_state["ma_long"] = ma_long
    st.session_state["obv_window"] = obv_window

    if ma_short >= ma_long:
        st.sidebar.error("⚠️ SMA ngắn hạn phải nhỏ hơn SMA dài hạn!")

    # 5. Tối ưu hóa Hyperopt
    st.sidebar.subheader("5. Tối ưu hóa Hyperopt")
    st.sidebar.caption("Chỉ chạy tối ưu trên tập **Train** để phòng tránh Data Snooping Bias.")
    eval_runs = st.sidebar.slider("Số lượt thử nghiệm (Evals):", min_value=15, max_value=80, value=30, step=5)

    if st.sidebar.button("🚀 Chạy Hyperopt trên Train", use_container_width=True):
        df_train_raw = df_full.loc[(df_full.index.date >= train_start) & (df_full.index.date <= train_end)].copy()
        if len(df_train_raw) < 100:
            st.sidebar.error("Dữ liệu Train quá ngắn để tối ưu hóa.")
        else:
            with st.sidebar.status("Đang thực hiện Bayesian Optimization...", expanded=True) as status:
                st.write("Đang tìm tham số SMA & OBV tốt nhất...")
                best_sma, best_obv = run_hyperopt_optimization(df_train_raw, initial_cash, commission, max_evals=eval_runs)
                st.session_state["ma_short"] = best_sma["ma_short"]
                st.session_state["ma_long"] = best_sma["ma_long"]
                st.session_state["obv_window"] = best_obv["obv_window"]
                status.update(label="✅ Tối ưu hoàn tất!", state="complete", expanded=False)
            st.sidebar.success(f"Kết quả tối ưu: SMA ({best_sma['ma_short']}, {best_sma['ma_long']}), OBV ({best_obv['obv_window']})")
            st.rerun()

    # ------------------ PHÂN CHIA DỮ LIỆU ------------------
    df_train = df_full.loc[(df_full.index.date >= train_start) & (df_full.index.date <= train_end)].copy()
    df_test = df_full.loc[(df_full.index.date >= test_start) & (df_full.index.date <= test_end)].copy()

    if df_train.empty:
        st.error("Khoảng thời gian tập TRAIN không có dữ liệu. Vui lòng chọn lại.")
        st.stop()
    if df_test.empty:
        st.error("Khoảng thời gian tập TEST không có dữ liệu. Vui lòng chọn lại.")
        st.stop()

    # Sinh tín hiệu trên cả Train và Test
    df_train_signals = generate_strategy_signals(df_train, ma_short, ma_long, obv_window)
    df_test_signals = generate_strategy_signals(df_test, ma_short, ma_long, obv_window)

    # Chạy mô phỏng 4 chiến lược trên Train
    strat_keys = [
        ("pos_sma", "SMA Crossover"),
        ("pos_obv", "OBV Crossover"),
        ("pos_and", "SMA + OBV (AND)"),
        ("pos_or", "SMA + OBV (OR)")
    ]

    train_results = {}
    for col_sig, name in strat_keys:
        train_results[name] = run_backtest_simulation(df_train_signals, col_sig, cash=initial_cash, commission=commission)

    test_results = {}
    for col_sig, name in strat_keys:
        test_results[name] = run_backtest_simulation(df_test_signals, col_sig, cash=initial_cash, commission=commission)

    # ------------------ GIAO DIỆN CÁC TABS ------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Bảng So Sánh Hiệu Quả",
        "📈 Biểu Đồ Đường Vốn (Equity Curves)",
        "🔍 Phân Tích Kỹ Thuật & Tín Hiệu",
        "📜 Chi Tiết Giao Dịch (Trade Log)",
        "ℹ️ Lý Thuyết & Quy Trình"
    ])

    # ==============================================================================
    # TAB 1: BẢNG SO SÁNH HIỆU QUẢ CHI TIẾT
    # ==============================================================================
    with tab1:
        st.subheader("⚖️ Bảng Tổng Hợp Kiểm Định Hiệu Quả Chiến Lược")
        st.caption(
            "So sánh 4 chiến lược trên cả 2 tập dữ liệu **Train** (Huấn luyện/Tối ưu) "
            "và **Test** (Kiểm định mẫu ngoài thực tế) với bộ tham số hiện tại: "
            f"**SMA Short = {ma_short}**, **SMA Long = {ma_long}**, **OBV MA = {obv_window}**."
        )

        def make_summary_df(results_dict):
            rows = []
            for name, (stats, _, _, _) in results_dict.items():
                rows.append({
                    "Chiến lược": name,
                    "Lợi nhuận [%]": round(stats["Return [%]"], 2),
                    "Sharpe Ratio": round(stats["Sharpe Ratio"], 2),
                    "Max Drawdown [%]": round(stats["Max Drawdown [%]"], 2),
                    "Số lệnh": stats["# Trades"],
                    "Tỷ lệ thắng [%]": round(stats["Win Rate [%]"], 2),
                    "Profit Factor": round(stats["Profit Factor"], 2),
                    "Lệnh tốt nhất [%]": round(stats["Best Trade [%]"], 2),
                    "Lệnh tệ nhất [%]": round(stats["Worst Trade [%]"], 2),
                    "Buy & Hold [%]": round(stats["Buy & Hold Return [%]"], 2),
                })
            return pd.DataFrame(rows)

        df_sum_train = make_summary_df(train_results)
        df_sum_test = make_summary_df(test_results)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown(f'<span class="badge-train">TẬP TRAIN ({train_start} ➜ {train_end}) — {len(df_train):,} phiên</span>', unsafe_allow_html=True)
            st.dataframe(
                df_sum_train.style.format({
                    "Lợi nhuận [%]": "{:+.2f}%",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown [%]": "{:.2f}%",
                    "Tỷ lệ thắng [%]": "{:.2f}%",
                    "Profit Factor": "{:.2f}",
                    "Lệnh tốt nhất [%]": "{:+.2f}%",
                    "Lệnh tệ nhất [%]": "{:+.2f}%",
                    "Buy & Hold [%]": "{:+.2f}%"
                }).background_gradient(subset=["Lợi nhuận [%]", "Sharpe Ratio"], cmap="YlGn"),
                use_container_width=True
            )

        with col_w2:
            st.markdown(f'<span class="badge-test">TẬP TEST ({test_start} ➜ {test_end}) — {len(df_test):,} phiên</span>', unsafe_allow_html=True)
            st.dataframe(
                df_sum_test.style.format({
                    "Lợi nhuận [%]": "{:+.2f}%",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown [%]": "{:.2f}%",
                    "Tỷ lệ thắng [%]": "{:.2f}%",
                    "Profit Factor": "{:.2f}",
                    "Lệnh tốt nhất [%]": "{:+.2f}%",
                    "Lệnh tệ nhất [%]": "{:+.2f}%",
                    "Buy & Hold [%]": "{:+.2f}%"
                }).background_gradient(subset=["Lợi nhuận [%]", "Sharpe Ratio"], cmap="YlGn"),
                use_container_width=True
            )

        st.markdown("---")

        # Biểu đồ so sánh Return & Sharpe
        col_bar1, col_bar2 = st.columns(2)
        with col_bar1:
            fig_bar_ret = go.Figure()
            fig_bar_ret.add_trace(go.Bar(
                x=df_sum_train["Chiến lược"],
                y=df_sum_train["Lợi nhuận [%]"],
                name="Train Return [%]",
                marker_color="#3B82F6"
            ))
            fig_bar_ret.add_trace(go.Bar(
                x=df_sum_test["Chiến lược"],
                y=df_sum_test["Lợi nhuận [%]"],
                name="Test Return [%]",
                marker_color="#F59E0B"
            ))
            fig_bar_ret.update_layout(
                title="So sánh Tỷ suất sinh lời (Return %)",
                barmode="group",
                template="plotly_white",
                height=350,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_bar_ret, use_container_width=True)

        with col_bar2:
            fig_bar_dd = go.Figure()
            fig_bar_dd.add_trace(go.Bar(
                x=df_sum_train["Chiến lược"],
                y=df_sum_train["Max Drawdown [%]"],
                name="Train Max DD [%]",
                marker_color="#EF4444"
            ))
            fig_bar_dd.add_trace(go.Bar(
                x=df_sum_test["Chiến lược"],
                y=df_sum_test["Max Drawdown [%]"],
                name="Test Max DD [%]",
                marker_color="#DC2626"
            ))
            fig_bar_dd.update_layout(
                title="So sánh Mức sụt giảm tối đa (Max Drawdown %)",
                barmode="group",
                template="plotly_white",
                height=350,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_bar_dd, use_container_width=True)

    # ==============================================================================
    # TAB 2: ĐƯỜNG CONG TĂNG TRƯỞNG VỐN (EQUITY CURVES)
    # ==============================================================================
    with tab2:
        st.subheader("📈 So Sánh Tăng Trưởng Tài Khoản (Equity Curves)")
        dataset_choice = st.radio("Chọn tập dữ liệu hiển thị:", ["Tập TEST (Kiểm định mẫu ngoài)", "Tập TRAIN (Huấn luyện)"], horizontal=True)

        chosen_results = test_results if "TEST" in dataset_choice else train_results
        chosen_df = df_test if "TEST" in dataset_choice else df_train

        # Tạo biểu đồ Plotly Interactive Equity Curve
        fig_equity = go.Figure()

        # Đường Buy & Hold Benchmark
        initial_p = chosen_df["Close"].iloc[0]
        bh_equity = (chosen_df["Close"] / initial_p) * initial_cash
        fig_equity.add_trace(go.Scatter(
            x=chosen_df.index,
            y=bh_equity,
            mode="lines",
            name="Buy & Hold (ACB)",
            line=dict(color="#9CA3AF", dash="dot", width=1.5)
        ))

        colors = {
            "SMA Crossover": "#2563EB",
            "OBV Crossover": "#7C3AED",
            "SMA + OBV (AND)": "#10B981",
            "SMA + OBV (OR)": "#F59E0B"
        }

        for name, (_, eq_series, _, _) in chosen_results.items():
            fig_equity.add_trace(go.Scatter(
                x=eq_series.index,
                y=eq_series.values,
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, "#000"), width=2.2)
            ))

        fig_equity.update_layout(
            title=f"Đường Vốn Tài Khoản ({dataset_choice}) - Vốn ban đầu: {initial_cash:,.0f} VND",
            xaxis_title="Thời gian",
            yaxis_title="Giá trị tài sản (VND)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=480,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Biểu đồ Drawdown
        st.subheader("📉 Biểu Đồ Mức Sụt Giảm (Underwater Drawdown)")
        fig_dd = go.Figure()
        for name, (_, _, dd_series, _) in chosen_results.items():
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index,
                y=dd_series.values,
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, "#000"), width=1.5)
            ))
        fig_dd.update_layout(
            xaxis_title="Thời gian",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            template="plotly_white",
            height=280,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ==============================================================================
    # TAB 3: PHÂN TÍCH KỸ THUẬT & ĐIỂM VÀO/RA LỆNH
    # ==============================================================================
    with tab3:
        st.subheader("🔍 Biểu Đồ Phân Tích Tín Hiệu Giao Dịch Trực Quan")
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            inspect_strat = st.selectbox(
                "Chọn chiến lược phân tích:",
                ["SMA + OBV (AND)", "SMA + OBV (OR)", "SMA Crossover", "OBV Crossover"],
                index=0
            )
        with col_sel2:
            inspect_period = st.selectbox(
                "Giai đoạn phân tích:",
                ["Tập TEST", "Tập TRAIN"],
                index=0
            )

        active_df_signals = df_test_signals if inspect_period == "Tập TEST" else df_train_signals
        col_sig_map = {
            "SMA Crossover": "pos_sma",
            "OBV Crossover": "pos_obv",
            "SMA + OBV (AND)": "pos_and",
            "SMA + OBV (OR)": "pos_or"
        }
        sig_col = col_sig_map[inspect_strat]

        # Tạo Subplots Plotly: Hàng 1 Giá & SMA & Điểm mua/bán, Hàng 2 OBV & OBV_MA, Hàng 3 Khối lượng
        fig_tech = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=[
                f"Giá Cổ Phiếu & Tín hiệu Mua/Bán ({inspect_strat})",
                f"Chỉ báo OBV & OBV MA ({obv_window})",
                "Khối Lượng Giao Dịch (Volume)"
            ]
        )

        # Candlestick
        fig_tech.add_trace(go.Candlestick(
            x=active_df_signals.index,
            open=active_df_signals["Open"],
            high=active_df_signals["High"],
            low=active_df_signals["Low"],
            close=active_df_signals["Close"],
            name="Giá nến",
            increasing_line_color="#10B981",
            decreasing_line_color="#EF4444"
        ), row=1, col=1)

        # SMA lines
        fig_tech.add_trace(go.Scatter(
            x=active_df_signals.index,
            y=active_df_signals["SMA_Short"],
            mode="lines",
            name=f"SMA Ngắn ({ma_short})",
            line=dict(color="#3B82F6", width=1.5)
        ), row=1, col=1)

        fig_tech.add_trace(go.Scatter(
            x=active_df_signals.index,
            y=active_df_signals["SMA_Long"],
            mode="lines",
            name=f"SMA Dài ({ma_long})",
            line=dict(color="#F59E0B", width=1.5)
        ), row=1, col=1)

        # Tín hiệu Buy / Sell
        buys = active_df_signals[active_df_signals[sig_col] == 1.0]
        sells = active_df_signals[active_df_signals[sig_col] == -1.0]

        if not buys.empty:
            fig_tech.add_trace(go.Scatter(
                x=buys.index,
                y=buys["Low"] * 0.98,
                mode="markers",
                name="Tín hiệu MUA",
                marker=dict(symbol="triangle-up", size=11, color="#10B981", line=dict(width=1, color="black"))
            ), row=1, col=1)

        if not sells.empty:
            fig_tech.add_trace(go.Scatter(
                x=sells.index,
                y=sells["High"] * 1.02,
                mode="markers",
                name="Tín hiệu BÁN",
                marker=dict(symbol="triangle-down", size=11, color="#EF4444", line=dict(width=1, color="black"))
            ), row=1, col=1)

        # Subplot 2: OBV & OBV_MA
        fig_tech.add_trace(go.Scatter(
            x=active_df_signals.index,
            y=active_df_signals["OBV"],
            mode="lines",
            name="OBV",
            line=dict(color="#8B5CF6", width=1.3)
        ), row=2, col=1)

        fig_tech.add_trace(go.Scatter(
            x=active_df_signals.index,
            y=active_df_signals["OBV_MA"],
            mode="lines",
            name=f"OBV MA ({obv_window})",
            line=dict(color="#EC4899", width=1.3, dash="dot")
        ), row=2, col=1)

        # Subplot 3: Volume Bar
        fig_tech.add_trace(go.Bar(
            x=active_df_signals.index,
            y=active_df_signals["Volume"],
            name="Volume",
            marker_color="#94A3B8"
        ), row=3, col=1)

        fig_tech.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=700,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_tech, use_container_width=True)

        st.info(
            f"💡 **Thống kê tín hiệu trên {inspect_period}**: "
            f"Số tín hiệu MUA: **{len(buys)}** | Số tín hiệu BÁN: **{len(sells)}**"
        )

    # ==============================================================================
    # TAB 4: SỔ LỆNH & NHẬT KÝ GIAO DỊCH
    # ==============================================================================
    with tab4:
        st.subheader("📜 Nhật Ký Chi Tiết Các Lệnh Đã Thực Hiện")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            selected_strat_trades = st.selectbox(
                "Chiến lược xem lệnh:",
                ["SMA + OBV (AND)", "SMA + OBV (OR)", "SMA Crossover", "OBV Crossover"],
                key="trade_log_strat"
            )
        with col_t2:
            selected_period_trades = st.selectbox(
                "Tập dữ liệu:",
                ["Tập TEST", "Tập TRAIN"],
                key="trade_log_period"
            )

        active_res = test_results if selected_period_trades == "Tập TEST" else train_results
        _, _, _, trades_dataframe = active_res[selected_strat_trades]

        if trades_dataframe.empty:
            st.warning("Chiến lược này không phát sinh giao dịch nào hoàn tất trong giai đoạn đã chọn.")
        else:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Tổng số lệnh", f"{len(trades_dataframe)}")
            win_cnt = len(trades_dataframe[trades_dataframe["Return [%]"] > 0])
            col_m2.metric("Lệnh Thắng / Thua", f"{win_cnt} / {len(trades_dataframe) - win_cnt}")
            col_m3.metric("Tổng PnL thực nhận", f"{trades_dataframe['PnL'].sum():,.0f} VND")
            col_m4.metric("Thời gian giữ lệnh TB", f"{trades_dataframe['Duration (days)'].mean():.1f} ngày")

            st.dataframe(
                trades_dataframe.style.format({
                    "Entry Date": lambda t: pd.to_datetime(t).strftime('%Y-%m-%d'),
                    "Exit Date": lambda t: pd.to_datetime(t).strftime('%Y-%m-%d'),
                    "Entry Price": "{:,.1f}",
                    "Exit Price": "{:,.1f}",
                    "Shares": "{:,.0f}",
                    "PnL": "{:+,.0f}",
                    "Return [%]": "{:+.2f}%",
                    "Duration (days)": "{:.0f}"
                }).background_gradient(subset=["Return [%]"], cmap="RdYlGn", vmin=-15, vmax=25),
                use_container_width=True
            )

            # Tải tệp CSV
            csv_buffer = io.StringIO()
            trades_dataframe.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Tải Nhật Ký Giao Dịch (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"trade_log_{selected_strat_trades.replace(' ', '_')}_{selected_period_trades.replace(' ', '_')}.csv",
                mime="text/csv"
            )

    # ==============================================================================
    # TAB 5: LÝ THUYẾT & NGUYÊN LÝ HOẠT ĐỘNG
    # ==============================================================================
    with tab5:
        st.subheader("ℹ️ Nguyên Lý Hoạt Động & Cơ Sở Lý Thuyết Chiến Lược")
        
        st.markdown("""
        ### 1. Chỉ báo Đường Trung Bình Động (SMA - Simple Moving Average)
        - **SMA Ngắn hạn (`ma_short`)** phản ánh xu hướng giá trong ngắn hạn.
        - **SMA Dài hạn (`ma_long`)** đóng vai trò xác nhận xu hướng chủ đạo dài hạn.
        - **Điểm giao cắt vàng (Golden Cross)**: Khi SMA ngắn cắt lên trên SMA dài $\\rightarrow$ Phát tín hiệu **MUA (BUY = 1)**.
        - **Điểm giao cắt tử thần (Death Cross)**: Khi SMA ngắn cắt xuống dưới SMA dài $\\rightarrow$ Phát tín hiệu **BÁN (SELL = -1)**.

        ---

        ### 2. Chỉ báo Cân Bằng Khối Lượng (OBV - On-Balance Volume)
        - **OBV** đo lường áp lực mua và bán thông qua việc cộng/trừ khối lượng giao dịch dựa trên mức giá đóng cửa:
          $$OBV_t = \\begin{cases} OBV_{t-1} + \\text{Volume}_t & \\text{nếu } Close_t > Close_{t-1} \\\\ OBV_{t-1} - \\text{Volume}_t & \\text{nếu } Close_t < Close_{t-1} \\\\ OBV_{t-1} & \\text{nếu } Close_t = Close_{t-1} \\end{cases}$$
        - Bổ sung đường trung bình động của OBV (**`obv_window`**) để tạo tín hiệu giao cắt:
          - $OBV_t > \\text{OBV\\_MA}_t$ và $OBV_{t-1} \\le \\text{OBV\\_MA}_{t-1} \\rightarrow$ **MUA (BUY = 1)**.
          - $OBV_t < \\text{OBV\\_MA}_t$ và $OBV_{t-1} \\ge \\text{OBV\\_MA}_{t-1} \\rightarrow$ **BÁN (SELL = -1)**.

        ---

        ### 3. Quy Tắc Kết Hợp (AND vs OR)
        1. **Chiến lược AND (Khắt khe / Đồng thuận cao)**:
           - **BUY**: Cả SMA **VÀ** OBV cùng phát tín hiệu MUA trong phiên.
           - **SELL**: Cả SMA **VÀ** OBV cùng phát tín hiệu BÁN trong phiên.
           - *Ưu điểm*: Lọc tối đa tín hiệu nhiễu, hạn chế giao dịch sai.
           - *Nhược điểm*: Rất ít khi hai chỉ báo giao cắt đúng cùng một ngày, có thể bỏ lỡ con sóng tăng.

        2. **Chiến lược OR (Linh hoạt / Bắt sớm xu hướng)**:
           - **BUY**: Hoặc SMA phát tín hiệu MUA, **HOẶC** OBV phát tín hiệu MUA.
           - **SELL**: Hoặc SMA phát tín hiệu BÁN, **HOẶC** OBV phát tín hiệu BÁN.
           - *Ưu điểm*: Bắt nhạy mọi cơ hội khi dòng tiền vào hoặc giá bứt phá.
           - *Nhược điểm*: Có thể phát sinh nhiều tín hiệu giả khi thị trường đi ngang (sideway).

        ---

        ### 4. Phương Pháp Luận Chuẩn Khoa Học: Train / Test Split
        - **Tập Train (2014 - 2019)**: Sử dụng để thử nghiệm, chạy thuật toán tối ưu hóa Bayesian (Hyperopt) tìm bộ tham số tốt nhất.
        - **Tập Test (2020 - 2023)**: Kiểm định ngoài mẫu (Out-of-sample) với đúng bộ tham số từ Train, tuyệt đối không dùng tập Test để tìm tham số.
        - Cách tiếp cận này đảm bảo kiểm định được khả năng ứng dụng thực tế và tính bền vững của chiến lược khi bước vào dữ liệu thị trường tương lai chưa từng biết trước.
        """)


if __name__ == "__main__":
    main()
