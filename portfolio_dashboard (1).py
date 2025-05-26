import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
import collections # For deque, a more efficient queue for FIFO

# Import the autorefresh component
from streamlit_autorefresh import st_autorefresh

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Portfolio Tracker")

# --- Automatic Refresh Setup ---
# Rerun the app every 60 seconds (60000 milliseconds)
st_autorefresh(interval=60000, key="data_autorefresh")

# --- Helper Functions (Cached) ---

def get_live_prices(tickers):
    if not tickers:
        return {}
    tickers_list = list(tickers) if not isinstance(tickers, list) else tickers
    
    live_prices_dict = {}
    for ticker_symbol in tickers_list:
        try:
            ticker = yf.Ticker(ticker_symbol)
            price = ticker.info.get('regularMarketPrice') 
            if price is None:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
            
            if price is not None:
                live_prices_dict[ticker_symbol] = price
            else:
                live_prices_dict[ticker_symbol] = 0 # Fallback
        except Exception as e:
            live_prices_dict[ticker_symbol] = 0 # Fallback
    return live_prices_dict


@st.cache_data(ttl=86400) 
def get_historical_prices(tickers, start_date, end_date):
    if not tickers:
        return pd.DataFrame()
    tickers_list = list(tickers) if not isinstance(tickers, list) else tickers
    
    try:
        data = yf.download(tickers_list, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close']
        elif 'Close' in data.columns: 
            return data[['Close']] 
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching historical prices: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600) 
def validate_indian_stock_symbol(symbol):
    if not symbol:
        return None, "Stock symbol cannot be empty."

    try:
        ticker = yf.Ticker(symbol)
        if ticker.info and ticker.info.get('exchange') in ['NSE', 'BSE', 'NSI', 'BOM']:
            return symbol, None
    except:
        pass

    symbol_ns = f"{symbol}.NS"
    try:
        ticker = yf.Ticker(symbol_ns)
        if ticker.info and ticker.info.get('regularMarketPrice') and ticker.info.get('exchange') in ['NSE', 'NSI']:
            return symbol_ns, None
    except:
        pass

    symbol_bo = f"{symbol}.BO"
    try:
        ticker = yf.Ticker(symbol_bo)
        if ticker.info and ticker.info.get('regularMarketPrice') and ticker.info.get('exchange') in ['BSE', 'BOM']:
            return symbol_bo, None
    except:
        pass

    return None, f"Could not find a valid Indian stock symbol for '{symbol}'. Please check the symbol or ensure it's listed on NSE/BSE. Try symbols like 'RELIANCE' or 'TCS.NS'."

def calculate_current_holdings(transactions_df):
    holdings = collections.defaultdict(lambda: {"quantity": 0, "total_cost": 0})
    
    sorted_txns = transactions_df.sort_values(by="Date").reset_index(drop=True)

    for index, txn in sorted_txns.iterrows():
        stock = txn["Stock"]
        action = txn["Action"]
        quantity = txn["Quantity"]
        price = txn["Price"]

        if action == "Buy":
            holdings[stock]["quantity"] += quantity
            holdings[stock]["total_cost"] += quantity * price
        elif action == "Sell":
            if holdings[stock]["quantity"] > 0:
                cost_per_share_before_sell = holdings[stock]["total_cost"] / holdings[stock]["quantity"]
                holdings[stock]["quantity"] -= quantity
                holdings[stock]["total_cost"] -= quantity * cost_per_share_before_sell
            
            if holdings[stock]["quantity"] < 0:
                holdings[stock]["quantity"] = 0
                holdings[stock]["total_cost"] = 0
            elif holdings[stock]["total_cost"] < 0 and holdings[stock]["quantity"] == 0:
                holdings[stock]["total_cost"] = 0
    
    final_holdings_list = []
    for stock, data in holdings.items():
        if data["quantity"] > 0:
            avg_buy_price = data["total_cost"] / data["quantity"] if data["quantity"] > 0 else 0
            final_holdings_list.append({
                "Stock": stock,
                "Quantity": data["quantity"],
                "Avg Buy Price": avg_buy_price
            })
    return final_holdings_list

def calculate_realized_pnl(transactions_df):
    realized_pnl = 0
    cost_basis_pool = collections.defaultdict(collections.deque)
    daily_cumulative_pnl = {} 
    current_cumulative_pnl_snapshot = 0 

    sorted_txns = transactions_df.sort_values(by="Date").reset_index(drop=True)

    for index, txn in sorted_txns.iterrows():
        stock = txn["Stock"]
        action = txn["Action"]
        quantity = txn["Quantity"]
        price = txn["Price"] 
        transaction_date = txn["Date"].date() 

        if action == "Buy":
            cost_basis_pool[stock].append({'qty': quantity, 'price': price})
        
        elif action == "Sell":
            sell_qty_remaining = quantity
            while sell_qty_remaining > 0 and cost_basis_pool[stock]:
                oldest_buy = cost_basis_pool[stock][0] 
                
                shares_from_this_lot = min(sell_qty_remaining, oldest_buy['qty'])
                
                pnl_per_share = price - oldest_buy['price']
                current_pnl_from_this_lot = shares_from_this_lot * pnl_per_share
                
                realized_pnl += current_pnl_from_this_lot
                current_cumulative_pnl_snapshot += current_pnl_from_this_lot 
                
                oldest_buy['qty'] -= shares_from_this_lot 
                sell_qty_remaining -= shares_from_this_lot 

                if oldest_buy['qty'] == 0:
                    cost_basis_pool[stock].popleft()
            
        daily_cumulative_pnl[transaction_date] = current_cumulative_pnl_snapshot
    
    if daily_cumulative_pnl:
        history_df = pd.DataFrame(list(daily_cumulative_pnl.items()), columns=["Date", "Cumulative Realized P&L"])
        history_df["Date"] = pd.to_datetime(history_df["Date"])
        
        history_df = history_df.sort_values(by="Date").drop_duplicates(subset=["Date"], keep="last")

        min_date = history_df["Date"].min()
        max_date = history_df["Date"].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        history_df = history_df.set_index("Date").reindex(full_date_range)
        history_df["Cumulative Realized P&L"] = history_df["Cumulative Realized P&L"].ffill().fillna(0) 
        history_df = history_df.reset_index()
        history_df.rename(columns={'index': 'Date'}, inplace=True)

        return realized_pnl, history_df
    
    return realized_pnl, pd.DataFrame(columns=["Date", "Cumulative Realized P&L"])


# --- Get Watchlist Stock Info (Market Cap in Cr, Day Chg as %, Industry added) ---
@st.cache_data(ttl=900) 
def get_watchlist_stock_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        data = {
            "Stock": symbol,
            "Industry": info.get('industry'), 
            "Market Cap (₹)": info.get('marketCap'), # Raw market cap
            "P/E Ratio": info.get('trailingPE'),
            "Current Price (₹)": info.get('regularMarketPrice'),
            "% Day Chg": info.get('regularMarketChangePercent'), # This is a decimal (e.g., 0.015)
            "EPS": info.get('trailingEps'),
            "Profit Growth (QoQ %)": info.get('earningsQuarterlyGrowth'), # This is a decimal (e.g., 0.05)
            "ROE (%)": info.get('returnOnEquity'), # This is a decimal (e.g., 0.12)
            "Beta": info.get('beta'),
        }

        # Convert percentages from decimal to actual percentage for display
        if data["% Day Chg"] is not None:
            data["% Day Chg"] *= 100
        if data["Profit Growth (QoQ %)"] is not None:
            data["Profit Growth (QoQ %)"] *= 100
        if data["ROE (%)"] is not None:
            data["ROE (%)"] *= 100
        
        # Convert Market Cap to Crores (1 Crore = 10,000,000)
        if data["Market Cap (₹)"] is not None:
            data["Market Cap (₹)"] = data["Market Cap (₹)"] / 10_000_000

        return data
    except Exception as e:
        return {
            "Stock": symbol,
            "Industry": None,
            "Market Cap (₹)": None,
            "P/E Ratio": None,
            "Current Price (₹)": None,
            "% Day Chg": None,
            "EPS": None,
            "Profit Growth (QoQ %)": None,
            "ROE (%)": None,
            "Beta": None,
        }

# --- Styling Function for Green/Red based on value ---
def color_positive_negative(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return 'color: green; font-weight: bold;'
        elif val < 0:
            return 'color: red; font-weight: bold;'
    return '' # Default, no specific color

# --- Initialize Session State ---
if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame(
        columns=["Date", "Stock", "Action", "Quantity", "Price"]
    )
    st.session_state.transactions["Date"] = pd.to_datetime(st.session_state.transactions["Date"])

if "watchlist_stocks" not in st.session_state:
    st.session_state.watchlist_stocks = [] # List of stock symbols

# --- Sidebar: Input New Transaction ---
st.sidebar.header("Add Transaction")
input_date = st.sidebar.date_input("Date", datetime.now())
stock_symbol_input = st.sidebar.text_input("Stock Symbol (e.g., RELIANCE, TCS.NS)").upper()
action = st.sidebar.selectbox("Action", ["Buy", "Sell"])
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1, step=1)
price = st.sidebar.number_input("Price per Share (₹)", min_value=0.01, value=100.00, step=0.01)

if st.sidebar.button("Add Transaction"):
    normalized_stock_symbol, error_message = validate_indian_stock_symbol(stock_symbol_input)
    
    if error_message:
        st.sidebar.error(error_message)
    else:
        # Pre-check for sufficient quantity if selling
        if action == "Sell":
            current_holdings_list = calculate_current_holdings(st.session_state.transactions.copy())
            current_holdings_df = pd.DataFrame(current_holdings_list)
            
            held_qty = current_holdings_df[current_holdings_df["Stock"] == normalized_stock_symbol]["Quantity"].sum()
            
            if quantity > held_qty:
                st.sidebar.warning(f"You only hold {held_qty} shares of {normalized_stock_symbol}. Cannot sell {quantity}.")
                st.stop() 

        new_transaction = pd.DataFrame({
            "Date": [input_date],
            "Stock": [normalized_stock_symbol],
            "Action": [action],
            "Quantity": [quantity],
            "Price": [price]
        })
        new_transaction["Date"] = pd.to_datetime(new_transaction["Date"])

        st.session_state.transactions = pd.concat(
            [st.session_state.transactions, new_transaction], ignore_index=True
        )
        st.sidebar.success(f"Added {action} of {quantity} {normalized_stock_symbol} at ₹{price:,.2f}")
        
        get_historical_prices.clear()
        validate_indian_stock_symbol.clear()
        st.rerun()

# --- Sidebar: Refresh and Clear options ---
st.sidebar.markdown("---")
if st.sidebar.button("Refresh Live Prices Now", type="primary"): 
    st.sidebar.success("Live prices refreshed!") 
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("Clear All Transactions", type="secondary"):
    st.session_state.transactions = pd.DataFrame(
        columns=["Date", "Stock", "Action", "Quantity", "Price"]
    )
    st.session_state.transactions["Date"] = pd.to_datetime(st.session_state.transactions["Date"]) 
    st.session_state.watchlist_stocks = [] 
    st.sidebar.success("All transactions and watchlist cleared.")
    st.cache_data.clear() 
    st.rerun()

# --- Main Dashboard Title (Image removed as requested) ---
st.title("Portfolio Tracker")

# --- Tabs for content organization ---
tab1, tab2, tab3 = st.tabs(["Dashboard", "Transactions", "Watchlist"])

with tab1: # Content for the "Dashboard" tab
    if not st.session_state.transactions.empty:
        
        total_realized_pnl, realized_pnl_history_df = calculate_realized_pnl(st.session_state.transactions.copy())

        st.subheader("Realized Profit/Loss")
        # Realized P&L: Use delta for color-coding (positive green, negative red)
        st.metric(
            "Total Realized P&L",
            f"₹{total_realized_pnl:,.2f}",
            delta=total_realized_pnl, 
            delta_color="normal" 
        )

        holdings_list = calculate_current_holdings(st.session_state.transactions.copy())
        holdings_df = pd.DataFrame(holdings_list)

        if not holdings_df.empty:
            holdings_df.index = holdings_df.index + 1
            
            st.subheader("Live Portfolio Performance (Unrealized)")
            
            unique_held_stocks = holdings_df["Stock"].tolist()
            live_prices = get_live_prices(unique_held_stocks) 
            
            holdings_df["Current Price"] = holdings_df["Stock"].map(live_prices)
            holdings_df["Current Price"] = holdings_df["Current Price"].fillna(holdings_df["Avg Buy Price"])
            holdings_df["Current Price"] = holdings_df["Current Price"].fillna(0) 

            holdings_df["Investment"] = holdings_df["Quantity"] * holdings_df["Avg Buy Price"]
            holdings_df["Current Value"] = holdings_df["Quantity"] * holdings_df["Current Price"]
            holdings_df["Unrealized P&L (₹)"] = holdings_df["Current Value"] - holdings_df["Investment"]
            
            holdings_df["Unrealized P&L (%)"] = holdings_df.apply(
                lambda row: (row["Unrealized P&L (₹)"] / row["Investment"]) * 100 if row["Investment"] != 0 else 0,
                axis=1
            )

            # Apply conditional formatting for Unrealized P&L columns using .map()
            st.dataframe(
                holdings_df.style.format({
                    "Avg Buy Price": "₹{:,.2f}",
                    "Current Price": "₹{:,.2f}",
                    "Investment": "₹{:,.2f}",
                    "Current Value": "₹{:,.2f}",
                    "Unrealized P&L (₹)": "₹{:,.2f}",
                    "Unrealized P&L (%)": "{:,.2f}%"
                }).map(color_positive_negative, subset=["Unrealized P&L (₹)", "Unrealized P&L (%)"]),
                use_container_width=True
            )

            total_current_value = holdings_df["Current Value"].sum()
            total_investment = holdings_df["Investment"].sum()
            total_unrealized_pnl_dollars = holdings_df["Unrealized P&L (₹)"].sum()
            total_unrealized_pnl_percent = (total_unrealized_pnl_dollars / total_investment) * 100 if total_investment != 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Current Value", f"₹{total_current_value:,.2f}")
            with col2:
                st.metric("Total Investment (Held)", f"₹{total_investment:,.2f}")
            with col3:
                # Total Unrealized P&L: Use delta for color-coding (positive green, negative red)
                st.metric(
                    "Total Unrealized P&L",
                    f"₹{total_unrealized_pnl_dollars:,.2f}",
                    delta=f"{total_unrealized_pnl_percent:,.2f}%",
                    delta_color="normal"
                )

            st.subheader("Portfolio Growth Over Time (Unrealized)")
            
            min_transaction_date = st.session_state.transactions["Date"].min()
            max_today = datetime.now()
            
            all_traded_stocks = st.session_state.transactions["Stock"].unique().tolist()
            
            all_historical_prices = get_historical_prices(
                all_traded_stocks, 
                min_transaction_date.strftime("%Y-%m-%d"), 
                (max_today + timedelta(days=1)).strftime("%Y-%m-%d")
            )
            
            if not all_historical_prices.empty:
                all_historical_prices.index = all_historical_prices.index.date 

            portfolio_history_data = []
            
            temp_transactions_for_history = st.session_state.transactions.sort_values(by="Date").reset_index(drop=True)
            current_holdings_state_for_history = collections.defaultdict(lambda: {"quantity": 0, "total_cost": 0})

            all_relevant_dates = sorted(list(set(
                [t.date() for t in st.session_state.transactions["Date"]] + 
                pd.date_range(start=min_transaction_date, end=max_today, freq='D').date.tolist()
            )))

            transaction_idx_for_history = 0
            for current_day in all_relevant_dates:
                while transaction_idx_for_history < len(temp_transactions_for_history) and \
                      temp_transactions_for_history.iloc[transaction_idx_for_history]["Date"].date() <= current_day:
                    
                    txn = temp_transactions_for_history.iloc[transaction_idx_for_history]
                    stock = txn["Stock"]
                    action = txn["Action"]
                    quantity = txn["Quantity"]
                    price = txn["Price"]

                    if action == "Buy":
                        current_holdings_state_for_history[stock]["quantity"] += quantity
                        current_holdings_state_for_history[stock]["total_cost"] += quantity * price
                    elif action == "Sell":
                        if current_holdings_state_for_history[stock]["quantity"] > 0:
                            cost_per_share_before_sell = current_holdings_state_for_history[stock]["total_cost"] / current_holdings_state_for_history[stock]["quantity"]
                            current_holdings_state_for_history[stock]["quantity"] -= quantity
                            current_holdings_state_for_history[stock]["total_cost"] -= quantity * cost_per_share_before_sell
                        
                        if current_holdings_state_for_history[stock]["quantity"] <= 0:
                            current_holdings_state_for_history.pop(stock, None) 
                        elif current_holdings_state_for_history[stock]["total_cost"] < 0: 
                            current_holdings_state_for_history[stock]["total_cost"] = 0
                    
                    transaction_idx_for_history += 1

                total_invested_day = 0
                current_value_day = 0

                for stock, data in current_holdings_state_for_history.items():
                    qty = data["quantity"]
                    total_cost_for_stock = data["total_cost"]

                    if qty > 0:
                        total_invested_day += total_cost_for_stock
                        
                        hist_price = None
                        if current_day in all_historical_prices.index and stock in all_historical_prices.columns:
                            hist_price = all_historical_prices.loc[current_day, stock]
                        
                        if pd.isna(hist_price) or hist_price is None:
                            if stock in all_historical_prices.columns:
                                valid_prices_for_stock = all_historical_prices[stock].dropna()
                                relevant_prices = valid_prices_for_stock[valid_prices_for_stock.index <= current_day]
                                if not relevant_prices.empty:
                                    hist_price = relevant_prices.iloc[-1]
                                else:
                                    hist_price = 0 
                            else:
                                hist_price = 0 
                                
                        current_value_day += qty * hist_price
                
                portfolio_history_data.append({
                    "Date": current_day,
                    "Total Invested": total_invested_day,
                    "Current Value": current_value_day
                })
            
            if portfolio_history_data:
                history_df = pd.DataFrame(portfolio_history_data)
                history_df["Date"] = pd.to_datetime(history_df["Date"])
                
                fig = px.line(
                    history_df, 
                    x="Date", 
                    y=["Total Invested", "Current Value"],
                    title="Investment Value vs. Current Portfolio Value Over Time",
                    labels={"value": "Amount (₹)", "variable": "Metric"},
                    color_discrete_map={
                        "Total Invested": "#636EFA",
                        "Current Value": "#00CC96"
                    }
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to plot portfolio growth. Add more transactions.")
    else: 
        st.info("No transactions added yet. Use the sidebar to input trades and see your portfolio performance.")


with tab2: # Content for the "Transactions" tab
    st.subheader("Your Transactions") 
    if not st.session_state.transactions.empty:
        editable_transactions_df = st.session_state.transactions.sort_values(by="Date", ascending=False).reset_index(drop=True)
        
        edited_df = st.data_editor(
            editable_transactions_df, 
            use_container_width=True,
            num_rows="dynamic", 
            column_config={
                "Date": st.column_config.DateColumn(format="YYYY-MM-DD"), 
                "Quantity": st.column_config.NumberColumn(format="%d"),
                "Price": st.column_config.NumberColumn(format="₹%,.2f")
            },
            key="transactions_data_editor_bottom" 
        )

        if not edited_df.equals(editable_transactions_df):
            try:
                required_cols = ["Date", "Stock", "Action", "Quantity", "Price"]
                if not all(col in edited_df.columns for col in required_cols):
                    st.error("Missing expected columns after editing. Please ensure all columns are present.")
                    st.stop()

                edited_df["Date"] = pd.to_datetime(edited_df["Date"])
                edited_df["Quantity"] = pd.to_numeric(edited_df["Quantity"], errors='coerce')
                edited_df["Price"] = pd.to_numeric(edited_df["Price"], errors='coerce')

                if edited_df["Quantity"].isnull().any() or edited_df["Price"].isnull().any():
                    st.error("Quantity and Price must be valid numbers.")
                    st.stop()
                
                for index, row in edited_df.iterrows():
                    symbol_check, symbol_error = validate_indian_stock_symbol(row["Stock"])
                    if symbol_error:
                        st.error(f"Error in row {index+1}: {symbol_error}. Please correct the stock symbol.")
                        st.stop() 
                    edited_df.loc[index, "Stock"] = symbol_check 

                    if row["Quantity"] < 1:
                        st.error(f"Error in row {index+1}: Quantity must be at least 1.")
                        st.stop()
                    if row["Price"] < 0.01:
                        st.error(f"Error in row {index+1}: Price must be at least ₹0.01.")
                        st.stop()

                st.session_state.transactions = edited_df
                st.success("Transactions updated successfully!")
                st.cache_data.clear() 
                st.rerun() 
            except Exception as e:
                st.error(f"Error updating transactions: {e}. Please ensure 'Date' is in YYYY-MM-DD format, Quantity is a number, and Price is a number.")
    else:
        st.info("No transactions added yet.")

with tab3: # Content for the "Watchlist" tab
    st.subheader("Your Stock Watchlist")

    # --- Add Stock to Watchlist ---
    watchlist_add_col, watchlist_status_col = st.columns([0.7, 0.3])
    with watchlist_add_col:
        new_watchlist_symbol_input = st.text_input("Add Stock to Watchlist (e.g., INFY, SBIN.NS)", key="new_watchlist_input").upper()
    with watchlist_status_col:
        st.markdown(" ") # Spacer
        if st.button("Add to Watchlist", key="add_to_watchlist_button"):
            if new_watchlist_symbol_input:
                normalized_symbol, error = validate_indian_stock_symbol(new_watchlist_symbol_input)
                if error:
                    st.error(error)
                elif normalized_symbol not in st.session_state.watchlist_stocks:
                    st.session_state.watchlist_stocks.append(normalized_symbol)
                    st.success(f"Added {normalized_symbol} to watchlist.")
                    get_watchlist_stock_info.clear() 
                    st.rerun()
                else:
                    st.info(f"{normalized_symbol} is already in your watchlist.")
            else:
                st.warning("Please enter a stock symbol.")

    st.markdown("---")

    # --- Display Watchlist (Using st.dataframe for guaranteed styling) ---
    if st.session_state.watchlist_stocks:
        watchlist_data = []
        with st.spinner("Fetching watchlist data..."):
            for symbol in st.session_state.watchlist_stocks:
                info = get_watchlist_stock_info(symbol)
                watchlist_data.append(info)
        
        watchlist_df = pd.DataFrame(watchlist_data)
        
        # Set 'Stock' as index for display purposes
        watchlist_df = watchlist_df.set_index("Stock")

        # Reorder columns for better readability
        display_columns = [
            "Industry", "Current Price (₹)", "% Day Chg", "Market Cap (₹)", "P/E Ratio", "EPS", 
            "Profit Growth (QoQ %)", "ROE (%)", "Beta"
        ]
        watchlist_df = watchlist_df[display_columns]

        st.subheader("Current Watchlist Stocks")
        
        # Apply styling for formatting and colors using .style and .map()
        styled_watchlist_df = watchlist_df.style.format({
            "Current Price (₹)": "₹{:,.2f}",
            "Market Cap (₹)": "₹{:,.2f} Cr", # Formatted for Crores
            "P/E Ratio": "{:,.2f}",
            "% Day Chg": "{:,.2f}%",  # Correctly show percentage
            "EPS": "₹{:,.2f}",
            "Profit Growth (QoQ %)": "{:,.2f}%",
            "ROE (%)": "{:,.2f}%",
            "Beta": "{:,.2f}"
        }, na_rep="N/A").map(color_positive_negative, subset=["% Day Chg", "Profit Growth (QoQ %)"]) # Apply color here

        st.dataframe(
            styled_watchlist_df, # Pass the styled DataFrame
            use_container_width=True
        )

        st.markdown("---")

        # --- Remove Stock from Watchlist (Separate Section for clarity and reliability) ---
        st.subheader("Remove Stocks from Watchlist")
        # Use a form to group the multiselect and button for better interaction
        with st.form("remove_watchlist_form"):
            symbols_to_remove = st.multiselect(
                "Select stocks to remove from watchlist",
                options=st.session_state.watchlist_stocks,
                key="remove_watchlist_multiselect"
            )
            submitted = st.form_submit_button("Remove Selected Stocks")

            if submitted:
                if symbols_to_remove:
                    st.session_state.watchlist_stocks = [
                        s for s in st.session_state.watchlist_stocks if s not in symbols_to_remove
                    ]
                    st.success(f"Removed {len(symbols_to_remove)} stock(s) from watchlist.")
                    get_watchlist_stock_info.clear() # Clear cache for watchlist data
                    st.rerun()
                else:
                    st.info("No stocks selected to remove.")
    else:
        st.info("Your watchlist is empty. Add stocks using the input above.")
