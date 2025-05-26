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

# --- PASTE YOUR GENERATED BASE64 STRING HERE ---
# After running generate_base64.py, copy the very long string it prints
# and paste it between the double quotes below.
# Example (this is a placeholder, replace with your actual string):
image_base64_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" 
# REMEMBER TO REPLACE THE ABOVE STRING WITH YOUR ACTUAL GENERATED BASE64 STRING!
# It will be much, much longer than this example.
# --------------------------------------------------

# --- Helper Functions (Cached) ---

# Removed @st.cache_data from get_live_prices so it always fetches fresh data on app rerun
def get_live_prices(tickers):
    if not tickers:
        return {}
    tickers_list = list(tickers) if not isinstance(tickers, list) else tickers
    
    live_prices_dict = {}
    for ticker_symbol in tickers_list:
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Use 'regularMarketPrice' for more up-to-the-minute data if available
            price = ticker.info.get('regularMarketPrice') 
            if price is None:
                # Fallback to current day's close if regularMarketPrice is not immediately available
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


@st.cache_data(ttl=86400) # Keep historical data cached for 24 hours (changes less frequently)
def get_historical_prices(tickers, start_date, end_date):
    if not tickers:
        return pd.DataFrame()
    tickers_list = list(tickers) if not isinstance(tickers, list) else tickers
    
    try:
        # yf.download handles multiple tickers efficiently
        data = yf.download(tickers_list, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()
        
        # If multiple tickers, 'Close' is a multi-level column, select all Close prices
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close']
        elif 'Close' in data.columns: # Single ticker
            return data[['Close']] # Return as DataFrame for consistency
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching historical prices: {e}")
        return pd.DataFrame()

# Function to validate and normalize Indian stock symbols
@st.cache_data(ttl=3600) # Cache symbol validation for 1 hour
def validate_indian_stock_symbol(symbol):
    if not symbol:
        return None, "Stock symbol cannot be empty."

    # Try symbol as-is (e.g., if user already entered .NS or .BO)
    try:
        ticker = yf.Ticker(symbol)
        # Check if info is available and if it's an Indian exchange
        if ticker.info and ticker.info.get('exchange') in ['NSE', 'BSE', 'NSI', 'BOM']:
            return symbol, None
    except:
        pass # Continue to try other options

    # Try appending .NS (NSE)
    symbol_ns = f"{symbol}.NS"
    try:
        ticker = yf.Ticker(symbol_ns)
        # Check for regularMarketPrice to ensure it's an active trading symbol
        if ticker.info and ticker.info.get('regularMarketPrice') and ticker.info.get('exchange') in ['NSE', 'NSI']:
            return symbol_ns, None
    except:
        pass

    # Try appending .BO (BSE)
    symbol_bo = f"{symbol}.BO"
    try:
        ticker = yf.Ticker(symbol_bo)
        # Check for regularMarketPrice to ensure it's an active trading symbol
        if ticker.info and ticker.info.get('regularMarketPrice') and ticker.info.get('exchange') in ['BSE', 'BOM']:
            return symbol_bo, None
    except:
        pass

    return None, f"Could not find a valid Indian stock symbol for '{symbol}'. Please check the symbol or ensure it's listed on NSE/BSE. Try symbols like 'RELIANCE' or 'TCS.NS'."

# Function to calculate current holdings (used for pre-sell checks and unrealized P&L)
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
            
            # Ensure holdings don't go negative or cost becomes negative for remaining shares
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


# --- Realized P&L Calculation Logic (FIFO) ---
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
            
            # Optional: warn if selling more than owned according to FIFO history
            # if sell_qty_remaining > 0:
            #     st.warning(f"Warning: Sold {quantity} shares of {stock} on {txn['Date'].date()}, but only {quantity - sell_qty_remaining} shares had a recorded purchase history. Realized P&L is calculated only for available shares under FIFO.")
        
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


# --- New Function: Get Watchlist Stock Info ---
@st.cache_data(ttl=900) # Cache for 15 minutes, as these metrics don't change as fast as live price
def get_watchlist_stock_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        data = {
            "Stock": symbol,
            "Market Cap (₹)": info.get('marketCap'),
            "P/E Ratio": info.get('trailingPE'),
            "Current Price (₹)": info.get('regularMarketPrice'),
            "% Day Chg": info.get('regularMarketChangePercent'),
            "EPS": info.get('trailingEps'),
            "Sales Growth (QoQ %)": info.get('revenueQuarterlyGrowth'),
            "Profit Growth (QoQ %)": info.get('earningsQuarterlyGrowth'),
            # ROCE is not directly available in yfinance info. Using ROE as a proxy or leaving N/A.
            "ROE (%)": info.get('returnOnEquity'),
            "Beta": info.get('beta'),
        }

        # Convert percentages from decimal to actual percentage
        if data["% Day Chg"] is not None:
            data["% Day Chg"] *= 100
        if data["Sales Growth (QoQ %)"] is not None:
            data["Sales Growth (QoQ %)"] *= 100
        if data["Profit Growth (QoQ %)"] is not None:
            data["Profit Growth (QoQ %)"] *= 100
        if data["ROE (%)"] is not None:
            data["ROE (%)"] *= 100

        return data
    except Exception as e:
        return {
            "Stock": symbol,
            "Market Cap (₹)": None,
            "P/E Ratio": None,
            "Current Price (₹)": None,
            "% Day Chg": None,
            "EPS": None,
            "Sales Growth (QoQ %)": None,
            "Profit Growth (QoQ %)": None,
            "ROE (%)": None,
            "Beta": None,
        }

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
                st.stop() # Stop execution to prevent adding invalid transaction

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
        
        # Clear specific caches that depend on transactions (historical data, symbol validation)
        get_historical_prices.clear()
        validate_indian_stock_symbol.clear()
        st.rerun()

# --- Sidebar: Refresh and Clear options ---
st.sidebar.markdown("---")
# This button still triggers a full rerun, which will fetch new live prices
if st.sidebar.button("Refresh Live Prices Now", type="primary"): 
    st.sidebar.success("Live prices refreshed!") 
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("Clear All Transactions", type="secondary"):
    st.session_state.transactions = pd.DataFrame(
        columns=["Date", "Stock", "Action", "Quantity", "Price"]
    )
    st.session_state.transactions["Date"] = pd.to_datetime(st.session_state.transactions["Date"]) # Ensure type consistency
    st.session_state.watchlist_stocks = [] # Also clear watchlist
    st.sidebar.success("All transactions and watchlist cleared.")
    # Clear all caches, including historical data and symbol validation
    st.cache_data.clear() # Clear all caches
    st.rerun()

# --- Main Dashboard Title with Image (Using Base64) ---
col_title, col_image = st.columns([0.8, 0.2]) # Adjust ratio as needed

with col_title:
    st.title("Portfolio Tracker")

with col_image:
    # Use the Base64 data URL here
    st.image(image_base64_data_url, width=150) # Adjust width as needed

# --- Tabs for content organization ---
tab1, tab2, tab3 = st.tabs(["Dashboard", "Transactions", "Watchlist"])

with tab1: # Content for the "Dashboard" tab
    if not st.session_state.transactions.empty:
        
        # Calculate Realized P&L first using the dedicated function
        total_realized_pnl, realized_pnl_history_df = calculate_realized_pnl(st.session_state.transactions.copy())

        st.subheader("Realized Profit/Loss")
        st.metric(
            "Total Realized P&L",
            f"₹{total_realized_pnl:,.2f}",
            delta_color="normal" 
        )

        # Get current holdings using the new helper function
        holdings_list = calculate_current_holdings(st.session_state.transactions.copy())
        holdings_df = pd.DataFrame(holdings_list)

        if not holdings_df.empty:
            # Make index start from 1
            holdings_df.index = holdings_df.index + 1
            
            st.subheader("Live Portfolio Performance (Unrealized)")
            
            unique_held_stocks = holdings_df["Stock"].tolist()
            live_prices = get_live_prices(unique_held_stocks) 
            
            holdings_df["Current Price"] = holdings_df["Stock"].map(live_prices)

            # Fallback for current price if live price not fetched (e.g., market closed, error)
            holdings_df["Current Price"] = holdings_df["Current Price"].fillna(holdings_df["Avg Buy Price"])
            holdings_df["Current Price"] = holdings_df["Current Price"].fillna(0) 

            holdings_df["Investment"] = holdings_df["Quantity"] * holdings_df["Avg Buy Price"]
            holdings_df["Current Value"] = holdings_df["Quantity"] * holdings_df["Current Price"]
            holdings_df["Unrealized P&L (₹)"] = holdings_df["Current Value"] - holdings_df["Investment"]
            
            holdings_df["Unrealized P&L (%)"] = holdings_df.apply(
                lambda row: (row["Unrealized P&L (₹)"] / row["Investment"]) * 100 if row["Investment"] != 0 else 0,
                axis=1
            )

            st.dataframe(
                holdings_df.style.format({
                    "Avg Buy Price": "₹{:,.2f}",
                    "Current Price": "₹{:,.2f}",
                    "Investment": "₹{:,.2f}",
                    "Current Value": "₹{:,.2f}",
                    "Unrealized P&L (₹)": "₹{:,.2f}",
                    "Unrealized P&L (%)": "{:,.2f}%"
                }),
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
                st.metric(
                    "Total Unrealized P&L",
                    f"₹{total_unrealized_pnl_dollars:,.2f}",
                    delta=f"{total_unrealized_pnl_percent:,.2f}%"
                )

            st.subheader("Portfolio Growth Over Time (Unrealized)")
            
            min_transaction_date = st.session_state.transactions["Date"].min()
            max_today = datetime.now()
            
            all_traded_stocks = st.session_state.transactions["Stock"].unique().tolist()
            
            # Fetch historical prices for all stocks involved in transactions
            all_historical_prices = get_historical_prices(
                all_traded_stocks, 
                min_transaction_date.strftime("%Y-%m-%d"), 
                (max_today + timedelta(days=1)).strftime("%Y-%m-%d")
            )
            
            if not all_historical_prices.empty:
                all_historical_prices.index = all_historical_prices.index.date 

            portfolio_history_data = []
            
            # Use a copy of transactions to avoid modifying the original during iteration
            temp_transactions_for_history = st.session_state.transactions.sort_values(by="Date").reset_index(drop=True)
            current_holdings_state_for_history = collections.defaultdict(lambda: {"quantity": 0, "total_cost": 0})

            # Generate a full date range from the first transaction to today
            all_relevant_dates = sorted(list(set(
                [t.date() for t in st.session_state.transactions["Date"]] + 
                pd.date_range(start=min_transaction_date, end=max_today, freq='D').date.tolist()
            )))

            transaction_idx_for_history = 0
            for current_day in all_relevant_dates:
                # Apply transactions that occurred on or before the current day
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
                            # Use FIFO logic for cost basis reduction
                            cost_per_share_before_sell = current_holdings_state_for_history[stock]["total_cost"] / current_holdings_state_for_history[stock]["quantity"]
                            current_holdings_state_for_history[stock]["quantity"] -= quantity
                            current_holdings_state_for_history[stock]["total_cost"] -= quantity * cost_per_share_before_sell
                        
                        # Clean up if quantity drops to zero or below
                        if current_holdings_state_for_history[stock]["quantity"] <= 0:
                            current_holdings_state_for_history.pop(stock, None) 
                        elif current_holdings_state_for_history[stock]["total_cost"] < 0: # Should not happen with correct FIFO, but as a safeguard
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
                            # Try to find the most recent available price if current_day is a holiday/weekend
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
    else: # This else belongs to "if not st.session_state.transactions.empty"
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
            key="transactions_data_editor_bottom" # Unique key for the widget
        )

        if not edited_df.equals(editable_transactions_df):
            try:
                # Ensure all required columns are present and typed correctly after editing
                required_cols = ["Date", "Stock", "Action", "Quantity", "Price"]
                if not all(col in edited_df.columns for col in required_cols):
                    st.error("Missing expected columns after editing. Please ensure all columns are present.")
                    st.stop()

                # Basic validation for data types
                edited_df["Date"] = pd.to_datetime(edited_df["Date"])
                edited_df["Quantity"] = pd.to_numeric(edited_df["Quantity"], errors='coerce')
                edited_df["Price"] = pd.to_numeric(edited_df["Price"], errors='coerce')

                # Validate data content
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
                st.cache_data.clear() # Clear all caches if transactions changed
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
                    get_watchlist_stock_info.clear() # Clear cache for watchlist data to get new stock
                    st.rerun()
                else:
                    st.info(f"{normalized_symbol} is already in your watchlist.")
            else:
                st.warning("Please enter a stock symbol.")

    st.markdown("---")

    # --- Display Watchlist ---
    if st.session_state.watchlist_stocks:
        watchlist_data = []
        with st.spinner("Fetching watchlist data..."):
            for symbol in st.session_state.watchlist_stocks:
                info = get_watchlist_stock_info(symbol)
                watchlist_data.append(info)
        
        watchlist_df = pd.DataFrame(watchlist_data)
        watchlist_df = watchlist_df.set_index("Stock")

        # Reorder columns for better readability
        display_columns = [
            "Current Price (₹)", "% Day Chg", "Market Cap (₹)", "P/E Ratio", "EPS", 
            "Sales Growth (QoQ %)", "Profit Growth (QoQ %)", "ROE (%)", "Beta"
        ]
        watchlist_df = watchlist_df[display_columns]

        st.dataframe(
            watchlist_df.style.format({
                "Current Price (₹)": "₹{:,.2f}",
                "Market Cap (₹)": "₹{:,.0f}", # No decimals for Market Cap
                "P/E Ratio": "{:,.2f}",
                "% Day Chg": "{:,.2f}%",
                "EPS": "₹{:,.2f}",
                "Sales Growth (QoQ %)": "{:,.2f}%",
                "Profit Growth (QoQ %)": "{:,.2f}%",
                "ROE (%)": "{:,.2f}%",
                "Beta": "{:,.2f}"
            }, na_rep="N/A"), # Handle missing data gracefully
            use_container_width=True
        )

        st.markdown("---")

        # --- Remove Stock from Watchlist ---
        st.subheader("Remove Stocks from Watchlist")
        symbols_to_remove = st.multiselect(
            "Select stocks to remove",
            options=st.session_state.watchlist_stocks,
            key="remove_watchlist_multiselect"
        )
        if st.button("Remove Selected", key="remove_watchlist_button"):
            if symbols_to_remove:
                st.session_state.watchlist_stocks = [
                    s for s in st.session_state.watchlist_stocks if s not in symbols_to_remove
                ]
                st.success(f"Removed {len(symbols_to_remove)} stock(s) from watchlist.")
                get_watchlist_stock_info.clear() # Clear cache for watchlist data as it changed
                st.rerun()
            else:
                st.info("No stocks selected to remove.")
    else:
        st.info("Your watchlist is empty. Add stocks using the input above.")
