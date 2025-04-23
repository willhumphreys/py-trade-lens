import os
import pandas as pd
from datetime import timedelta

def verify_trade_execution(formatted_trades_dir, valid_data):
    """
    Verify that trade filled prices correspond to market data levels.
    Checks if the FilledPrice is crossed anytime up to nine hours after PlaceDateTime.
    Adds a 'PriceCrossed' column to each formatted-trades file.
    
    Args:
        formatted_trades_dir (str): Directory containing formatted-trades CSV files
        valid_data (pd.DataFrame): DataFrame containing market data
    """
    print("\nVerifying trade execution against market data...")
    
    # Ensure dateTime is in datetime format
    if isinstance(valid_data['dateTime'].iloc[0], str):
        valid_data['dateTime'] = pd.to_datetime(valid_data['dateTime'])
    
    # Filter out invalid rows
    valid_data = valid_data[(valid_data['high'] != -1) & (valid_data['low'] != -1)]
    
    # Process each trade file
    files_processed = 0
    trades_verified = 0
    trades_not_verified = 0
    
    for filename in os.listdir(formatted_trades_dir):
        if not filename.endswith('.csv'):
            continue
            
        file_path = os.path.join(formatted_trades_dir, filename)
        trader_id = os.path.splitext(filename)[0]
        
        print(f"Verifying trades for trader {trader_id}...")
        
        # Read trade file
        trades_df = pd.read_csv(file_path)
        
        # Convert PlaceDateTime to datetime if needed
        if isinstance(trades_df['PlaceDateTime'].iloc[0], str):
            trades_df['PlaceDateTime'] = pd.to_datetime(trades_df['PlaceDateTime'])
        
        # Add a new column to track if price was crossed
        trades_df['PriceCrossed'] = False
        
        # Check each trade
        for index, trade in trades_df.iterrows():
            place_time = trade['PlaceDateTime']
            filled_price = trade['FilledPrice']
            
            # Define time window for checking (place time to place time + 9 hours)
            end_time = place_time + timedelta(hours=9)
            
            # Filter valid_data for the time window
            market_data_window = valid_data[
                (valid_data['dateTime'] >= place_time) &
                (valid_data['dateTime'] <= end_time)
            ]
            
            # Check if the filled price is crossed (meaning price moved through this level)
            # For this we check if the low is below filled_price AND high is above filled_price
            # in any single time period, OR if the price moved from above to below (or vice versa)
            # across consecutive time periods
            
            price_crossed = False
            
            # First check: within single candle (low <= filled_price <= high)
            if any((market_data_window['low'] <= filled_price) & 
                   (market_data_window['high'] >= filled_price)):
                price_crossed = True
            
            # Second check: across consecutive candles (if price moves from above to below or vice versa)
            if not price_crossed and len(market_data_window) > 1:
                # Sort by dateTime to ensure consecutive ordering
                sorted_data = market_data_window.sort_values(by='dateTime')
                
                # Check for price crossing between consecutive candles
                for i in range(len(sorted_data) - 1):
                    current_candle = sorted_data.iloc[i]
                    next_candle = sorted_data.iloc[i + 1]
                    
                    # Check if price moved from above filled_price to below (or vice versa)
                    if ((current_candle['low'] > filled_price and next_candle['high'] < filled_price) or
                        (current_candle['high'] < filled_price and next_candle['low'] > filled_price)):
                        price_crossed = True
                        break
            
            # Update the dataframe
            trades_df.at[index, 'PriceCrossed'] = price_crossed
            
            if price_crossed:
                trades_verified += 1
            else:
                trades_not_verified += 1
        
        # Save the updated file
        trades_df.to_csv(file_path, index=False)
        files_processed += 1
    
    # Print summary
    print(f"\nTrade execution verification completed:")
    print(f"  Files processed: {files_processed}")
    print(f"  Trades verified (price crossed): {trades_verified}")
    print(f"  Trades not verified (price not crossed): {trades_not_verified}")
    print(f"  Verification rate: {trades_verified/(trades_verified+trades_not_verified)*100:.2f}%")
    
    return trades_verified, trades_not_verified