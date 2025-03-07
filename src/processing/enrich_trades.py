import os
import pandas as pd
import logging
import sys
import csv

from src.processing.data_reader import read_trader_profit_csv


def extract_trader_info(file_name):
    parts = file_name.split('.')
    trader_id = parts[0]

    return trader_id

def load_csv_data(file_path):
    return read_trader_profit_csv(file_path)

def create_dataframe(data):
    return pd.DataFrame(data, columns=['PlaceDateTime', 'FilledPrice', 'ClosingPrice', 'Profit', 'RunningTotalProfit', 'State'])

def add_profit_marker(df):
    df['ProfitMarker'] = df['Profit'] > 0
    return df

def calculate_profit_factor(df):
    gross_profit = df[df['Profit'] > 0]['Profit'].sum()
    gross_loss = abs(df[df['Profit'] < 0]['Profit'].sum())
    if gross_loss == 0:
        return float('inf')  # If there's no loss, the Profit Factor is infinite
    return gross_profit / gross_loss

def calculate_max_drawdown(df):
    running_max = df['RunningTotalProfit'].cummax()
    drawdown = running_max - df['RunningTotalProfit']
    max_drawdown = drawdown.max()
    return max_drawdown

def calculate_max_profit(df):
    return df['RunningTotalProfit'].max()

def calculate_composite_score(max_profit, max_drawdown, profit_factor):
    return (max_profit - max_drawdown) * profit_factor

def calculate_risk_reward_balance(max_profit, max_drawdown, profit_factor):
    if max_drawdown == 0:
        return float('inf')  # Prevent division by zero
    return (max_profit / max_drawdown) * profit_factor

def process_file(file_name, input_dir):
    trader_id = extract_trader_info(file_name)
    file_path = os.path.join(input_dir, file_name)
    data = load_csv_data(file_path)
    df = create_dataframe(data)
    df = add_profit_marker(df)

    max_drawdown = calculate_max_drawdown(df)
    max_profit = calculate_max_profit(df)
    profit_factor = calculate_profit_factor(df)
    composite_score = calculate_composite_score(max_profit, max_drawdown, profit_factor)
    risk_reward_balance = calculate_risk_reward_balance(max_profit, max_drawdown, profit_factor)

    return trader_id, max_drawdown, max_profit, profit_factor, composite_score, risk_reward_balance

def process_and_calculate_summary(scenario, input_directory, output_directory):
      # Directory with CSV files

    summary_output_dir = os.path.join(output_directory, "summary")

    create_output_directory(summary_output_dir)

    summary_records = []

    for file_name in os.listdir(input_directory):
        if file_name.endswith('.csv'):
            trader_id, max_drawdown, max_profit, profit_factor, composite_score, risk_reward_balance = process_file(file_name, input_directory)
            summary_records.append((trader_id, max_drawdown, max_profit, profit_factor, composite_score, risk_reward_balance))

    summary_df = pd.DataFrame(summary_records, columns=['TraderID', 'MaxDrawdown', 'MaxProfit', 'ProfitFactor', 'CompositeScore', 'RiskRewardBalance'])
    summary_df.sort_values(by='CompositeScore', ascending=False, inplace=True)
    summary_file_path = os.path.join(summary_output_dir, 'summary.csv')
    summary_df.to_csv(summary_file_path, index=False)
    logging.info(f"Summary saved to {summary_file_path}")

    filtered_df = filter_scenarios(summary_df)
    sorted_filtered_df = sort_scenarios(filtered_df)
    filtered_summary_path = os.path.join(summary_output_dir, 'filtered_summary.csv')
    save_filtered_summary(sorted_filtered_df, filtered_summary_path)
    logging.info(f"Filtered summary saved to {filtered_summary_path}")

    best_trades_base_path = os.path.join(output_directory, "trades", scenario + ".csv")
    filtered_trade_path = os.path.join(output_directory, "trades", "filtered-" + scenario + ".csv")

    process_filtered_summary(filtered_summary_path, scenario, best_trades_base_path, filtered_trade_path )



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_filtered_summary(summary_file, scenario, best_trades_base_path, filtered_trade_path):
    """
    Process each entry in the filtered summary to extract trader IDs and scenarios, and find matching trades.
    """
    with open(summary_file, 'r') as summary_csv:
        reader = csv.DictReader(summary_csv)
        for row in reader:
            trader_id = row['TraderID']
            find_matching_trades(trader_id, scenario, best_trades_base_path, filtered_trade_path)


def find_matching_trades(trader_id, scenario, best_trades_base_path, filtered_trade_path):

    found_match = False
    print(f"Looking for best trades file in: {best_trades_base_path}")

    found_match = False


    # Check if the best trades file exists
    if not os.path.exists(best_trades_base_path):
        print(f"Base path not found: {best_trades_base_path}")
        sys.exit(1)


    with open(best_trades_base_path, 'r', newline='') as infile, \
            open(filtered_trade_path, 'a', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

        # Check if the output file is empty; if so, write the header
        if os.path.getsize(filtered_trade_path) == 0:
            writer.writeheader()

        # Iterate over rows and search for a matching trader ID
        for row in reader:
            if row['traderid'] == trader_id:
                writer.writerow(row)
                found_match = True
                print(f"Found matching trade for trader ID {trader_id} in {best_trades_base_path}")
                break  # Exit loop once match is found

    if not found_match:
        print(f"No matching trade found for trader ID {trader_id} in {best_trades_base_path}")


    if not os.path.exists(best_trades_base_path):
        print(f"Base path not found: {best_trades_base_path}")
        sys.exit(1)

    # Sort the output file by CompositeScore after writing
    if os.path.exists("output/summary"):
        try:
            print(f"Sorting {"output/summary"} by CompositeScore...")
            df = pd.read_csv("output/summary/summary.csv")
            if 'CompositeScore' in df.columns:
                df = df.sort_values(by='CompositeScore', ascending=False)  # Sort descending by CompositeScore
                df.to_csv("output/summary/summary.csv", index=False)  # Save back to CSV
            else:
                print(f"Warning: 'CompositeScore' column not found in {"output/summary"}. Skipping sorting.")
        except Exception as e:
            print(f"Error sorting {"output/summary"}: {e}")
        else:
            print(f"Successfully sorted {"output/summary"} by CompositeScore.")

def create_output_directory(path):
    os.makedirs(path, exist_ok=True)
    logging.info(f"Output directory '{path}' created or already exists.")

def load_summary(csv_file):
    """Load the summary CSV file into a DataFrame."""
    return pd.read_csv(csv_file)

def filter_scenarios(df, composite_threshold=0.90, min_profit_factor=1.2, max_drawdown_ratio=0.5):
    """
    Filter scenarios based on composite score, profit factor, and max drawdown relative to max profit.

    Parameters:
    - composite_threshold: Percentile threshold for filtering by composite score.
    - min_profit_factor: Minimum acceptable profit factor.
    - max_drawdown_ratio: Maximum allowed ratio of max drawdown to max profit.
    """
    # Filter by Composite Score (e.g., top 10%)
    composite_score_threshold = df['CompositeScore'].quantile(composite_threshold)
    filtered_df = df[df['CompositeScore'] >= composite_score_threshold]

    # Filter by Profit Factor (e.g., Profit Factor >= 1.2)
    filtered_df = filtered_df[filtered_df['ProfitFactor'] >= min_profit_factor]

    # Filter by Max Drawdown relative to Max Profit (e.g., Max Drawdown <= 50% of Max Profit)
    filtered_df = filtered_df[filtered_df['MaxDrawdown'] <= max_drawdown_ratio * filtered_df['MaxProfit']]

    return filtered_df

def sort_scenarios(df, sort_by='CompositeScore', ascending=False):
    """Sort scenarios by a specific column."""
    return df.sort_values(by=sort_by, ascending=ascending)

def save_filtered_summary(df, output_file):
    """Sort by CompositeScore, round specified columns, and save the DataFrame to a CSV file."""
    # Sort by CompositeScore in descending order
    if 'CompositeScore' in df.columns:
        df = df.sort_values(by='CompositeScore', ascending=False)

    # Ensure specific columns are rounded to 2 decimal places
    if {'ProfitFactor', 'CompositeScore', 'RiskRewardBalance'}.issubset(df.columns):
        df[['ProfitFactor', 'CompositeScore', 'RiskRewardBalance']] = \
            df[['ProfitFactor', 'CompositeScore', 'RiskRewardBalance']].round(2)

    # Save the DataFrame to the file
    df.to_csv(output_file, index=False)
