import os
import pandas as pd
import numpy as np # Import numpy
import logging
import sys
import csv

# Assuming read_trader_profit_csv exists and works as expected
# from src.processing.data_reader import read_trader_profit_csv
# Placeholder if the import doesn't work in this context
def read_trader_profit_csv(file_path):
    # In a real scenario, this would read your specific CSV format
    # For demonstration, assume it reads into columns needed
    try:
        # Adjust columns based on your actual CSV structure
        # Ensure 'Profit' and 'RunningTotalProfit' are present or calculated
        return pd.read_csv(file_path) # Placeholder read
    except Exception as e:
        logging.error(f"Error reading CSV {file_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)


def extract_trader_info(file_name):
    """Extracts trader ID from filename."""
    # Consider more robust parsing if filename format varies
    parts = os.path.splitext(file_name) # Use os.path.splitext
    trader_id = parts[0]
    return trader_id

def load_csv_data(file_path):
    """Loads trade data using the defined reader."""
    logging.debug(f"Loading data from: {file_path}")
    return read_trader_profit_csv(file_path)

def create_dataframe(data):
    """Creates DataFrame, ensuring necessary columns exist."""
    # Define expected columns; adjust based on read_trader_profit_csv output
    expected_columns = ['PlaceDateTime', 'FilledPrice', 'ClosingPrice', 'Profit', 'RunningTotalProfit', 'State']
    # If data is already a DataFrame from load_csv_data, just validate
    if isinstance(data, pd.DataFrame):
        # Check if essential columns are present
        if 'Profit' not in data.columns or 'RunningTotalProfit' not in data.columns:
            logging.error("Input data missing essential 'Profit' or 'RunningTotalProfit' columns.")
            # Attempt calculation if possible, otherwise return empty/raise error
            # Example: if 'ClosingPrice' in data.columns and 'FilledPrice' in data.columns:
            #     data['Profit'] = data['ClosingPrice'] - data['FilledPrice'] # Adjust based on your logic
            #     data['RunningTotalProfit'] = data['Profit'].cumsum()
            # else:
            #     return pd.DataFrame() # Cannot proceed
            return pd.DataFrame() # Return empty if essential cols missing
        return data
    else:
        # If data is not a DataFrame (e.g., list of lists), create it
        logging.warning("Input data is not a DataFrame. Attempting creation.")
        try:
            df = pd.DataFrame(data, columns=expected_columns)
            # Validate essential columns again after creation
            if 'Profit' not in df.columns or 'RunningTotalProfit' not in df.columns:
                logging.error("Created DataFrame missing essential 'Profit' or 'RunningTotalProfit'.")
                return pd.DataFrame()
            return df
        except Exception as e:
            logging.error(f"Error creating DataFrame: {e}")
            return pd.DataFrame()


# --- Metric Calculation Functions ---

def calculate_profit_factor(df):
    """Calculates profit factor from Profit column."""
    if 'Profit' not in df.columns or df['Profit'].isnull().all():
        return 0.0 # Or np.nan, depending on desired handling
    gross_profit = df.loc[df['Profit'] > 0, 'Profit'].sum()
    gross_loss = abs(df.loc[df['Profit'] < 0, 'Profit'].sum())
    if gross_loss == 0:
        # Handle case with no losses: return large number or specific indicator
        return np.inf # Standard definition
    # Ensure gross_profit is non-negative (can happen with NaNs/bad data)
    gross_profit = max(gross_profit, 0)
    return gross_profit / gross_loss

def calculate_max_drawdown(df):
    """Calculates max drawdown from RunningTotalProfit column."""
    if 'RunningTotalProfit' not in df.columns or df['RunningTotalProfit'].isnull().all():
        return 0.0 # Or np.nan
    running_max = df['RunningTotalProfit'].cummax()
    drawdown = running_max - df['RunningTotalProfit']
    # Ensure drawdown calculation didn't produce NaNs that break max()
    max_drawdown = drawdown.dropna().max()
    return max_drawdown if pd.notna(max_drawdown) else 0.0

def calculate_total_profit(df):
    """Calculates total profit."""
    if 'Profit' not in df.columns or df['Profit'].isnull().all():
        return 0.0
    return df['Profit'].sum()

def calculate_trade_count(df):
    """Calculates trade count."""
    return len(df)

def calculate_recovery_factor(total_profit, max_drawdown):
    """Calculates recovery factor."""
    if max_drawdown == 0:
        # Handle case with no drawdown
        return np.inf if total_profit > 0 else 0.0
    # Avoid division by zero if max_drawdown is somehow negative (bad data)
    if max_drawdown < 0:
        return 0.0
    return total_profit / max_drawdown

def calculate_sortino_ratio(df, target_return_pa=0.0):
    """
    Calculates the Sortino ratio.
    Assumes df['Profit'] represents profit per trade.
    Requires estimating periods per year if target is annualized.
    For simplicity here, we'll use target_return=0 per trade.
    """
    if 'Profit' not in df.columns or len(df) < 2:
        return 0.0 # Not enough data

    target_return_per_trade = 0 # Assuming target is 0 for simplicity

    # Calculate downside returns
    downside_returns = df.loc[df['Profit'] < target_return_per_trade, 'Profit']

    if downside_returns.empty:
        # No returns below target, Sortino is infinite if avg profit > target, else 0
        avg_profit = df['Profit'].mean()
        return np.inf if avg_profit > target_return_per_trade else 0.0

    # Calculate Downside Deviation (std dev of returns below target)
    expected_value = df['Profit'].mean()
    downside_diff_sq = (downside_returns - target_return_per_trade).pow(2)
    downside_deviation = np.sqrt(downside_diff_sq.sum() / len(df)) # Use N in denominator

    if downside_deviation == 0:
        # Possible if all losses are exactly the target (unlikely with 0 target)
        avg_profit = df['Profit'].mean()
        return np.inf if avg_profit > target_return_per_trade else 0.0

    # Calculate Sortino Ratio
    sortino = (expected_value - target_return_per_trade) / downside_deviation
    return sortino if pd.notna(sortino) else 0.0


def calculate_max_drawdown_duration(df):
    """
    Calculates the maximum duration (in number of trades) of any drawdown period.
    """
    if 'RunningTotalProfit' not in df.columns or len(df) < 2:
        return 0

    running_max = df['RunningTotalProfit'].cummax()
    drawdown = running_max - df['RunningTotalProfit']

    in_drawdown = drawdown > 0
    # Find start of drawdown periods (where it transitions from not in DD to in DD)
    drawdown_starts = (~in_drawdown.shift(1, fill_value=False)) & in_drawdown
    drawdown_groups = drawdown_starts.cumsum()

    # Filter only periods in drawdown and group by the drawdown period identifier
    drawdown_periods = drawdown_groups[in_drawdown]

    if drawdown_periods.empty:
        return 0 # No drawdown periods found

    # Calculate lengths of each drawdown period
    drawdown_lengths = drawdown_periods.groupby(drawdown_periods).count()

    return drawdown_lengths.max() if not drawdown_lengths.empty else 0


# --- Isolated Score Calculation ---

def calculate_single_isolated_score(metrics):
    """
    Calculates the isolated composite score for a single trader
    based on their calculated metrics.

    Args:
        metrics (dict): A dictionary containing the required scalar metrics:
                        'profit_factor', 'recovery_factor', 'sortino_ratio',
                        'tradecount', 'max_drawdown_duration'.

    Returns:
        float: The calculated isolated composite score. Returns NaN if required metrics missing.
    """
    # --- Parameters for Scoring (same as in score_calculator.py) ---
    required_metrics_keys = [
        'profit_factor', 'recovery_factor', 'sortino_ratio',
        'tradecount', 'max_drawdown_duration'
    ]
    caps = {'profit_factor': 10.0, 'recovery_factor': 50.0, 'sortino_ratio': 5.0}
    floors = {'sortino_ratio': 0.0}
    weights = {
        'capped_pf': 0.30, 'capped_rf': 0.30, 'capped_sortino': 0.20,
        'log_tc': 0.10, 'norm_mdd_dur': 0.10
    }

    # --- Validate Inputs ---
    if not all(key in metrics for key in required_metrics_keys):
        logging.warning(f"Missing required metrics for score calculation. Got: {metrics.keys()}")
        return np.nan # Cannot calculate score

    # --- Handle Inf/NaN in input metrics ---
    pf = metrics['profit_factor']
    rf = metrics['recovery_factor']
    sr = metrics['sortino_ratio']
    tc = metrics['tradecount']
    mdd_dur = metrics['max_drawdown_duration']

    # Replace inf with NaN, then fill NaN with 0 (adjust fill value if needed)
    pf = np.nan if np.isinf(pf) else pf
    rf = np.nan if np.isinf(rf) else rf
    sr = np.nan if np.isinf(sr) else sr

    pf = 0.0 if pd.isna(pf) else pf
    rf = 0.0 if pd.isna(rf) else rf
    sr = 0.0 if pd.isna(sr) else sr
    tc = 0 if pd.isna(tc) else max(0, tc) # Ensure non-negative count
    mdd_dur = 0 if pd.isna(mdd_dur) else max(0, mdd_dur) # Ensure non-negative duration

    # --- Apply Caps/Floors/Transformations ---
    capped_pf = np.clip(pf, floors.get('profit_factor', None), caps.get('profit_factor'))
    capped_rf = np.clip(rf, floors.get('recovery_factor', None), caps.get('recovery_factor'))
    capped_sortino = np.clip(sr, floors.get('sortino_ratio', None), caps.get('sortino_ratio'))
    log_tc = np.log(tc + 1)
    norm_mdd_dur = 1 / (mdd_dur + 1)

    # --- Calculate Weighted Score ---
    score = (
            (capped_pf * weights['capped_pf']) +
            (capped_rf * weights['capped_rf']) +
            (capped_sortino * weights['capped_sortino']) +
            (log_tc * weights['log_tc']) +
            (norm_mdd_dur * weights['norm_mdd_dur'])
    )

    return score if pd.notna(score) else 0.0 # Return 0 if calculation resulted in NaN


# --- File and Summary Processing ---

def process_file(file_name, input_dir):
    """Processes a single trader file to calculate metrics and composite score."""
    logging.info(f"Processing file: {file_name}")
    trader_id = extract_trader_info(file_name)
    file_path = os.path.join(input_dir, file_name)

    # Load and validate data
    df = load_csv_data(file_path)
    if df.empty:
        logging.warning(f"Skipping {file_name} due to load error or empty data.")
        # Return NaNs or default values for this trader
        return trader_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, np.nan # Added NaNs for new metrics

    # Calculate base metrics
    max_drawdown_val = calculate_max_drawdown(df)
    # max_profit_val = calculate_max_profit(df) # Peak profit - not used in new score
    profit_factor_val = calculate_profit_factor(df)
    # risk_reward_balance_val = calculate_risk_reward_balance(max_profit_val, max_drawdown_val, profit_factor_val) # Old metric

    # Calculate metrics needed for isolated score
    total_profit_val = calculate_total_profit(df)
    trade_count_val = calculate_trade_count(df)
    recovery_factor_val = calculate_recovery_factor(total_profit_val, max_drawdown_val)
    sortino_ratio_val = calculate_sortino_ratio(df)
    max_drawdown_duration_val = calculate_max_drawdown_duration(df)

    # Prepare metrics dict for scoring function
    isolated_score_inputs = {
        'profit_factor': profit_factor_val,
        'recovery_factor': recovery_factor_val,
        'sortino_ratio': sortino_ratio_val,
        'tradecount': trade_count_val,
        'max_drawdown_duration': max_drawdown_duration_val
    }

    # Calculate the isolated composite score
    composite_score_val = calculate_single_isolated_score(isolated_score_inputs)
    logging.info(f"Trader {trader_id} | Score Inputs: {isolated_score_inputs} | Isolated Score: {composite_score_val:.4f}")

    # Return trader_id and all calculated metrics, including the new score
    # Adjust the return tuple and the columns in process_and_calculate_summary accordingly
    return (trader_id, max_drawdown_val, total_profit_val, profit_factor_val,
            recovery_factor_val, sortino_ratio_val, trade_count_val,
            max_drawdown_duration_val, composite_score_val)


def process_and_calculate_summary(scenario, input_directory, output_directory):
    """Processes all trader files in a directory, calculates metrics and score, and saves summary."""
    summary_output_dir = os.path.join(output_directory, "summary")
    create_output_directory(summary_output_dir)

    summary_records = []
    logging.info(f"Starting summary calculation for directory: {input_directory}")

    # --- Process each file ---
    files_to_process = [f for f in os.listdir(input_directory) if f.endswith('.csv') and os.path.isfile(os.path.join(input_directory, f))]
    logging.info(f"Found {len(files_to_process)} CSV files to process.")

    for file_name in files_to_process:
        try:
            # Get all metrics from process_file
            record = process_file(file_name, input_directory)
            summary_records.append(record)
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}", exc_info=True)
            # Optionally append a record with NaNs or skip the file

    if not summary_records:
        logging.warning("No trader records were successfully processed. Summary file will be empty.")
        # Create empty files? Or just log and exit?
        # Define column names even if empty
        column_names = ['TraderID', 'MaxDrawdown', 'TotalProfit', 'ProfitFactor',
                        'RecoveryFactor', 'SortinoRatio', 'TradeCount',
                        'MaxDrawdownDuration', 'CompositeScore']
        summary_df = pd.DataFrame(columns=column_names)
    else:
        # --- Create DataFrame ---
        # Update column names to match the returned tuple from process_file
        column_names = ['TraderID', 'MaxDrawdown', 'TotalProfit', 'ProfitFactor',
                        'RecoveryFactor', 'SortinoRatio', 'TradeCount',
                        'MaxDrawdownDuration', 'CompositeScore']
        summary_df = pd.DataFrame(summary_records, columns=column_names)

        # --- Sort and Save Full Summary ---
        # Sort by the new composite score
        summary_df.sort_values(by='CompositeScore', ascending=False, inplace=True, na_position='last')

    summary_file_path = os.path.join(summary_output_dir, f'{scenario}_summary.csv') # Add scenario to filename
    try:
        summary_df.round(4).to_csv(summary_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logging.info(f"Full summary saved to {summary_file_path}")
    except Exception as e:
        logging.error(f"Error saving summary file {summary_file_path}: {e}")


    # --- Filter, Sort, and Save Filtered Summary ---
    if not summary_df.empty:
        try:
            # Use the new composite score for filtering quantile
            # Adjust filtering parameters as needed
            filtered_df = filter_scenarios(
                summary_df,
                composite_threshold=0.90, # Top 10% by isolated score
                min_profit_factor=1.2,
                # Adjust max_drawdown_ratio filter to use TotalProfit if desired
                # Assuming filter_scenarios uses 'MaxProfit' which might be peak profit based on old code
                # For consistency, let's assume filter_scenarios needs updating or we filter manually here
                max_drawdown_ratio=0.5 # MaxDrawdown <= 50% of TotalProfit (Check filter_scenarios logic)
            )
            # Re-sort just in case filtering changed order (unlikely with quantile)
            sorted_filtered_df = sort_scenarios(filtered_df, sort_by='CompositeScore', ascending=False)
            filtered_summary_path = os.path.join(summary_output_dir, f'{scenario}_filtered_summary.csv')
            save_filtered_summary(sorted_filtered_df, filtered_summary_path) # Assumes this saves correctly
            logging.info(f"Filtered summary saved to {filtered_summary_path}")

            # --- Process Filtered Summary (Optional - if still needed) ---
            # This part finds corresponding rows in a base trades file. Check if still required.
            best_trades_base_path = os.path.join(output_directory, "trades", scenario + ".csv") # Path to the raw trades summary
            filtered_trade_path = os.path.join(output_directory, "trades", "filtered-" + scenario + ".csv") # Output for filtered trades
            if os.path.exists(best_trades_base_path):
                 # Ensure output file is empty before starting
                 if os.path.exists(filtered_trade_path): os.remove(filtered_trade_path)
                 process_filtered_summary(filtered_summary_path, scenario, best_trades_base_path, filtered_trade_path)
            else:
                raise FileNotFoundError(f"Base trades file not found: {best_trades_base_path}")

        except Exception as e:
            logging.error(f"Error during filtering or filtered processing: {e}", exc_info=True)
    else:
        logging.warning("Summary DataFrame is empty, skipping filtering.")


# --- Helper functions (Assumed to exist and work as intended) ---

def create_output_directory(path):
    """Creates directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logging.info(f"Output directory '{path}' created or already exists.")

def filter_scenarios(df, composite_threshold=0.90, min_profit_factor=1.2, max_drawdown_ratio=0.5):
    """
    Filters DataFrame based on thresholds.
    NOTE: Assumes 'CompositeScore', 'ProfitFactor', 'MaxDrawdown', 'TotalProfit' columns exist.
          The original used 'MaxProfit' (peak) for the ratio, updated here to use 'TotalProfit'.
    """
    df_filtered = df.copy()
    initial_count = len(df_filtered)
    logging.info(f"Filtering {initial_count} records...")

    # Filter by Composite Score Quantile
    if 'CompositeScore' in df_filtered.columns and not df_filtered['CompositeScore'].isnull().all():
        score_threshold = df_filtered['CompositeScore'].quantile(composite_threshold)
        if pd.notna(score_threshold):
            logging.info(f"Filtering by Composite Score >= {score_threshold:.4f}")
            df_filtered = df_filtered[df_filtered['CompositeScore'] >= score_threshold]
        else:
            logging.warning("Could not calculate composite score threshold. Skipping score filter.")
    else:
        logging.warning("CompositeScore column missing or all NaN. Skipping score filter.")

    # Filter by Profit Factor
    if 'ProfitFactor' in df_filtered.columns:
        pf_numeric = pd.to_numeric(df_filtered['ProfitFactor'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
        logging.info(f"Filtering by Profit Factor >= {min_profit_factor}")
        df_filtered = df_filtered[pf_numeric >= min_profit_factor]
    else:
        logging.warning("ProfitFactor column missing. Skipping profit factor filter.")

    # Filter by Max Drawdown relative to Total Profit
    if 'MaxDrawdown' in df_filtered.columns and 'TotalProfit' in df_filtered.columns:
        max_dd_numeric = pd.to_numeric(df_filtered['MaxDrawdown'], errors='coerce').fillna(0)
        total_profit_numeric = pd.to_numeric(df_filtered['TotalProfit'], errors='coerce')
        # Apply filter only where total profit is positive
        valid_profit_mask = total_profit_numeric > 0
        logging.info(f"Filtering by Max Drawdown <= {max_drawdown_ratio * 100}% of Total Profit (where Total Profit > 0)")
        # Keep rows that are NOT (valid profit AND drawdown exceeds ratio)
        df_filtered = df_filtered[~(valid_profit_mask & (max_dd_numeric > max_drawdown_ratio * total_profit_numeric))]
    else:
        logging.warning("MaxDrawdown or TotalProfit column missing. Skipping drawdown ratio filter.")

    logging.info(f"Filtering complete. {len(df_filtered)} records remaining from {initial_count}.")
    return df_filtered


def sort_scenarios(df, sort_by='CompositeScore', ascending=False):
    """Sorts DataFrame by a specific column."""
    if sort_by in df.columns:
        df_sorted = df.sort_values(by=sort_by, ascending=ascending, na_position='last')
        return df_sorted
    else:
        logging.warning(f"Sort column '{sort_by}' not found. Returning unsorted.")
        return df

def save_filtered_summary(df, output_file):
    """Saves the filtered and sorted summary DataFrame."""
    # Assumes df is already sorted. Rounding might be desired here too.
    logging.info(f"Saving filtered summary to {output_file}")
    try:
        # Round numeric columns before saving
        numeric_cols = df.select_dtypes(include=np.number).columns
        df_save = df.copy()
        df_save[numeric_cols] = df_save[numeric_cols].round(4)
        df_save.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logging.info(f"Successfully saved filtered summary with {len(df_save)} rows.")
    except Exception as e:
        logging.error(f"Error saving filtered summary file {output_file}: {e}")


# --- Functions below related to processing filtered summary (Assumed to exist) ---
# These might need adjustments based on the columns present in the filtered summary CSV

def process_filtered_summary(summary_file, scenario, best_trades_base_path, filtered_trade_path):
    """
    Placeholder: Processes filtered summary to find matching trades in another file.
    """
    logging.info(f"Processing filtered summary {summary_file} to find trades in {best_trades_base_path}")
    # Implementation depends on the structure of best_trades_base_path
    # and whether this step is still required.
    # Ensure the required 'TraderID' column exists in summary_file.
    try:
        summary_df = pd.read_csv(summary_file)
        if 'TraderID' not in summary_df.columns:
            logging.error(f"'TraderID' column not found in {summary_file}. Cannot find matching trades.")
            return

        # Ensure output file is empty/created
        with open(filtered_trade_path, 'w') as f:
            pass # Just create/truncate the file

        # Iterate through filtered traders
        for trader_id in summary_df['TraderID']:
            find_matching_trades(str(trader_id), scenario, best_trades_base_path, filtered_trade_path)

    except Exception as e:
        logging.error(f"Error processing filtered summary {summary_file}: {e}", exc_info=True)


def find_matching_trades(trader_id, scenario, best_trades_base_path, filtered_trade_path):
    """
    Placeholder: Finds and appends matching trades for a trader_id.
    """
    # Implementation depends heavily on the format of best_trades_base_path
    found_match = False
    logging.debug(f"Searching for trader {trader_id} in {best_trades_base_path}")

    try:
        if not os.path.exists(best_trades_base_path):
            logging.error(f"Base trades file not found: {best_trades_base_path}")
            return # Cannot proceed for this trader

        # Read base file line by line or using pandas chunking for large files
        # Example using simple iteration (adjust for actual format)
        header_written = os.path.exists(filtered_trade_path) and os.path.getsize(filtered_trade_path) > 0
        with open(best_trades_base_path, 'r', newline='') as infile, \
                open(filtered_trade_path, 'a', newline='') as outfile:

            reader = csv.DictReader(infile)
            # Ensure 'traderid' column exists in the base file
            if 'traderid' not in reader.fieldnames:
                logging.error(f"'traderid' column not found in {best_trades_base_path}")
                return

            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            if not header_written:
                writer.writeheader()
                header_written = True # Ensure header is written only once

            for row in reader:
                # Compare IDs (ensure consistent type, e.g., both strings)
                if str(row.get('traderid', '')).strip() == str(trader_id).strip():
                    writer.writerow(row)
                    found_match = True
                    logging.debug(f"Found and wrote matching trade for trader ID {trader_id}")
                    # Decide if you need only the first match or all matches
                    # break # Uncomment to stop after first match

        if not found_match:
            logging.debug(f"No matching trade found for trader ID {trader_id} in {best_trades_base_path}")

    except Exception as e:
        logging.error(f"Error finding matching trades for {trader_id}: {e}", exc_info=True)

# --- Example main execution block (if this script were run directly) ---
# if __name__ == "__main__":
#     # Example usage:
#     # Assuming you have CSV files for traders in 'input_trades/scenario_xyz'
#     # and the base trades file is 'output/trades/scenario_xyz.csv'
#     scenario_name = "scenario_xyz"
#     input_dir = os.path.join("input_trades", scenario_name)
#     output_dir = "output" # Base output directory
#
#     if not os.path.isdir(input_dir):
#         print(f"Error: Input directory not found: {input_dir}")
#         sys.exit(1)
#
#     process_and_calculate_summary(scenario_name, input_dir, output_dir)
#     print("Processing complete.")

