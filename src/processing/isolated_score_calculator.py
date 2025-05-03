# src/processing/score_calculator.py (Using Isolated Score Method)

import os
import pandas as pd
import numpy as np
import logging
import sys
import csv

# Configure logging (ensure it's configured in the main script)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_output_directory(path):
    """Creates the output directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logging.info(f"Output directory '{path}' created or already exists.")
    elif not path:
        logging.warning("Path provided for directory creation is empty or None.")

# Removed Z-score functions

def calculate_isolated_composite_score(df):
    """
    Calculates a composite score for each trader based *only* on their individual metrics.
    Applies transformations and caps to handle different scales and extreme values.
    Assumes input DataFrame contains the necessary raw metric columns.

    Args:
        df (pd.DataFrame): DataFrame containing raw metrics for traders (can be for one or multiple scenarios).

    Returns:
        pd.DataFrame: DataFrame with added 'CompositeScore' column.

    Raises:
        ValueError: If required columns for calculation are missing.
    """
    # --- 1. Define Required Columns and Parameters for Scoring ---
    # Ensure these columns are present in the input DataFrame (likely from Athena)
    required_columns = [
        'profit_factor',
        'recovery_factor',
        'sortino_ratio',
        'tradecount',
        'max_drawdown_duration'
        # Add 'traderid' if needed for logging/debugging within this function
        # Add 'scenario' if you want to log it here
    ]
    # Define caps and floors to manage extreme values (adjust these based on expected ranges/importance)
    caps = {
        'profit_factor': 10.0,    # Example: Cap profit factor at 10
        'recovery_factor': 50.0,  # Example: Cap recovery factor at 50
        'sortino_ratio': 5.0      # Example: Cap Sortino at 5
    }
    floors = {
        'sortino_ratio': 0.0      # Example: Ensure Sortino doesn't go below 0 for scoring
        # Add floors for others if needed, e.g., profit_factor: 0.0
    }
    # Define weights for combining the metrics (ensure they sum to 1 or scale score later)
    weights = {
        'capped_pf': 0.30,
        'capped_rf': 0.30,
        'capped_sortino': 0.20,
        'log_tc': 0.10,
        'norm_mdd_dur': 0.10
    }
    logging.info("Using Isolated Score parameters:")
    logging.info(f"  Required Columns: {required_columns}")
    logging.info(f"  Caps: {caps}")
    logging.info(f"  Floors: {floors}")
    logging.info(f"  Weights: {weights}")


    # --- 2. Check for Required Columns ---
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input data for isolated composite score: {missing_cols}. "
                         f"Please ensure the metrics file contains these columns: {required_columns}")

    # --- 3. Prepare Data and Apply Transformations/Caps ---
    df_score = df.copy() # Work on a copy to avoid modifying original df directly

    logging.info("Preparing data for isolated scoring...")
    # Convert columns to numeric, coercing errors
    for col in required_columns:
        if col in df_score.columns:
            df_score[col] = pd.to_numeric(df_score[col], errors='coerce')
        else:
            # This case is caught by the check above, but added for safety
            logging.error(f"Required column '{col}' missing during numeric conversion.")
            df_score[col] = np.nan # Add column as NaN if somehow missed

    # Handle potential infinite values first
    logging.info("Handling infinite values...")
    for col in ['profit_factor', 'recovery_factor', 'sortino_ratio']:
        if col in df_score.columns:
            inf_count = np.isinf(df_score[col]).sum()
            if inf_count > 0:
                logging.info(f"Replacing {inf_count} inf/-inf values in '{col}' with NaN.")
                df_score[col] = df_score[col].replace([np.inf, -np.inf], np.nan)
            # Fill NaNs (from coercion or inf replacement) with 0 before capping/flooring
            # Consider if 0 is the right fill value for your logic (e.g., maybe 1 for profit factor?)
            fill_value = 0
            nan_count = df_score[col].isnull().sum()
            if nan_count > 0:
                logging.info(f"Filling {nan_count} NaN values in '{col}' with {fill_value}.")
                df_score[col] = df_score[col].fillna(fill_value)


    # Apply Caps and Floors
    logging.info("Applying caps and floors...")
    if 'profit_factor' in df_score.columns:
        floor = floors.get('profit_factor', None) # Get floor if defined, else None
        df_score['capped_pf'] = df_score['profit_factor'].clip(lower=floor, upper=caps.get('profit_factor'))
        logging.debug(f"Applied floor={floor}, cap={caps.get('profit_factor')} to profit_factor -> capped_pf.")
    if 'recovery_factor' in df_score.columns:
        floor = floors.get('recovery_factor', None)
        df_score['capped_rf'] = df_score['recovery_factor'].clip(lower=floor, upper=caps.get('recovery_factor'))
        logging.debug(f"Applied floor={floor}, cap={caps.get('recovery_factor')} to recovery_factor -> capped_rf.")
    if 'sortino_ratio' in df_score.columns:
        floor = floors.get('sortino_ratio', None)
        df_score['capped_sortino'] = df_score['sortino_ratio'].clip(lower=floor, upper=caps.get('sortino_ratio'))
        logging.debug(f"Applied floor={floor}, cap={caps.get('sortino_ratio')} to sortino_ratio -> capped_sortino.")

    # Apply Transformations
    logging.info("Applying transformations (log tradecount, normalize duration)...")
    if 'tradecount' in df_score.columns:
        # Ensure tradecount is numeric, non-negative, and fill NaNs
        df_score['tradecount'] = pd.to_numeric(df_score['tradecount'], errors='coerce').fillna(0).clip(lower=0)
        df_score['log_tc'] = np.log(df_score['tradecount'] + 1) # Add 1 before log to handle 0 count
        logging.debug("Calculated log(tradecount + 1) -> log_tc.")
    if 'max_drawdown_duration' in df_score.columns:
        # Ensure duration is numeric, non-negative, and fill NaNs
        df_score['max_drawdown_duration'] = pd.to_numeric(df_score['max_drawdown_duration'], errors='coerce').fillna(0).clip(lower=0)
        # Invert duration so higher score is better, add 1 to avoid division by zero
        df_score['norm_mdd_dur'] = 1 / (df_score['max_drawdown_duration'] + 1)
        logging.debug("Calculated 1 / (max_drawdown_duration + 1) -> norm_mdd_dur.")


    # --- 4. Calculate Weighted Composite Score ---
    logging.info("Calculating final weighted isolated composite score...")
    # Initialize score column
    df_score['CompositeScore'] = 0.0
    missing_component_cols = []

    # Sum weighted components
    for component, weight in weights.items():
        if component in df_score.columns:
            # Ensure component column is numeric before weighting
            component_values = pd.to_numeric(df_score[component], errors='coerce').fillna(0)
            df_score['CompositeScore'] += component_values * weight
            logging.debug(f"Added weighted component '{component}' (Weight: {weight})")
        else:
            missing_component_cols.append(component)
            logging.warning(f"Component '{component}' needed for score calculation is missing. Skipping.")

    if missing_component_cols:
        logging.error(f"Composite score calculation potentially incomplete due to missing components: {missing_component_cols}")


    # --- 5. Add Score to Original DataFrame ---
    # Add the final score back to the original DataFrame passed into the function
    df['CompositeScore'] = df_score['CompositeScore']
    logging.info("Isolated Composite Score calculation complete.")

    # Optional: Clean up intermediate columns from the *copy* if desired,
    # but returning the original df with just the new score is cleaner.
    # cols_to_drop = ['capped_pf', 'capped_rf', 'capped_sortino', 'log_tc', 'norm_mdd_dur']
    # df_score.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    return df # Return the original DataFrame with the 'CompositeScore' column added/updated


# --- Functions needed by main script (filtering, sorting, saving) ---
# These operate on the DataFrame *after* the score has been calculated

def filter_strategies(df, composite_quantile_threshold=0.90, min_profit_factor=1.2, max_drawdown_ratio=0.5):
    """
    Filters strategies based on the ISOLATED composite score and potentially other raw metrics.
    Note: Filtering by quantile might be less meaningful for an isolated score unless
          you still want to select the top % based on this score within the current subset.

    Parameters:
    - df (pd.DataFrame): DataFrame containing trader metrics including 'CompositeScore'.
    - composite_quantile_threshold: Percentile threshold for filtering by the composite score.
                                     Set to 0 or None to disable quantile filtering.
    - min_profit_factor: Minimum acceptable raw profit factor.
    - max_drawdown_ratio: Maximum allowed ratio of max drawdown to total profit.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df_filtered = df.copy() # Work on a copy
    initial_count = len(df_filtered)
    if initial_count == 0:
        logging.warning("Input DataFrame for filtering is empty. Returning empty DataFrame.")
        return df_filtered

    logging.info(f"Filtering strategies. Initial count: {initial_count}")

    # --- Filter by Composite Score Quantile (Optional for Isolated Score) ---
    if composite_quantile_threshold is not None and composite_quantile_threshold > 0:
        if 'CompositeScore' in df_filtered.columns and not df_filtered['CompositeScore'].isnull().all():
            df_filtered['CompositeScore'] = pd.to_numeric(df_filtered['CompositeScore'], errors='coerce')
            if not df_filtered['CompositeScore'].isnull().all():
                try:
                    score_threshold = df_filtered['CompositeScore'].quantile(composite_quantile_threshold)
                    if pd.isna(score_threshold):
                        logging.warning(f"Could not calculate {composite_quantile_threshold} quantile for CompositeScore. Skipping score filter.")
                    else:
                        logging.info(f"Filtering by Composite Score >= {score_threshold:.4f} ({composite_quantile_threshold*100:.1f}th percentile)")
                        df_filtered = df_filtered[df_filtered['CompositeScore'] >= score_threshold]
                        logging.info(f"Strategies after Composite Score filter: {len(df_filtered)} ({len(df_filtered)/initial_count*100:.1f}%)")
                except Exception as e:
                    logging.warning(f"Error calculating CompositeScore quantile: {e}. Skipping score filter.")
            else:
                logging.warning("CompositeScore column contains only NaNs after coercion. Skipping score filter.")
        else:
            logging.warning("CompositeScore column not found or all NaN. Skipping score filter.")
    else:
        logging.info("Quantile filtering based on Composite Score is disabled.")


    # --- Filter by Raw Profit Factor ---
    if 'profit_factor' in df_filtered.columns:
        pf_numeric = pd.to_numeric(df_filtered['profit_factor'], errors='coerce').fillna(0)
        logging.info(f"Filtering by Profit Factor >= {min_profit_factor}")
        count_before = len(df_filtered)
        df_filtered = df_filtered[pf_numeric >= min_profit_factor]
        removed_count = count_before - len(df_filtered)
        logging.info(f"Strategies after Profit Factor filter: {len(df_filtered)} (Removed {removed_count})")
    else:
        logging.warning("profit_factor column not found. Skipping profit factor filter.")

    # --- Filter by Max Drawdown relative to Total Profit ---
    if 'max_drawdown' in df_filtered.columns and 'totalprofit' in df_filtered.columns:
        max_dd_numeric = pd.to_numeric(df_filtered['max_drawdown'], errors='coerce')
        total_profit_numeric = pd.to_numeric(df_filtered['totalprofit'], errors='coerce')

        valid_comparison = (total_profit_numeric > 0) & (~max_dd_numeric.isna()) & (~total_profit_numeric.isna())
        exceeds_ratio = max_dd_numeric > max_drawdown_ratio * total_profit_numeric

        rows_to_remove_mask = valid_comparison & exceeds_ratio
        num_to_remove = rows_to_remove_mask.sum()

        logging.info(f"Filtering by Max Drawdown <= {max_drawdown_ratio * 100}% of Total Profit (where Total Profit > 0)")
        df_filtered = df_filtered[~rows_to_remove_mask]
        logging.info(f"Strategies after Max Drawdown filter: {len(df_filtered)} (Removed {num_to_remove})")
    else:
        logging.warning("max_drawdown or totalprofit column not found. Skipping max drawdown filter.")


    final_count = len(df_filtered)
    logging.info(f"Filtering complete. Final count: {final_count} ({final_count/initial_count*100:.1f}% of initial)")
    return df_filtered

def sort_strategies(df, sort_by='CompositeScore', ascending=False):
    """Sort strategies DataFrame by a specific column."""
    if sort_by in df.columns:
        logging.info(f"Sorting strategies by {sort_by} {'ascending' if ascending else 'descending'}")
        df_sort = df.copy()
        df_sort[sort_by] = pd.to_numeric(df_sort[sort_by], errors='coerce')
        return df_sort.sort_values(by=sort_by, ascending=ascending, na_position='last')
    else:
        logging.warning(f"Sort column '{sort_by}' not found. Returning unsorted DataFrame.")
        return df

def save_summary(df, output_file):
    """Rounds specified numeric columns and saves the DataFrame to a CSV file."""
    if df.empty:
        logging.warning(f"DataFrame is empty. Skipping save to {output_file}")
        return

    logging.info(f"Saving summary to {output_file}")
    df_save = df.copy()
    numeric_cols = df_save.select_dtypes(include=np.number).columns
    cols_to_round = [
        col for col in numeric_cols
        if not ('id' in col.lower() or 'count' in col.lower() or 'duration' in col.lower() or 'year' in col.lower())
    ]

    try:
        df_save[cols_to_round] = df_save[cols_to_round].round(4)
        logging.info(f"Rounded numeric columns: {cols_to_round}")
    except Exception as e:
        logging.warning(f"Could not round columns: {e}")

    try:
        parent_dir = os.path.dirname(output_file)
        if parent_dir:
            create_output_directory(parent_dir)
        df_save.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logging.info(f"Successfully saved summary with {len(df_save)} rows to {output_file}")
    except Exception as e:
        logging.error(f"Error saving CSV file to {output_file}: {e}")


# --- Main Processing Function (to be called by the main script) ---
# This function now processes a single metrics file using the ISOLATED score.
# It no longer inherently assumes Z-score, the calculation is done inside calculate_isolated_composite_score.
def process_metrics_file_isolated(scenario_name, metrics_file_path, output_directory):
    """
    Main processing function for Isolated Score method. Reads a metrics CSV file,
    calculates the isolated composite score, filters, sorts, and saves summaries.

    Args:
        scenario_name (str): Name of the scenario being processed (used for output file names).
                             Can be None if processing a combined file.
        metrics_file_path (str): Path to the single CSV file containing pre-calculated metrics.
        output_directory (str): Base directory where summary and filtered folders will be created.
    """
    logging.info(f"--- Starting ISOLATED Score processing for: {scenario_name or 'Combined File'} ---")
    logging.info(f"Reading metrics file from: {metrics_file_path}")

    # --- Create Output Dirs ---
    summary_output_dir = os.path.join(output_directory, "summary")
    filtered_output_dir = os.path.join(output_directory, "filtered")
    create_output_directory(summary_output_dir)
    create_output_directory(filtered_output_dir)

    # --- Load Data ---
    try:
        # Use low_memory=False if columns have mixed types, otherwise default is fine
        summary_df = pd.read_csv(metrics_file_path, low_memory=False)
        logging.info(f"Loaded {len(summary_df)} records from metrics file: {metrics_file_path}")
        summary_df.columns = summary_df.columns.str.strip() # Clean column names
        logging.info(f"Columns loaded: {summary_df.columns.tolist()}")

        # Check for traderid (case-insensitive) - important for potential later steps
        traderid_col_found = False
        for col in summary_df.columns:
            if col.lower() == 'traderid':
                if col != 'traderid':
                    summary_df.rename(columns={col: 'traderid'}, inplace=True)
                traderid_col_found = True
                break
        if not traderid_col_found:
            logging.warning("Column 'traderid' (case-insensitive) not found.")

    except FileNotFoundError:
        logging.error(f"FATAL: Metrics file not found: {metrics_file_path}")
        sys.exit(1) # Exit if input file is missing
    except Exception as e:
        logging.error(f"FATAL: Error loading metrics CSV from {metrics_file_path}: {e}", exc_info=True)
        sys.exit(1) # Exit on load error

    # --- Calculate ISOLATED Composite Score ---
    logging.info("Calculating ISOLATED composite score...")
    try:
        # This function modifies summary_df by adding 'CompositeScore'
        summary_df_with_scores = calculate_isolated_composite_score(summary_df) # Call the isolated score function
    except ValueError as e: # Catch missing columns error
        logging.error(f"FATAL: Error calculating composite score: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: Unexpected error during composite score calculation: {e}", exc_info=True)
        sys.exit(1)


    # --- Save Full Summary (with ISOLATED composite score) ---
    logging.info("Sorting full summary by CompositeScore...")
    summary_df_sorted = sort_strategies(summary_df_with_scores.copy(), sort_by='CompositeScore', ascending=False)
    # Use suffix to indicate isolated score method
    file_prefix = scenario_name if scenario_name else "combined"
    full_summary_file_path = os.path.join(summary_output_dir, f'{file_prefix}_full_summary_isolated_score.csv')
    save_summary(summary_df_sorted, full_summary_file_path)

    # --- Filter Strategies ---
    logging.info("Filtering strategies based on ISOLATED composite score and other criteria...")
    # Adjust filter parameters as needed
    filtered_df = filter_strategies(
        summary_df_sorted.copy(),
        composite_quantile_threshold=0.90, # Example: Keep top 10% by isolated score (adjust or disable)
        min_profit_factor=1.2,
        max_drawdown_ratio=0.5
    )

    # --- Sort Filtered Strategies ---
    logging.info("Sorting filtered summary by CompositeScore...")
    sorted_filtered_df = sort_strategies(filtered_df, sort_by='CompositeScore', ascending=False)

    # --- Save Filtered Summary ---
    filtered_summary_path = os.path.join(filtered_output_dir, f'{file_prefix}_filtered_summary_isolated_score.csv')
    save_summary(sorted_filtered_df, filtered_summary_path)

    logging.info(f"--- ISOLATED Score processing finished successfully for: {scenario_name or 'Combined File'} ---")

