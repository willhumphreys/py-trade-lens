# main.py (Modified to Combine Scenarios, Calculate Z-Score on Aggregate)

import argparse
import os
import shutil
import logging
import sys
import subprocess # For calling lzop
import tempfile # For temporary files
import pandas as pd # Import pandas

import boto3
from botocore.exceptions import ClientError # For S3 error handling

# Import the Z-score processing function and helpers from score_calculator
# Ensure this file exists and contains the Z-score logic
# We might need to slightly adjust how process_metrics_file is called or wrap its core logic
from src.processing.score_calculator import (
    calculate_zscore_composite_score,
    filter_strategies,
    sort_strategies,
    save_summary,
    create_output_directory # Import helper if needed directly
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# --- Constants ---
S3_METRICS_BUCKET = "mochi-prod-aggregated-trades" # Bucket containing aggregated metrics

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Download aggregated metrics for ALL scenarios under a symbol, combine, calculate Z-Score, and save results.")
    parser.add_argument("--symbol", required=True, help="The symbol name (e.g. 'AAPL_polygon_min')")
    # Removed --scenario argument
    return parser.parse_args()

def list_scenarios_in_s3(s3_client, bucket, symbol):
    """Lists scenario 'directories' (common prefixes) under the symbol prefix in S3."""
    scenarios = []
    paginator = s3_client.get_paginator('list_objects_v2')
    prefix = f"{symbol}/"
    logging.info(f"Listing scenarios under s3://{bucket}/{prefix}...")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
            if "CommonPrefixes" in page:
                for common_prefix in page.get('CommonPrefixes', []):
                    # Extract scenario name from prefix like 'AAPL_polygon_min/scenario_string/'
                    full_prefix = common_prefix.get('Prefix')
                    if full_prefix:
                        # Remove symbol part and trailing slash
                        scenario_name = full_prefix[len(prefix):].strip('/')
                        if scenario_name: # Ensure it's not empty
                            scenarios.append(scenario_name)
                            logging.debug(f"Found scenario prefix: {full_prefix} -> scenario: {scenario_name}")

        logging.info(f"Found {len(scenarios)} potential scenarios under symbol '{symbol}'.")
        if not scenarios:
            logging.warning(f"No scenarios found directly under s3://{bucket}/{prefix}. Check path structure or bucket content.")
        return scenarios
    except ClientError as e:
        logging.error(f"S3 Error listing scenarios under prefix '{prefix}': {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error listing scenarios: {e}")
        return []


def construct_s3_metrics_key(symbol, scenario):
    """Constructs the S3 key for the aggregated metrics LZO file for a specific scenario."""
    filename = f"aggregated-{symbol}_{scenario}_aggregationQueryTemplate-all.csv.lzo"
    s3_key = f"{symbol}/{scenario}/{filename}"
    # Log the specific key being constructed for clarity
    # logging.info(f"Constructed S3 metrics key for scenario '{scenario}': s3://{S3_METRICS_BUCKET}/{s3_key}")
    return s3_key

def download_from_s3(s3_client, bucket, key, download_path):
    """Downloads a file from S3, creating parent directories if needed."""
    logging.info(f"Attempting to download s3://{bucket}/{key} to {download_path}...")
    try:
        parent_dir = os.path.dirname(download_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        s3_client.download_file(bucket, key, download_path)
        logging.info(f"S3 download successful: {download_path}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logging.warning(f"S3 Warning: File not found at s3://{bucket}/{key}") # Changed to warning
        elif e.response['Error']['Code'] == '403':
            logging.error(f"S3 Error: Access denied for s3://{bucket}/{key}.")
        else:
            logging.error(f"S3 Error downloading file s3://{bucket}/{key}: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during S3 download of s3://{bucket}/{key}: {e}")
        return False

def decompress_lzo(lzo_file_path, output_csv_path):
    """Decompresses an LZO file using the lzop command-line tool."""
    parent_dir = os.path.dirname(output_csv_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    command = ["lzop", "-d", "-f", "-o", output_csv_path, lzo_file_path]
    logging.info(f"Attempting to decompress '{lzo_file_path}' to '{output_csv_path}'...")
    try:
        subprocess.check_output(["which", "lzop"])
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        logging.info(f"LZO decompression successful for {os.path.basename(lzo_file_path)}.")
        logging.debug(f"lzop stdout: {process.stdout}")
        logging.debug(f"lzop stderr: {process.stderr}")
        if not os.path.exists(output_csv_path):
            logging.error(f"Decompression command ran, but output file '{output_csv_path}' was not created.")
            return False
        if os.path.getsize(output_csv_path) == 0:
            logging.warning(f"Decompressed file '{output_csv_path}' is empty.")
        return True
    except FileNotFoundError:
        logging.error("Error: 'lzop' command not found. Please ensure lzop is installed and in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during LZO decompression of '{lzo_file_path}' (return code {e.returncode}):")
        stderr_output = e.stderr.decode('utf-8', errors='replace') if e.stderr else "N/A"
        logging.error(f"lzop stderr: {stderr_output}")
        if os.path.exists(output_csv_path):
            try:
                os.remove(output_csv_path)
                logging.info(f"Removed potentially incomplete output file: {output_csv_path}")
            except OSError as remove_err:
                logging.warning(f"Could not remove incomplete output file '{output_csv_path}': {remove_err}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during decompression of '{lzo_file_path}': {e}", exc_info=True)
        return False


def main():
    # --- Basic Setup ---
    output_dir_base = "output" # Main output folder
    if os.path.exists(output_dir_base):
        logging.info(f"Deleting existing '{output_dir_base}' directory...")
        try:
            shutil.rmtree(output_dir_base)
        except OSError as e:
            logging.error(f"Error removing directory {output_dir_base}: {e}. Manual cleanup might be needed.")
    os.makedirs(output_dir_base, exist_ok=True)
    logging.info(f"Created fresh '{output_dir_base}' directory.")

    args = parse_arguments()
    # Base output directory for the symbol
    symbol_output_directory = os.path.join(output_dir_base, args.symbol)
    temp_base_dir = os.path.join(symbol_output_directory, "temp_combined") # Central temp dir for this run
    final_output_dir = os.path.join(symbol_output_directory, "combined_scores") # Dir for final combined output

    os.makedirs(symbol_output_directory, exist_ok=True)
    os.makedirs(temp_base_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)


    s3_client = boto3.client("s3")

    # --- List Scenarios ---
    scenarios = list_scenarios_in_s3(s3_client, S3_METRICS_BUCKET, args.symbol)
    if not scenarios:
        logging.error(f"No scenarios found for symbol '{args.symbol}'. Exiting.")
        sys.exit(1)

    # --- Loop Through Scenarios to Download, Decompress, and Read Data ---
    all_scenario_dfs = []
    failed_downloads_or_decomp = []
    processed_scenario_count = 0

    for scenario in scenarios:
        logging.info(f"--- Processing Scenario for Data Aggregation: {scenario} ---")

        # Define paths within the central temp dir
        temp_lzo_path = os.path.join(temp_base_dir, f"{args.symbol}_{scenario}_metrics.csv.lzo")
        decompressed_csv_path = os.path.join(temp_base_dir, f"{args.symbol}_{scenario}_metrics.csv")

        # Download
        s3_metrics_key = construct_s3_metrics_key(args.symbol, scenario)
        if not download_from_s3(s3_client, S3_METRICS_BUCKET, s3_metrics_key, temp_lzo_path):
            logging.warning(f"Failed to download metrics file for scenario '{scenario}'. Skipping.")
            failed_downloads_or_decomp.append(scenario + " (Download Failed)")
            continue # Move to the next scenario

        # Decompress
        if not decompress_lzo(temp_lzo_path, decompressed_csv_path):
            logging.warning(f"Failed to decompress metrics file for scenario '{scenario}'. Skipping.")
            failed_downloads_or_decomp.append(scenario + " (Decompression Failed)")
            # Clean up downloaded LZO if decompression failed
            try:
                if os.path.exists(temp_lzo_path): os.remove(temp_lzo_path)
            except OSError: pass
            continue # Move to the next scenario

        # Read CSV into DataFrame
        try:
            logging.info(f"Reading decompressed CSV: {decompressed_csv_path}")
            # Attempt to read with low_memory=False for potentially mixed types
            df = pd.read_csv(decompressed_csv_path, low_memory=False)
            # Add scenario identifier
            df['scenario'] = scenario
            all_scenario_dfs.append(df)
            processed_scenario_count += 1
            logging.info(f"Successfully read and stored data for scenario '{scenario}'.")
        except pd.errors.EmptyDataError:
            logging.warning(f"Metrics file for scenario '{scenario}' is empty. Skipping.")
            failed_downloads_or_decomp.append(scenario + " (Empty File)")
        except Exception as e:
            logging.error(f"Error reading CSV for scenario '{scenario}': {e}", exc_info=True)
            failed_downloads_or_decomp.append(scenario + " (Read Error)")

        # Clean up individual decompressed CSV and LZO file after reading
        finally:
            try:
                if os.path.exists(temp_lzo_path): os.remove(temp_lzo_path)
                if os.path.exists(decompressed_csv_path): os.remove(decompressed_csv_path)
            except OSError as e:
                logging.warning(f"Could not remove temp files for scenario '{scenario}': {e}")

    # --- End of Scenario Loop ---

    # --- Combine DataFrames ---
    if not all_scenario_dfs:
        logging.error(f"No scenario data was successfully processed for symbol '{args.symbol}'. Cannot calculate combined scores.")
        sys.exit(1)

    logging.info(f"Combining data from {len(all_scenario_dfs)} scenarios...")
    try:
        combined_df = pd.concat(all_scenario_dfs, ignore_index=True)
        logging.info(f"Combined DataFrame created with {len(combined_df)} total records.")
        # Optional: Log memory usage
        logging.info(f"Combined DataFrame memory usage: {combined_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    except Exception as e:
        logging.error(f"Error concatenating DataFrames: {e}", exc_info=True)
        sys.exit(1)

    # --- Calculate Z-Score Composite Score on Combined Data ---
    logging.info("Calculating Z-Score composite scores on combined data...")
    try:
        # The calculate function modifies the DataFrame in place
        combined_df_with_scores = calculate_zscore_composite_score(combined_df.copy()) # Pass a copy
        logging.info("Z-Score composite score calculation complete for combined data.")
    except ValueError as e: # Catch missing columns error from score calculator
        logging.error(f"FATAL: Error calculating composite score on combined data: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: Unexpected error during combined Z-Score processing: {e}", exc_info=True)
        sys.exit(1)

    # --- Save Combined Results ---
    logging.info("Sorting combined results by CompositeScore...")
    combined_df_sorted = sort_strategies(combined_df_with_scores, sort_by='CompositeScore', ascending=False)

    combined_output_filename = os.path.join(final_output_dir, f"{args.symbol}_all_scenarios_zscores.csv")
    logging.info(f"Saving combined results with Z-Scores to: {combined_output_filename}")
    save_summary(combined_df_sorted, combined_output_filename) # Use the save function from score_calculator


    logging.info("Filtering combined strategies...")
    filtered_combined_df = filter_strategies(
        combined_df_sorted.copy(),
        composite_quantile_threshold=0.95, # Example: Top 5% across all scenarios
        min_profit_factor=1.0, # Adjust filters as needed
        max_drawdown_ratio=1.0
    )
    filtered_output_filename = os.path.join(final_output_dir, f"{args.symbol}_all_scenarios_filtered_zscores.csv")
    logging.info(f"Saving filtered combined results to: {filtered_output_filename}")
    save_summary(filtered_combined_df, filtered_output_filename)


    # --- Final Cleanup ---
    logging.info(f"Cleaning up temporary base directory: {temp_base_dir}")
    try:
        shutil.rmtree(temp_base_dir)
    except OSError as e:
        logging.warning(f"Could not remove temporary base directory '{temp_base_dir}': {e}")

    # --- Final Summary Log ---
    logging.info("="*50)
    logging.info(f"Overall Processing Summary for Symbol: {args.symbol}")
    logging.info(f"Total scenarios found: {len(scenarios)}")
    logging.info(f"Scenarios successfully read: {processed_scenario_count}")
    logging.info(f"Scenarios failed/skipped: {len(failed_downloads_or_decomp)}")
    if failed_downloads_or_decomp:
        logging.warning("Failed/Skipped Scenarios during data aggregation:")
        for fs in failed_downloads_or_decomp:
            logging.warning(f"  - {fs}")
    logging.info(f"Combined results saved in: {final_output_dir}")
    logging.info("Processing finished.")
    logging.info("="*50)


if __name__ == "__main__":
    main()