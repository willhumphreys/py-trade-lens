import argparse
import os
import shutil

import boto3

from extractor import download_and_unzip_trades
from processing.coloured_trades_and_profit import process_and_plot_files
from src.processing.download_and_filter_minute_stock_data import download_and_read_minute_data, filter_valid_minute_data
from src.processing.enrich_trades import process_and_calculate_summary
from src.processing.trade_execution_verifier import verify_trade_execution
from trader_verification import verify_matching_trader_ids
from uploading.s3_directory_compression_utilities import compress_and_push_all_scenarios


def parse_arguments():
    parser = argparse.ArgumentParser(description="Download and unpack a trade archive from S3.")
    parser.add_argument("--symbol", required=True, help="The symbol name (e.g. 'btc-1mF')")
    parser.add_argument("--scenario", required=True,
                        help="The scenario string (e.g. 's_-3000..-100..400___l_100..7500..400___...')")
    return parser.parse_args()


def main():
    # Clean up the output directory
    output_dir = "output"
    if os.path.exists(output_dir):
        print(f"Deleting existing '{output_dir}' directory...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created fresh '{output_dir}' directory.")

    args = parse_arguments()
    output_directory = os.path.join(output_dir, args.symbol, args.scenario)

    os.makedirs(output_directory, exist_ok=True)

    s3_client = boto3.client("s3")

    minute_data = download_and_read_minute_data(args.symbol, s3_client, output_directory)

    # Display information about the data
    print(f"\nMinute data for {args.symbol}:")
    print(f"Total rows: {len(minute_data)}")

    valid_data = filter_valid_minute_data(minute_data)
    print(f"Valid price rows: {len(valid_data)}")

    trade_extracts_bucket =  os.environ.get('MOCHI_PROD_TRADE_EXTRACTS')

    download_and_unzip_trades(args.symbol, args.scenario, output_directory, trade_extracts_bucket, s3_client)
    formatted_trades_dir = os.path.join(output_directory, "trades", "formatted-trades")
    verify_matching_trader_ids(formatted_trades_dir, os.path.join(output_directory, "trades", args.scenario + ".csv"))
    verify_trade_execution(formatted_trades_dir, valid_data)

    process_and_calculate_summary(args.scenario, formatted_trades_dir, output_directory)
    process_and_plot_files(formatted_trades_dir, os.path.join(output_directory, "graphs"), valid_data)

    compress_and_push_all_scenarios(os.path.join(output_dir, args.symbol), "mochi-prod-trade-performance-graphs", s3_client)

if __name__ == "__main__":
    main()
