import argparse
import os
import shutil

from extractor import download_and_unzip_trades
from processing.coloured_trades_and_profit import process_and_plot_files
from src.processing.enrich_trades import process_and_calculate_summary
from trader_verification import verify_matching_trader_ids


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

    download_and_unzip_trades(args.symbol, args.scenario, output_directory, "mochi-trade-extracts")
    formatted_trades_dir = os.path.join(output_directory, "trades", "formatted-trades")
    verify_matching_trader_ids(formatted_trades_dir, os.path.join(output_directory, "trades", args.scenario + ".csv"))

    process_and_calculate_summary(args.scenario, formatted_trades_dir, output_directory)
    process_and_plot_files(formatted_trades_dir, output_directory.join("graphs"))


if __name__ == "__main__":
    main()
