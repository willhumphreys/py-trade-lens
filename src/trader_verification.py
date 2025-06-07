# trader_verification.py
import csv
import os
import sys

import boto3

from extractor import download_and_unzip_trades
from extractor_lzo import download_and_decompress_trader_file


def verify_matching_trader_ids(trades_dir, trader_csv_file):
    """
    Confirm that for each trade file in the given trades directory,
    the filename (without the '.csv' extension) exists in the trader CSV file.

    Assumes the trader CSV file has a header with a column named 'traderId'.

    :param trades_dir: Directory where trade CSV files are stored.
    :param trader_csv_file: Path to the trader CSV file.
    :raises RuntimeError: If a matching traderId is not found.
    """
    trader_ids = set()
    with open(trader_csv_file, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Assumes column name is exactly "traderId"
            trader_ids.add(str(row["traderid"]).strip())

    print(f"Found trader IDs in trader CSV: {trader_ids}")

    # For each CSV file in the trades directory, check for a matching traderId.
    for fname in os.listdir(trades_dir):
        if fname.lower().endswith(".csv"):
            trade_trader_id = os.path.splitext(fname)[0]
            if trade_trader_id not in trader_ids:
                raise RuntimeError(f"Trade file '{fname}' does not have a matching traderId in {trader_csv_file}")
            else:
                print(f"Verified: Trade file '{fname}' matches traderId.")

    print("All trade files have matching trader IDs.")


# def run_pipeline(symbol, scenario, output_dir, s3_client):
#     """
#     Run the full pipeline:
#       1. Download and unzip the trade archive.
#       2. Download and decompress the trader file.
#       3. Verify that every trade CSV file in output/trades has a matching traderId
#          from the trader CSV file.
#
#     :param symbol: The symbol name (example: "btc-1mF")
#     :param scenario: The scenario name (example: "s_-3000..-100..400___")
#     :param output_dir: Base directory for storing archives and extracted output.
#     """
#     print("Starting trade extraction...")
#     download_and_unzip_trades(symbol, scenario, output_dir, "mochi-prod-trade-extracts", s3_client)
#
#     print("Starting trader extraction...")
#     trader_csv = download_and_decompress_trader_file(symbol, scenario, output_dir=output_dir)
#
#     trades_dir = os.path.join(output_dir, "trades")
#     print("Starting verification of trade files...")
#     verify_matching_trader_ids(trades_dir, trader_csv)
#     print("Pipeline completed successfully.")
#
#
# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python trader_verification.py <symbol> <scenario>")
#         sys.exit(1)
#
#     symbol = sys.argv[1]
#     scenario = sys.argv[2]
#
#     s3_client = boto3.client("s3")
#
#     run_pipeline(symbol, scenario, os.path.join("output", symbol, scenario), s3_client)
