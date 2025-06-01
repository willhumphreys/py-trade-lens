import os
import pandas as pd
import boto3
import tempfile
import subprocess
from pathlib import Path


def download_and_read_minute_data(symbol, s3_client, output_dir=None, back_test_id=None):
    """
    Download and read minute data for a given stock symbol from S3.

    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL')
        s3_client: The boto3 S3 client object.
        output_dir (str, optional): Directory to save decompressed file. If None, uses a temp directory
        back_test_id (str, optional): The back test ID to prefix S3 keys with.

    Returns:
        pandas.DataFrame: Minute data for the specified symbol
    """
    # Get the bucket name from environment variable
    data_bucket = os.environ.get('MOCHI_DATA_BUCKET')
    if not data_bucket:
        raise ValueError("MOCHI_DATA_BUCKET environment variable not set")

    # Construct the S3 key for the file
    base_key = f"stocks/{symbol.split('_')[0]}/polygon/{symbol}.csv.lzo"
    s3_key = f"{back_test_id}/{base_key}" if back_test_id else base_key

    print(f"Downloading minute data for {symbol} from s3://{data_bucket}/{s3_key}")

    # Paths for compressed and decompressed files
    compressed_file_path = os.path.join(output_dir, f"{symbol}.csv.lzo")
    decompressed_file_path = os.path.join(output_dir, f"{symbol}.csv")

    try:
        # Download the compressed file from S3
        s3_client.download_file(data_bucket, s3_key, compressed_file_path)
        print(f"Downloaded file to {compressed_file_path}")

        # Decompress the LZO file using lzop command line tool
        try:
            subprocess.run(
                ["lzop", "-d", compressed_file_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"Decompressed file to {decompressed_file_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to decompress LZO file: {e.stderr.decode()}")
        except FileNotFoundError:
            raise RuntimeError("lzop command not found. Please install lzop: sudo apt-get install lzop")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(decompressed_file_path)

        # Parse the dateTime column as datetime
        df['dateTime'] = pd.to_datetime(df['dateTime'])

        return df

    except Exception as e:
        print(f"Error processing minute data for {symbol}: {str(e)}")
        raise
    finally:
        # Clean up temporary files if they exist
        if os.path.exists(compressed_file_path):
            os.remove(compressed_file_path)
            print(f"Removed compressed file: {compressed_file_path}")


def filter_valid_minute_data(df):
    """
    Filter out invalid rows (those with -1 values) from minute data.

    Args:
        df (pandas.DataFrame): Minute data DataFrame

    Returns:
        pandas.DataFrame: Filtered DataFrame with only valid rows
    """
    # Filter out rows with -1 in open, high, low, close, or volume
    return df[(df['open'] != -1) & (df['high'] != -1) & (df['low'] != -1) & 
              (df['close'] != -1) & (df['volume'] != -1)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and read minute data for a stock symbol")
    parser.add_argument("--symbol", required=True, help="Stock ticker symbol (e.g., 'AAPL')")
    parser.add_argument("--output-dir", help="Directory to save files")
    args = parser.parse_args()

    s3_client = boto3.client("s3")

    minute_data = download_and_read_minute_data(args.symbol, s3_client, args.output_dir)

    # Display information about the data
    print(f"\nMinute data for {args.symbol}:")
    print(f"Total rows: {len(minute_data)}")

    valid_data = filter_valid_minute_data(minute_data)
    print(f"Valid price rows: {len(valid_data)}")

    print("\nSample data:")
    print(valid_data.head())
