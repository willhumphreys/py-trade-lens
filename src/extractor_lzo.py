# extractor_lzo.py
import os
import boto3
import subprocess


def download_and_decompress_lzo(symbol, scenario, output_dir="output", bucket_name="mochi-traders"):
    """
    Downloads an LZO-compressed CSV file from S3, saves the archive into an archive directory,
    and decompresses it into an output directory under "traders".
    """
    s3_key = f"{symbol}/{scenario}/traders--{scenario}___{symbol}.csv.lzo"

    # Set up archive destination for traders
    archives_dir = os.path.join(output_dir, "archives", "traders")
    os.makedirs(archives_dir, exist_ok=True)
    local_lzo_file = os.path.join(archives_dir, f"traders-{scenario}.csv.lzo")

    print(f"Downloading s3://{bucket_name}/{s3_key} to {local_lzo_file} ...")
    s3_client = boto3.client("s3")
    s3_client.download_file(bucket_name, s3_key, local_lzo_file)

    # Set up output destination for the decompressed CSV file
    traders_output_dir = os.path.join(output_dir, "traders")
    os.makedirs(traders_output_dir, exist_ok=True)
    destination_csv_path = os.path.join(traders_output_dir, f"traders-{scenario}.csv")

    print(f"Decompressing {local_lzo_file} into {destination_csv_path} ...")
    try:
        with open(destination_csv_path, "wb") as csv_out:
            subprocess.run(["lzop", "-d", "-c", local_lzo_file], stdout=csv_out, check=True)
    except subprocess.CalledProcessError as e:
        print("Decompression failed:", e)
        raise

    print(f"LZO decompression completed. CSV written to {destination_csv_path}")
