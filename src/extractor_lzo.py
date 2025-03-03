import boto3
import subprocess

def download_and_decompress_lzo(symbol, scenario, bucket_name="mochi-traders"):
    """
    Downloads an LZO-compressed CSV file from S3 and decompresses it locally using lzop,
    which is better suited for lzop-formatted files.
    """
    s3_key = f"{symbol}/{scenario}/traders--{scenario}___{symbol}.csv.lzo"

    # Local filenames
    local_lzo_file = "traders.csv.lzo"
    local_csv_file = "traders.csv"

    # Download the LZO file from S3
    print(f"Downloading s3://{bucket_name}/{s3_key} to {local_lzo_file}")
    s3_client = boto3.client("s3")
    s3_client.download_file(bucket_name, s3_key, local_lzo_file)

    # Decompress the LZO file using the system's lzop tool
    print(f"Decompressing {local_lzo_file} into {local_csv_file}")
    try:
        with open(local_csv_file, "wb") as csv_out:
            subprocess.run(["lzop", "-d", "-c", local_lzo_file], stdout=csv_out, check=True)
    except subprocess.CalledProcessError as e:
        print("Decompression failed:", e)
        raise

    # (Optional) remove the .lzo file after successful decompression
    # os.remove(local_lzo_file)

    print(f"LZO decompression completed. CSV written to {local_csv_file}")
