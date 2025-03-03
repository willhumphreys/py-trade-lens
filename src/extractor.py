import boto3
import zipfile

def download_and_unzip(symbol, scenario, bucket_name="mochi-trade-extracts"):
    """
    Downloads the specified trade archive from the given S3 bucket
    and unzips it into a folder named after the scenario.

    :param symbol: The symbol name (e.g. "btc-1mF")
    :param scenario: The scenario name (e.g. "s_-3000..-100..400___...")
    :param bucket_name: The S3 bucket containing the archives (default: "mochi-trade-extracts")
    """
    s3_client = boto3.client("s3")

    # Example object key: <symbol>/<scenario>.zip
    object_key = f"{symbol}/{scenario}.zip"
    local_zip_file = f"{scenario}.zip"

    print(f"Downloading s3://{bucket_name}/{object_key} ...")
    s3_client.download_file(bucket_name, object_key, local_zip_file)

    destination_folder = scenario
    print(f"Unzipping {local_zip_file} to {destination_folder} ...")
    with zipfile.ZipFile(local_zip_file, "r") as zf:
        zf.extractall(destination_folder)

    # (Optionally) clean up by removing the ZIP file:
    # os.remove(local_zip_file)

    print("Done downloading and unzipping.")
