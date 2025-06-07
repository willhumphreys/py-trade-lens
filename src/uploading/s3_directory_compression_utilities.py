import os
import zipfile
import tempfile
import boto3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compress_directory_to_zip(source_dir: str, zip_file: str) -> None:
    """
    Compress the provided directory into a ZIP archive.

    :param source_dir: The source directory to compress.
    :param zip_file: The output ZIP file path.
    :raises Exception: When file operations fail.
    """
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Determine the relative path and use '/' as separator in the ZIP file.
                relative_path = os.path.relpath(file_path, source_dir).replace(os.sep, "/")
                zf.write(file_path, arcname=relative_path)


def compress_and_push_scenario_zip(
        scenario_dir: str,
        s3_key: str,
        bucket_name: str = "mochi-prod-trade-extracts",
        s3_client=None,
        back_test_id=None
) -> None:
    """
    Compresses the provided scenario directory into a ZIP file and uploads it to S3.

    :param scenario_dir: The local directory containing the scenario files.
    :param s3_key: The S3 key to use when uploading the ZIP archive.
    :param bucket_name: The S3 bucket to which the archive is uploaded.
    :param s3_client: Optional boto3 S3 client. If None, a new client is created.
    :param back_test_id: The back test ID to prefix S3 keys with.
    """
    if not os.path.isdir(scenario_dir):
        logger.warning("Scenario directory does not exist or is not a directory: %s", scenario_dir)
        return

    if s3_client is None:
        s3_client = boto3.client("s3")

    temp_zip = None
    try:
        # Create a temporary ZIP file
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            temp_zip = tmp.name

        # Compress the directory to the temporary ZIP file.
        compress_directory_to_zip(scenario_dir, temp_zip)
        logger.info("Successfully compressed '%s' into temporary ZIP: %s", scenario_dir, temp_zip)

        # Prefix the S3 key with back_test_id if provided
        final_s3_key = f"{back_test_id}/{s3_key}" if back_test_id else s3_key

        # Upload the ZIP file to S3.
        s3_client.upload_file(temp_zip, bucket_name, final_s3_key)
        logger.info("Uploaded compressed ZIP as key '%s' to bucket '%s'", final_s3_key, bucket_name)
    except Exception as e:
        logger.error("Error during ZIP compression/upload for directory '%s': %s", scenario_dir, str(e), exc_info=True)
    finally:
        # Clean up the temporary ZIP file.
        if temp_zip and os.path.exists(temp_zip):
            try:
                os.remove(temp_zip)
            except Exception as e:
                logger.warning("Could not delete temporary ZIP file '%s': %s", temp_zip, str(e))


def compress_and_push_all_scenarios(
        symbol_dir: str,
        bucket_name: str,
        s3_client: boto3.client,
        back_test_id: str
) -> None:
    """
    Loops through all scenario directories within the symbol directory, compressing and uploading each.
    The S3 key for each ZIP is in the format "symbol/scenarioName.zip" or "back_test_id/symbol/scenarioName.zip" if back_test_id is provided.

    :param symbol_dir: The parent directory containing scenario subdirectories.
    :param bucket_name: The S3 bucket to which the archive is uploaded.
    :param s3_client: Optional boto3 S3 client.
    :param back_test_id: The back test ID to prefix S3 keys with.
    """

    if not os.path.isdir(symbol_dir):
        logger.warning("Symbol directory does not exist or is not a directory: %s", symbol_dir)
        raise ValueError("Symbol directory does not exist or is not a directory")

    symbol = os.path.basename(symbol_dir)


    for scenario in os.listdir(symbol_dir):
        scenario_path = os.path.join(symbol_dir, scenario)
        if os.path.isdir(scenario_path):
            # Create an S3 key that mirrors the directory structure: symbol/scenarioName.zip
            s3_key = f"{back_test_id}/{symbol}/{scenario}.zip"
            compress_and_push_scenario_zip(scenario_path, s3_key, bucket_name, s3_client, back_test_id)
