import argparse
from extractor import download_and_unzip
from extractor_lzo import download_and_decompress_lzo

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download and unpack a trade archive from S3.")
    parser.add_argument(
        "--symbol",
        required=True,
        help="The symbol name (e.g. 'btc-1mF')"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="The scenario string (e.g. 's_-3000..-100..400___l_100..7500..400___...')"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    download_and_unzip(args.symbol, args.scenario)
    download_and_decompress_lzo(args.symbol, args.scenario)

if __name__ == "__main__":
    main()
