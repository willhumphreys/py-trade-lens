import os
import time
import logging
import matplotlib

from src.processing.data_reader import read_trader_profit_csv

matplotlib.use('Agg')  # Ensure non-interactive backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import gc
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_output_directory(path):
    os.makedirs(path, exist_ok=True)

def extract_trader_info(file_name):
    parts = file_name.split('.')
    trader_id = parts[0]
    return trader_id

def load_csv_data(file_path):
    return read_trader_profit_csv(file_path)

def create_dataframe(data):
    return pd.DataFrame(data, columns=['PlaceDateTime', 'FilledPrice', 'ClosingPrice', 'Profit', 'RunningTotalProfit', 'State'])

def add_profit_marker(df):
    df['ProfitMarker'] = df['Profit'] > 0
    return df

def plot_data(df, trader_id, output_path, tick_data):
    fig, ax1 = plt.subplots(figsize=(70, 14))

    # Plot profit points
    ax1.scatter(df[df['ProfitMarker']]['PlaceDateTime'],
                df[df['ProfitMarker']]['ClosingPrice'],
                color='green', label='Profit', marker='o')

    # Plot loss points
    ax1.scatter(df[~df['ProfitMarker']]['PlaceDateTime'],
                df[~df['ProfitMarker']]['ClosingPrice'],
                color='red', label='Loss', marker='x')

    # Plot close price from tick_data
    if tick_data is not None:
        # Filter out invalid data points (where close is -1)
        valid_tick_data = tick_data[tick_data['close'] != -1]

        # Convert datetime string to datetime object if needed
        if isinstance(valid_tick_data['dateTime'].iloc[0], str):
            valid_tick_data['dateTime'] = pd.to_datetime(valid_tick_data['dateTime'])

        # Sort by datetime to ensure line is drawn correctly
        valid_tick_data = valid_tick_data.sort_values(by='dateTime')

        # Plot the close price line
        ax1.plot(valid_tick_data['dateTime'],
                 valid_tick_data['close'],
                 color='purple',
                 label='Close Price',
                 linewidth=1)

    # Format x-axis to show months
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=90)

    # Create secondary y-axis for running total profit
    ax2 = ax1.twinx()
    ax2.plot(df['PlaceDateTime'], df['RunningTotalProfit'], color='blue', label='Running Total Profit', linewidth=2)

    # Set labels and title
    ax1.set_xlabel('Placed Date Time')
    ax1.set_ylabel('Closing Price')
    ax2.set_ylabel('Running Total Profit')
    plt.title('Closing Price and Running Total Profit Over Time with Profit and Loss Points')

    # Update legend to include the new line
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True)

    # Save and close figure
    output_file_path = os.path.join(output_path, f'trades-and-profit-{trader_id}.png')
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')

def process_file(file_name, input_dir):
    trader_id = extract_trader_info(file_name)
    file_path = os.path.join(input_dir, file_name)
    data = load_csv_data(file_path)
    df = create_dataframe(data)
    df = add_profit_marker(df)
    return df, trader_id

def process_and_plot_file(file_name, input_dir, output_dir, tick_data):
    start_time = time.time()
    df, trader_id = process_file(file_name, input_dir)
    plot_data(df, trader_id, output_dir, tick_data)
    # Cleanup
    del df
    gc.collect()
    elapsed = time.time() - start_time
    logging.info(f"Processed {file_name} in {elapsed:.2f} seconds")
    return file_name

def process_and_plot_files(input_dir, output_dir, tick_data):
    create_output_directory(output_dir)
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for file_name in files:
            futures.append(executor.submit(process_and_plot_file, file_name, input_dir, output_dir, tick_data))
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                logging.info(f"Finished processing {result}")
            except Exception as exc:
                logging.error(f"File processing generated an exception: {exc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for the given symbol and scenario")
    parser.add_argument('--symbol', required=True, help='Symbol for the currency pair, e.g., gbpusd-1m-btmH')
    args = parser.parse_args()
    base_scenario_dir = f'output6/{args.symbol}'
    for scenario_dir in os.listdir(base_scenario_dir):
        scenario_path = os.path.join(base_scenario_dir, scenario_dir)
        if os.path.isdir(scenario_path):
            input_dir = os.path.join(scenario_path, 'profits')
            graph_output_dir = os.path.join(scenario_path, 'coloured')
            process_and_plot_files(input_dir, graph_output_dir, valid_data)
            gc.collect()
    logging.info("All files processed and plots created.")
