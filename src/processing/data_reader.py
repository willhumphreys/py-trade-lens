import csv
from datetime import datetime

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def read_trader_profit_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header
        header_map = {name: index for index, name in enumerate(header)}

        for row in reader:
            date_str = row[header_map['PlaceDateTime']]
            if len(date_str) == 10:  # 'YYYY-MM-DD' has length 10
                date_str += ' 00:00:00'

            date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            filled_price = int(row[header_map['FilledPrice']])
            closing_price = int(row[header_map['ClosingPrice']])
            profit = int(row[header_map['Profit']])
            running_total_profit = int(row[header_map['RunningTotalProfit']])
            state = row[header_map['State']]
            data.append((date, filled_price, closing_price, profit, running_total_profit, state))
    return data
