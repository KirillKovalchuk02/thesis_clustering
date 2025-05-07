
# from functions import get_tickers_stocks, get_close_prices, double_listed_stocks
# import pandas as pd




# #GET THE STOCKS

# us_exchanges = ['NMS', 'NYQ', 'NGM']
# eu_exchanges = ['PAR', 'FRA', 'LSE', 'AMS']
# asia_exchanges = ['SHH', 'JPX', 'HKG']

# selected_exchanges = us_exchanges + eu_exchanges + asia_exchanges

# full_selected_stocks = {}
# df_all_stocks = pd.DataFrame()
# for exchange in selected_exchanges:
#     print(f'Extracting from {exchange}')
#     exchanges = [exchange]
#     selected_stocks_dict, ticker_list = get_tickers_stocks(50000, exchanges, 20)

#     full_selected_stocks.update(selected_stocks_dict)

#     if len(ticker_list) > 0: 
#         print('YES')
#         df = get_close_prices(ticker_list, period = 1, start = '2024-01-01')
#         df_all_stocks = pd.concat([df_all_stocks, df], axis=1)

# doubly_listed_tickers = double_listed_stocks(full_selected_stocks)

# for ticker_to_drop in doubly_listed_tickers:
#     try:
#         df_all_stocks = df_all_stocks.drop(columns=[ticker_to_drop])
#     except:
#         pass

# df_all_stocks = df_all_stocks.ffill() #ffill again after concatenating the tickers


# df_all_stocks.to_csv('stocks_data_out_sample.csv', index=True)





import requests
import pandas as pd
import time
from datetime import datetime
import pytz

# Your free Finnhub API key
api_key = 'd0dovj9r01qv1dmisbg0d0dovj9r01qv1dmisbgg'

# Function to get daily closing prices from Finnhub
def get_finnhub_closing_prices(tickers, start_date='2024-01-01', end_date='2024-12-31'):
    base_url = 'https://finnhub.io/api/v1/stock/candle'
    start_timestamp = int(pd.Timestamp(start_date, tz='UTC').timestamp())
    end_timestamp = int(pd.Timestamp(end_date, tz='UTC').timestamp())

    all_data = {}

    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            params = {
                'symbol': ticker,
                'resolution': 'D',  # Daily frequency
                'from': start_timestamp,
                'to': end_timestamp,
                'token': api_key
            }
            response = requests.get(base_url, params=params)
            data = response.json()

            #if data.get('s') != 'ok':
            #    print(f"Failed to fetch data for {ticker}: {data.get('s')}")
            #    continue

            # Create DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(data['t'], unit='s'),
                'close': data['c']
            }).set_index('date')

            all_data[ticker] = df['close']

            # Optional: short sleep to be respectful, but not required for free tier
            time.sleep(0.1)

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return pd.DataFrame(all_data).sort_index()

# Example usage
tickers = list(pd.read_csv('stocks_data_filled.csv'))
df = get_finnhub_closing_prices(tickers)
#print(df.head())


df.to_csv('out_sample_stocks_2024.csv')




#GETTING CRYPTOS FROM COINBASE 50 INDEX
#https://www.marketvector.com/factsheets/download/COIN50.d.pdf


# from binance.client import Client
# import pandas as pd
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# import time


# def get_binance_close_prices(ticker_list, period=1, start='2024-01-01', interval='1d'):
#     client = Client()

#     start_dt = datetime.strptime(start, '%Y-%m-%d')
#     end_dt = start_dt + relativedelta(years=period)
#     start_str = start_dt.strftime('%d %b %Y')
#     end_str = end_dt.strftime('%d %b %Y')

#     all_closes = {}

#     for ticker in ticker_list:
#         print(f"Fetching data for: {ticker}")
        
#         try:
#             klines = client.get_historical_klines(
#                 ticker, interval, start_str, end_str
#             )

#             if not klines:
#                 continue

#             df = pd.DataFrame(klines, columns=[
#                 'timestamp', 'open', 'high', 'low', 'close', 'volume',
#                 'close_time', 'quote_asset_volume', 'number_of_trades',
#                 'taker_buy_base', 'taker_buy_quote', 'ignore'
#             ])

#             # Convert timestamp and set as index
#             df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#             df.set_index('timestamp', inplace=True)

#             # Extract close prices
#             df = df[['close']].astype(float)
            


#             all_closes[ticker] = df['close']

#             # Sleep to respect rate limits
#             time.sleep(0.1) 

#         except Exception as e:
#             print(f"Error fetching data for {ticker}: {e}")

#     if not all_closes:
#         print("No data fetched for any tickers. Please check the tickers and date range.")
#         return pd.DataFrame()

#     # Combine data into a single DataFrame
#     df_combined = pd.concat(all_closes, axis=1)

#     return df_combined


# coinbase_50_cryptos = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE', 'ADA', 'LINK', 'XLM', 'AVAX', 'SHIB', 'DOT', 'LTC', 'BCH', 
#                        'UNI', 'NEAR', 'PEPE', 'APT', 'ICP', 'ETC', 'AAVE', 'RNDR', 'ATOM', 'MATIC', 'ALGO', 'EOS', 'MKR', 
#                        'ASI', 'QNT', 'BONK', 'STX', 'INJ', 'GRT', 'LDO', 'XTZ', 'CRV', 'SAND', 'ZEC', 'HNT', 'JASMY', 'MANA', 
#                        'AXS', 'WIF', 'CHZ', 'COMP', 'APE', 'AERO', '1INCH', 'SNX', 'ROSE', 'LPT']

# binance_tickers = [f"{ticker}USDT" for ticker in coinbase_50_cryptos]

# df = get_binance_close_prices(binance_tickers, start='2022-01-01', period=2)
# df = df.drop(columns=['HNTUSDT', 'PEPEUSDT', 'APTUSDT', 'BONKUSDT', 'LDOUSDT', 'APEUSDT']).dropna()

# df.to_csv('cryptos_data_out_sample.csv', index=True)