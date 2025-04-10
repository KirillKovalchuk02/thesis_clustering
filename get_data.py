
from functions import get_tickers_stocks, get_close_prices, double_listed_stocks
import pandas as pd




#GET THE STOCKS

us_exchanges = ['NMS', 'NYQ', 'NGM']
eu_exchanges = ['PAR', 'FRA', 'LSE', 'AMS']
asia_exchanges = ['SHH', 'JPX', 'HKG']

selected_exchanges = us_exchanges + eu_exchanges + asia_exchanges

full_selected_stocks = {}
df_all_stocks = pd.DataFrame()
for exchange in selected_exchanges:
    print(f'Extracting from {exchange}')
    exchanges = [exchange]
    selected_stocks_dict, ticker_list = get_tickers_stocks(50000, exchanges, 20)

    full_selected_stocks.update(selected_stocks_dict)

    if len(ticker_list) > 0: 
        print('YES')
        df = get_close_prices(ticker_list, period = 2, start = '2022-01-01')
        df_all_stocks = pd.concat([df_all_stocks, df], axis=1)

doubly_listed_tickers = double_listed_stocks(full_selected_stocks)

for ticker_to_drop in doubly_listed_tickers:
    try:
        df_all_stocks = df_all_stocks.drop(columns=[ticker_to_drop])
    except:
        pass

df_all_stocks = df_all_stocks.ffill() #ffill again after concatenating the tickers


df_all_stocks.to_csv('stocks_data.csv', index=False)



#GETTING CRYPTOS FROM COINBASE 50 INDEX
#https://www.marketvector.com/factsheets/download/COIN50.d.pdf

coinbase_50_cryptos = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE', 'ADA', 'LINK', 'XLM', 'AVAX', 'SHIB', 'DOT', 'LTC', 'BCH', 
                       'UNI', 'NEAR', 'PEPE', 'APT', 'ICP', 'ETC', 'AAVE', 'RNDR', 'ATOM', 'MATIC', 'ALGO', 'EOS', 'MKR', 
                       'ASI', 'QNT', 'BONK', 'STX', 'INJ', 'GRT', 'LDO', 'XTZ', 'CRV', 'SAND', 'ZEC', 'HNT', 'JASMY', 'MANA', 
                       'AXS', 'WIF', 'CHZ', 'COMP', 'APE', 'AERO', '1INCH', 'SNX', 'ROSE', 'LPT']

crypto_tickers_fixed = [tick + "-USD" for tick in coinbase_50_cryptos]

cryptos_df = get_close_prices(crypto_tickers_fixed, period = 2, start = '2022-01-01')

print('Number of NA values in cryptos_df is ', df.isna().any().sum())

cryptos_df.to_csv('cryptos_data.csv', index=False)