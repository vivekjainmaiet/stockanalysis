import requests
import pandas as pd

class TickerTape():

    def __init__(self, ticker):
        # input ticker sample want to obtain news for
        self.ticker = ticker

    def get_yearly_finance(self):
        stock = self.ticker
        url = f"https://api.tickertape.in/stocks/financials/income/{self.ticker}/annual/normal"
        response = requests.get(url).json()
        data = pd.DataFrame(response['data'])
        df_financial = data[['displayPeriod','incTrev','incEbi','incPbi','incPbt','incNinc','incEps','incDps','incPyr']]

        df_financial = df_financial.rename(
            columns={
                "displayPeriod": "Financial_Year",
                "incTrev": "Total_Revenue",
                "incEbi": "EBITDA",
                "incPbi": "PBIT",
                "incPbt": "PBT",
                "incNinc": "Net_Income",
                "incEps": "EPS",
                "incDps": "DPS",
                "incPyr": "Payout_ratio"
            })
        return df_financial


if __name__ == "__main__":
    tickertape = TickerTape('ITC')
    df = tickertape.get_yearly_finance()
    print(df.columns)
