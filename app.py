import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.write("""
<style>
h1 { margin-top: -5rem; }
</style>
""", unsafe_allow_html=True)

st.title('Dự đoán Thị trường Chứng khoán')

def main():
    option = st.sidebar.selectbox('Lựa chọn', ['Trực quan hoá','Dữ liệu mới nhất', 'Dự đoán'])
    if option == 'Trực quan hoá':
        tech_indicators()
    elif option == 'Dữ liệu mới nhất':
        dataframe()
    else:
        predict()



@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



option = st.sidebar.text_input('Nhập một ký hiệu cổ phiếu', value='VFS')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Nhập khoảng thời gian', value=255)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Ngày bắt đầu', value=before)
end_date = st.sidebar.date_input('Ngày kết thúc', today)
if st.sidebar.button('Xác nhận'):
    if start_date < end_date:
        st.sidebar.success('Ngày bắt đầu: `%s`\n\nNgày kết thúc: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Lỗi: Ngày kết thúc phải sau ngày bắt đầu')




data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    st.subheader('Chỉ số Kỹ thuật')
    option = st.radio('Chọn 1 chỉ số kỹ thuật để trực quan hoá', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Expoenetial Moving Average')
        st.line_chart(ema)


def dataframe():
    st.header('Dữ liệu mới nhất')
    st.dataframe(data.tail(10))



def predict():
    model = st.radio('Chọn mô hình', ['LinearRegression', 'ExtraTreesRegressor', 'KNeighborsRegressor'])
    num = st.number_input('Số ngày cần dự đoán?', value=3)
    num = int(num)
    if st.button('Dự đoán'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)



def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Ngày {day}: {i}')
        day += 1


if __name__ == '__main__':
    main()
