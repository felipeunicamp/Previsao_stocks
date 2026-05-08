import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
import plotly.io as pio
import pandas as pd
import yfinance as yf
# --- CÓDIGO ANTIGO COMENTADO ---
# from curl_cffi import requests
# -------------------------------
# --- NOVO CÓDIGO ---
import requests
# -------------------
from datetime import timedelta

torch.manual_seed(42)
np.random.seed(42)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=80, num_layers=2, output_size=15):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def gerar_previsao(ticker: str, periodo_anos: int, salvar_modelo: bool = True):
    """
    Treina LSTM e retorna gráfico + métricas.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # −− Dados via yfinance −−
    # --- CÓDIGO ANTIGO COMENTADO ---
    # session = requests.Session(impersonate="chrome")
    # -------------------------------
    # --- NOVO CÓDIGO ---
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    })
    # -------------------
    stock = yf.Ticker(f'{ticker.upper()}.SA', session=session)
    data = stock.history(period=f'{periodo_anos}y')

    if data.empty:
        return {"erro": f"Dados não encontrados para {ticker}"}

    df = data.reset_index()[['Date', 'Close']]

    # −− Separar treino/teste (80/20) −−
    split = int(0.8 * len(df))
    df_train, df_teste = df.iloc[:split], df.iloc[split:]
    y_train, y_test = df_train['Close'].values, df_teste['Close'].values

    # −− Normalizar −−
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(y_train.reshape(-1, 1))
    y_train_s = scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_test_s = scaler.transform(y_test.reshape(-1, 1)).flatten()

    # −− Sequências LSTM −−
    lookback, horizonte = 60, 15
    X_tr, y_tr = [], []
    for i in range(len(y_train_s) - lookback - horizonte + 1):
        X_tr.append(y_train_s[i:i + lookback])
        y_tr.append(y_train_s[i + lookback:i + lookback + horizonte])
    X_te, y_te = [], []
    for i in range(len(y_test_s) - lookback - horizonte + 1):
        X_te.append(y_test_s[i:i + lookback])
        y_te.append(y_test_s[i + lookback:i + lookback + horizonte])

    X_train = torch.FloatTensor(np.array(X_tr)).reshape(-1, 60, 1)
    y_train = torch.FloatTensor(np.array(y_tr))
    X_test = torch.FloatTensor(np.array(X_te)).reshape(-1, 60, 1)
    y_test_t = torch.FloatTensor(np.array(y_te))

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    model = LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        loss_avg = 0
        for Xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            opt.step()
            loss_avg += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f'  Época {epoch+1}/200 — Loss: {loss_avg/len(loader):.6f}')

    # −− Predição −−
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()

    y_train_real = scaler.inverse_transform(y_train.numpy().reshape(-1, 1)).reshape(-1, 15)
    y_test_real = scaler.inverse_transform(y_test_t.numpy().reshape(-1, 1)).reshape(-1, 15)
    y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1, 15)

    mape = mean_absolute_percentage_error(y_test_real, y_pred_real)

    # −− Salvar modelo −−
    caminho = None
    if salvar_modelo:
        nome = f'modelo_lstm_{ticker.lower()}.pth'
        caminho = os.path.join(os.getcwd(), nome)
        torch.save(model.state_dict(), caminho)

    # −− Montar DataFrame para o gráfico −−
    train_end = split - horizonte + 1
    test_start = split + lookback

    df_plot = pd.concat([
        pd.DataFrame({'Date': df['Date'].iloc[lookback:train_end],
                      'Price': y_train_real[:, 0],
                      'Tipo': 'Treino'}),
        pd.DataFrame({'Date': df['Date'].iloc[split:split + lookback],
                      'Price': df['Close'].iloc[split:split + lookback].values,
                      'Tipo': 'Real (Teste)'}),
        pd.DataFrame({'Date': df['Date'].iloc[test_start:test_start + len(y_pred_real)],
                      'Price': y_test_real[:, 0],
                      'Tipo': 'Real (Teste)'}),
        pd.DataFrame({'Date': df['Date'].iloc[test_start:test_start + len(y_pred_real)],
                      'Price': y_pred_real[:, 0],
                      'Tipo': 'Previsto'}),
    ], ignore_index=True)

    # −− Gráfico Plotly −−
    fig = px.line(df_plot, x='Date', y='Price', color='Tipo',
                  title=f'🪀 LSTM — Previsão {ticker.upper()} ({periodo_anos} ano(s)) | MAPE: {mape:.2%}',
                  labels={'Price': 'Preço (R$)', 'Date': 'Data', 'Tipo': 'Tipo'})
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'),
                      hovermode='x unified', height=600)

    return {"grafico": pio.to_html(fig, include_plotlyjs='cdn', full_html=False),
            "mape": mape, "modelo": caminho}


def previsao_futuro(ticker: str, periodo_anos: int = 5):
    """
    Carrega modelo LSTM salvo e prevê os próximos 15 dias.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # --- CÓDIGO ANTIGO COMENTADO ---
    # session = requests.Session(impersonate="chrome")
    # -------------------------------
    # --- NOVO CÓDIGO ---
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    })
    # -------------------
    stock = yf.Ticker(f'{ticker.upper()}.SA', session=session)
    data = stock.history(period=f'{periodo_anos}y')

    if data.empty:
        return {"erro": f"Dados não encontrados para {ticker}"}

    close_prices = data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_prices)

    seq_len = 60
    scaled_data = scaler.transform(close_prices)
    last_sequence = scaled_data[-seq_len:].reshape(1, seq_len, 1)

    # −− Carregar modelo salvo −−
    caminho_modelo = f'modelo_lstm_{ticker.lower()}.pth'
    try:
        model = LSTM(input_size=1, hidden_size=80, num_layers=2, output_size=15)  # ← agora funciona!
        model.load_state_dict(torch.load(caminho_modelo, map_location='cpu'))
        model.eval()
    except FileNotFoundError:
        return {"erro": f"Modelo não encontrado. Execute o treinamento primeiro: {caminho_modelo}"}

    seq_tensor = torch.FloatTensor(last_sequence)
    with torch.no_grad():
        pred = model(seq_tensor)

    predictions = pred.cpu().detach().numpy()[0]
    valores_previstos = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    from datetime import datetime, timedelta
    ultima_data = data.index[-1]
    dias_uteis = []
    data_atual = ultima_data + timedelta(days=1)
    while len(dias_uteis) < 15:
        if data_atual.weekday() < 5:
            dias_uteis.append(data_atual)
        data_atual += timedelta(days=1)

    previsoes = []
    for i in range(15):
        previsoes.append({
            "dia": i + 1,
            "data": dias_uteis[i].strftime('%d/%m/%Y'),
            "valor": round(float(valores_previstos[i]), 2)
        })

    return {
        "ticker": ticker.upper(),
        "ultimo_preco": round(float(close_prices[-1][0]), 2),
        "ultima_data": ultima_data.strftime('%d/%m/%Y'),
        "previsoes": previsoes
    }

def gerar_previsao_exponencial(ticker: str, periodo_anos: int):
    # --- CÓDIGO ANTIGO COMENTADO ---
    # session = requests.Session(impersonate="chrome")
    # -------------------------------
    # --- NOVO CÓDIGO ---
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    })
    # -------------------
    stock = yf.Ticker(f'{ticker.upper()}.SA', session=session)
    data = stock.history(period=f'{periodo_anos}y')

    if data.empty:
        return {"erro": f"Dados não encontrados para {ticker}"}

    data2 = pd.DataFrame(data)[['Close']].reset_index()
    data2['Date'] = pd.to_datetime(data2['Date'])
    data2 = data2.rename(columns={'Date': 'ds', 'Close': 'y'})
    data2.set_index('ds', inplace=True)
    data2['y'] = data2['y'].interpolate(method='linear', limit_direction='both')

    # Backtest dinâmico (últimos 30 dias)
    split = 30
    if len(data2) <= split * 2:
        return {"erro": "Dados insuficientes para backtest de 30 dias."}

    df_pratica = data2.iloc[:-split]
    df_real = data2.iloc[-split:]

    # Modelo Backtest
    model_backtest = ExponentialSmoothing(
        df_pratica['y'], trend='additive', seasonal='multiplicative',
        seasonal_periods=min(252, len(df_pratica)//2), damped_trend=False
    ).fit()
    previsao_backtest = model_backtest.forecast(len(df_real))
    previsao_backtest.index = df_real.index

    mape = mean_absolute_percentage_error(df_real['y'], previsao_backtest)

    # Modelo Real (Previsão Futura 30 dias)
    model_real = ExponentialSmoothing(
        data2['y'], trend='additive', seasonal='multiplicative',
        seasonal_periods=min(252, len(data2)//2), damped_trend=False
    ).fit()
    previsao_futura = model_real.forecast(30)

    # DataFrame para o Gráfico
    df_plot_treino = pd.DataFrame({'data': df_pratica.index, 'preco': df_pratica['y'], 'serie': 'Treino'})
    df_plot_real = pd.DataFrame({'data': df_real.index, 'preco': df_real['y'], 'serie': 'Real'})
    df_plot_previsao = pd.DataFrame({'data': previsao_backtest.index, 'preco': previsao_backtest, 'serie': 'Previsão'})

    df_plot = pd.concat([df_plot_treino, df_plot_real, df_plot_previsao], ignore_index=True)

    # Gráfico Plotly Express (SEM plotly.go)
    fig = px.line(df_plot, x='data', y='preco', color='serie',
                  color_discrete_map={'Treino': 'lightblue', 'Real': 'green', 'Previsão': 'orange'},
                  title=f'📈 Suavização Exponencial — Backtest {ticker.upper()} | MAPE: {mape:.2%}',
                  labels={'preco': 'Preço (R$)', 'data': 'Data', 'serie': 'Séries'})

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'),
        hovermode='x unified', height=600
    )

    cutoff = df_pratica.index[-1]
    fig.add_vline(x=cutoff, line_dash='dash', line_color='red', line_width=2, opacity=0.7)

    grafico_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

    # Formatar previsão futura para a tabela (dias úteis)
    dias_uteis = []
    data_atual = data2.index[-1] + timedelta(days=1)
    while len(dias_uteis) < 30:
        if data_atual.weekday() < 5:
            dias_uteis.append(data_atual)
        data_atual += timedelta(days=1)

    previsoes_lista = []
    for i in range(30):
        previsoes_lista.append({
            "dia": i + 1,
            "data": dias_uteis[i].strftime('%d/%m/%Y'),
            "valor": round(float(previsao_futura.iloc[i]), 2)
        })

    return {
        "grafico": grafico_html,
        "mape": mape,
        "previsoes": previsoes_lista,
        "ultimo_preco": round(float(data2['y'].iloc[-1]), 2),
        "ultima_data": data2.index[-1].strftime('%d/%m/%Y')
    }