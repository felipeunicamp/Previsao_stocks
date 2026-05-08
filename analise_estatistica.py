import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# from curl_cffi import requests  # <-- Código antigo comentado
import requests  # <-- Novo import
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import numpy as np


def gerar_analise(ticker: str, periodo_anos: int):
    """
    Gera todos os gráficos da análise estatística e retorna uma lista de HTMLs.

    Args:
        ticker: código da ação (ex: "PETR4")
        periodo_anos: quantidade de anos (ex: 1, 3, 5)

    Returns:
        dict com os HTMLs de cada gráfico
    """
    # session = requests.Session(impersonate="chrome")  # <-- Código antigo comentado

    
    stock = yf.Ticker(f'{ticker.upper()}.SA')
    data = stock.history(period=f'{periodo_anos}y')

    if data.empty:
        return {"erro": f"Dados não encontrados para {ticker}"}

    data2 = pd.DataFrame(data)
    data2 = data2[['Close']]

    # Tratamento
    data2 = data2.reset_index()
    data2['Date'] = pd.to_datetime(data2['Date'])
    data2 = data2.rename(columns={'Date': 'ds', 'Close': 'y'})
    data2.set_index('ds', inplace=True)
    data2['y'] = data2['y'].interpolate(method='linear', limit_direction='both')

    resultados = {}

    # ―― 1. Cotação histórica ――
    cotacao = px.line(
        data2.reset_index(), x='ds', y='y',
        title=f'\ud83d\udcc8 Cotação {ticker.upper()} — Período {periodo_anos} ano(s)',
        labels={'ds': 'Período', 'y': 'Preço (R$)'}
    )
    cotacao.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        hovermode='x unified',
        height=500
    )
    resultados['cotacao'] = pio.to_html(cotacao, include_plotlyjs='cdn', full_html=False)

    # ―― 2. Decomposição Sazonal ――
    decompose = seasonal_decompose(data2['y'], model='multiplicative', period=365)

    fig_dec = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Série Original', 'Tendência', 'Sazonal', 'Resíduo'),
        vertical_spacing=0.08
    )
    fig_dec.add_trace(go.Scatter(x=data2.index, y=data2['y'], name='Original',
                                 line=dict(color='#00ff88')), row=1, col=1)
    fig_dec.add_trace(go.Scatter(x=decompose.trend.index, y=decompose.trend, name='Tendência',
                                 line=dict(color='#ffd700')), row=2, col=1)
    fig_dec.add_trace(go.Scatter(x=decompose.seasonal.index, y=decompose.seasonal, name='Sazonal',
                                 line=dict(color='#00bfff')), row=3, col=1)
    fig_dec.add_trace(go.Scatter(x=decompose.resid.index, y=decompose.resid, name='Resíduo',
                                 line=dict(color='#ff4757')), row=4, col=1)
    fig_dec.update_layout(
        height=900,
        title_text=f'\ud83d\udd0d Decomposição Sazonal — {ticker.upper()} ({periodo_anos} ano(s))',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        showlegend=False
    )
    resultados['decomposicao'] = pio.to_html(fig_dec, include_plotlyjs='cdn', full_html=False)

    # ―― 3. Month Plot, Quarter Plot, ACF, PACF ――
    df_temp = pd.DataFrame({'value': data2['y'].values}, index=data2.index)
    df_temp['month'] = df_temp.index.month
    monthly_avg = df_temp.groupby('month')['value'].mean()
    months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
              'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

    df_temp['quarter'] = df_temp.index.quarter
    quarterly_avg = df_temp.groupby('quarter')['value'].mean()
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']

    acf_vals = acf(data2['y'], nlags=100, fft=True)
    pacf_vals = pacf(data2['y'], nlags=100)
    lags = np.arange(len(acf_vals))
    n = len(data2['y'])
    ci = 1.96 / np.sqrt(n)

    fig_ts = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Média por Mês', 'Média por Trimestre', 'ACF', 'PACF'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Month plot
    fig_ts.add_trace(
        go.Scatter(x=months, y=monthly_avg.values, mode='lines+markers',
                   name='Monthly', line=dict(color='#00ff88', width=2),
                   marker=dict(size=8, color='#00ff88')),
        row=1, col=1
    )
    # Quarter plot
    fig_ts.add_trace(
        go.Scatter(x=quarters, y=quarterly_avg.values, mode='lines+markers',
                   name='Quarterly', line=dict(color='#ffd700', width=2),
                   marker=dict(size=8, color='#ffd700')),
        row=1, col=2
    )
    # ACF
    fig_ts.add_trace(
        go.Scatter(x=lags, y=acf_vals, mode='markers',
                   marker=dict(color='#00bfff', size=6), name='ACF'),
        row=2, col=1
    )
    for lag in lags:
        fig_ts.add_trace(
            go.Scatter(x=[lag, lag], y=[0, acf_vals[lag]], mode='lines',
                       line=dict(color='#00bfff', width=1), showlegend=False,
                       hoverinfo='skip'),
            row=2, col=1
        )
    fig_ts.add_hrect(y0=-ci, y1=ci, fillcolor='lightblue', opacity=0.2,
                     line_width=0, row=2, col=1)
    fig_ts.add_hline(y=0, line_dash='dash', line_color='white', row=2, col=1)

    # PACF
    fig_ts.add_trace(
        go.Scatter(x=lags, y=pacf_vals, mode='markers',
                   marker=dict(color='#ffd700', size=6), name='PACF'),
        row=2, col=2
    )
    for lag in lags:
        fig_ts.add_trace(
            go.Scatter(x=[lag, lag], y=[0, pacf_vals[lag]], mode='lines',
                       line=dict(color='#ffd700', width=1), showlegend=False,
                       hoverinfo='skip'),
            row=2, col=2
        )
    fig_ts.add_hrect(y0=-ci, y1=ci, fillcolor='lightblue', opacity=0.2,
                     line_width=0, row=2, col=2)
    fig_ts.add_hline(y=0, line_dash='dash', line_color='white', row=2, col=2)

    fig_ts.update_layout(
        height=800,
        title_text=f'\ud83d\udcca Análise de Série Temporal — {ticker.upper()}',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        showlegend=False
    )
    fig_ts.update_xaxes(title_text='Mês', row=1, col=1)
    fig_ts.update_xaxes(title_text='Trimestre', row=1, col=2)
    fig_ts.update_xaxes(title_text='Lag', row=2, col=1)
    fig_ts.update_xaxes(title_text='Lag', row=2, col=2)
    fig_ts.update_yaxes(title_text='Preço (Média)', row=1, col=1)
    fig_ts.update_yaxes(title_text='Preço (Média)', row=1, col=2)
    fig_ts.update_yaxes(title_text='ACF', row=2, col=1)
    fig_ts.update_yaxes(title_text='PACF', row=2, col=2)

    resultados['serie_temporal'] = pio.to_html(fig_ts, include_plotlyjs='cdn', full_html=False)

    return resultados
