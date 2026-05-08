from flask import Flask, render_template, request, redirect, url_for
from analise_estatistica import gerar_analise
from previsao import gerar_previsao, previsao_futuro
from previsao import gerar_previsao_exponencial

app = Flask(__name__)

@app.route('/')
def home():
    return redirect(url_for('inicio'))

@app.route('/inicio')
def inicio():
    return render_template('inicio.html')

@app.route('/analise', methods=['POST'])
def analise():
    ticker = request.form.get('ticker')
    periodo = request.form.get('periodo')

    resultados = gerar_analise(ticker, int(periodo))

    if "erro" in resultados:
        return render_template('analise-estatistica.html',
                               ticker=ticker.upper(),
                               periodo=periodo,
                               erro=resultados["erro"])

    return render_template('analise-estatistica.html',
                           ticker=ticker.upper(),
                           periodo=periodo,
                           grafico_cotacao=resultados['cotacao'],
                           grafico_decomposicao=resultados['decomposicao'],
                           grafico_serie=resultados['serie_temporal'])

@app.route('/previsao', methods=['POST'])
def previsao():
    ticker = request.form.get('ticker')
    periodo = request.form.get('periodo')
    resultado = gerar_previsao(ticker, int(periodo))

    if 'erro' in resultado:
        return render_template('previsao.html', erro=resultado['erro'])

    futuro = previsao_futuro(ticker, int(periodo))

    return render_template('previsao.html',
                           ticker=ticker.upper(),
                           periodo=periodo,
                           mape=resultado['mape'],
                           modelo=resultado['modelo'],
                           grafico=resultado['grafico'],
                           futuro=futuro)

@app.route('/previsao_exponencial', methods=['POST'])
def previsao_exponencial():
    ticker = request.form.get('ticker')
    periodo = request.form.get('periodo')
    resultado = gerar_previsao_exponencial(ticker, int(periodo))

    if 'erro' in resultado:
        return render_template('previsao_exponencial.html', erro=resultado['erro'])

    return render_template('previsao_exponencial.html',
                           ticker=ticker.upper(),
                           periodo=periodo,
                           mape=resultado['mape'],
                           grafico=resultado['grafico'],
                           futuro=resultado['previsoes'],
                           ultimo_preco=resultado['ultimo_preco'],
                           ultima_data=resultado['ultima_data'])

if __name__ == '__main__':
    app.run(debug=True)