"""
Scanner de Tendência x Lateralização - Forex M1
=================================================

Escaneia uma lista de pares no MetaTrader5, em M1, dentro de uma janela
de horário/dia, e classifica cada par matematicamente como:

    - "TENDÊNCIA DEFINIDA"  -> bom pra seguir fluxo de vela
    - "LATERALIZADO"        -> evitar, alta chance de martingale/erro

Métricas usadas (todas calculadas sobre os preços de fechamento M1):

1. Efficiency Ratio (Kaufman)
   ER = |close[-1] - close[0]| / soma(|close[i] - close[i-1]|)
   -> 1 = andou reto (tendência). 0 = andou de um lado pro outro (lateral).

2. ADX (Average Directional Index) clássico (Wilder, período 14)
   -> mede força da tendência, direção não importa.

3. Regressão linear sobre os closes -> slope (direção) e R² (consistência)

4. % de alternância de cor das velas (mesma lógica do analisador de pavios)
   -> complementa o ADX/ER, pega lateralização "vela a vela".

Score final = combinação ponderada normalizada dessas 4 métricas.

Requisitos:
    pip install MetaTrader5 pandas numpy

Uso:
    python scanner_tendencia_forex.py
    (edite os parâmetros no bloco __main__ no final do arquivo)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Conexão com o MT5
# ---------------------------------------------------------------------------
def conectar_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"Falha ao inicializar MT5: {mt5.last_error()}")


def desconectar_mt5():
    mt5.shutdown()


# ---------------------------------------------------------------------------
# Coleta de dados
# ---------------------------------------------------------------------------
def obter_velas_m1(par: str, dia: str, hora_inicial: str, hora_final: str) -> pd.DataFrame:
    """
    dia: 'YYYY-MM-DD'
    hora_inicial / hora_final: 'HH:MM'
    """
    dt_inicio = datetime.strptime(f"{dia} {hora_inicial}", "%Y-%m-%d %H:%M")
    dt_fim = datetime.strptime(f"{dia} {hora_final}", "%Y-%m-%d %H:%M")

    rates = mt5.copy_rates_range(par, mt5.TIMEFRAME_M1, dt_inicio, dt_fim)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


# ---------------------------------------------------------------------------
# Métrica 1: Efficiency Ratio (Kaufman)
# ---------------------------------------------------------------------------
def efficiency_ratio(closes: np.ndarray) -> float:
    if len(closes) < 2:
        return 0.0
    deslocamento_liquido = abs(closes[-1] - closes[0])
    caminho_total = np.sum(np.abs(np.diff(closes)))
    if caminho_total == 0:
        return 0.0
    return deslocamento_liquido / caminho_total


# ---------------------------------------------------------------------------
# Métrica 2: ADX (Wilder, período 14) - implementação manual
# ---------------------------------------------------------------------------
def calcular_adx(df: pd.DataFrame, periodo: int = 14) -> float:
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    if len(df) < periodo * 2:
        return np.nan

    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    tr = np.zeros(len(df))

    for i in range(1, len(df)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # suavização de Wilder
    atr = pd.Series(tr).ewm(alpha=1 / periodo, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / periodo, adjust=False).mean().values / np.where(atr == 0, np.nan, atr)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / periodo, adjust=False).mean().values / np.where(atr == 0, np.nan, atr)

    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1 / periodo, adjust=False).mean().values

    return float(np.nanmean(adx[-periodo:]))  # média das últimas leituras


# ---------------------------------------------------------------------------
# Métrica 3: Regressão linear (slope + R²)
# ---------------------------------------------------------------------------
def regressao_linear(closes: np.ndarray) -> tuple[float, float]:
    x = np.arange(len(closes))
    if len(closes) < 2:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, closes, 1)
    pred = slope * x + intercept
    ss_res = np.sum((closes - pred) ** 2)
    ss_tot = np.sum((closes - np.mean(closes)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return float(slope), float(r2)


# ---------------------------------------------------------------------------
# Métrica 4: % de alternância de cor (lateralização "vela a vela")
# ---------------------------------------------------------------------------
def percentual_alternancia_cor(df: pd.DataFrame) -> float:
    cor = np.where(df["close"] > df["open"], 1, -1)  # 1 = alta, -1 = baixa
    if len(cor) < 2:
        return 0.0
    trocas = np.sum(cor[1:] != cor[:-1])
    return trocas / (len(cor) - 1)


# ---------------------------------------------------------------------------
# Classificação final
# ---------------------------------------------------------------------------
def classificar_par(df: pd.DataFrame) -> dict:
    closes = df["close"].values

    er = efficiency_ratio(closes)
    adx = calcular_adx(df)
    slope, r2 = regressao_linear(closes)
    alternancia = percentual_alternancia_cor(df)

    # normaliza ADX pra escala 0-1 (considerando 50 como teto prático)
    adx_norm = min(adx / 50, 1.0) if not np.isnan(adx) else 0.0

    # score combinado (pesos ajustáveis)
    # baixa alternância de cor = bom p/ tendência, por isso (1 - alternancia)
    score = (0.35 * er) + (0.35 * adx_norm) + (0.15 * r2) + (0.15 * (1 - alternancia))

    if score >= 0.6 and (adx >= 25 if not np.isnan(adx) else False):
        classificacao = "TENDÊNCIA DEFINIDA"
    elif score <= 0.35:
        classificacao = "LATERALIZADO"
    else:
        classificacao = "INDEFINIDO"

    return {
        "efficiency_ratio": round(er, 3),
        "adx": round(adx, 2) if not np.isnan(adx) else None,
        "r2_regressao": round(r2, 3),
        "direcao": "ALTA" if slope > 0 else "BAIXA",
        "pct_alternancia_cor": round(alternancia * 100, 1),
        "score": round(score, 3),
        "classificacao": classificacao,
        "n_velas": len(df),
    }


# ---------------------------------------------------------------------------
# Scanner principal
# ---------------------------------------------------------------------------
def escanear_pares(pares: list[str], dia: str, hora_inicial: str, hora_final: str) -> pd.DataFrame:
    conectar_mt5()
    resultados = []

    try:
        for par in pares:
            df = obter_velas_m1(par, dia, hora_inicial, hora_final)
            if df.empty:
                resultados.append({"par": par, "erro": "sem dados / símbolo indisponível"})
                continue

            info = classificar_par(df)
            info["par"] = par
            resultados.append(info)
    finally:
        desconectar_mt5()

    df_resultado = pd.DataFrame(resultados)
    if "score" in df_resultado.columns:
        df_resultado = df_resultado.sort_values("score", ascending=False)

    colunas_ordem = [
        "par", "classificacao", "score", "adx", "efficiency_ratio",
        "r2_regressao", "direcao", "pct_alternancia_cor", "n_velas",
    ]
    colunas_existentes = [c for c in colunas_ordem if c in df_resultado.columns]
    return df_resultado[colunas_existentes].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PARES = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        "USDCAD", "EURJPY", "GBPJPY", "CADJPY","GBPCAD","NZDUSD"
    ]

    DIA = datetime.now().strftime("%Y-%m-%d")
    HORA_INICIAL = "10:00"
    HORA_FINAL = "11:20"

    resultado = escanear_pares(PARES, DIA, HORA_INICIAL, HORA_FINAL)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    print(f"\nScan de tendência x lateralização - {DIA} {HORA_INICIAL} às {HORA_FINAL} (M1)\n")
    print(resultado.to_string(index=False))

    print("\nPares com TENDÊNCIA DEFINIDA:")
    definidos = resultado[resultado["classificacao"] == "TENDÊNCIA DEFINIDA"]
    print(definidos["par"].tolist() if not definidos.empty else "Nenhum no momento.")