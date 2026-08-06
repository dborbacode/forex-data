"""
Analisador de Pavios (Wicks) em Forex via MetaTrader5
=======================================================

Escaneia um ou mais pares de moedas em um timeframe e intervalo de
data/hora escolhidos, e calcula a porcentagem de velas cujo maior
pavio (superior ou inferior) ultrapassa um percentual do range total
da vela.

Requisitos:
    - Terminal MetaTrader5 instalado e logado (conta demo funciona)
    - pip install MetaTrader5 pandas

Definição de "pavio grande" (parametrizável via --threshold):
    range_total   = high - low
    pavio_superior = high - max(open, close)
    pavio_inferior = min(open, close) - low
    maior_pavio    = max(pavio_superior, pavio_inferior)

    Uma vela é considerada "com pavio grande" quando:
        maior_pavio / range_total >= threshold (ex: 0.40 = 40%)

Exemplo de uso:
    python pavios_forex.py --symbols GBPUSD,EURJPY,EURUSD \
        --timeframe M5 --date 2026-08-05 \
        --start 08:00 --end 12:00 --threshold 40

Saída esperada:
    GBP/USD  -> 60.0% das velas de M5 têm pavios (18/30 velas)
    EUR/JPY  -> 50.0% das velas de M1 têm pavios (75/150 velas)
"""

import argparse
from datetime import datetime, time
from zoneinfo import ZoneInfo
import sys

try:
    import MetaTrader5 as mt5
except ImportError:
    print("Pacote MetaTrader5 não encontrado. Instale com: pip install MetaTrader5")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Pacote pandas não encontrado. Instale com: pip install pandas")
    sys.exit(1)


TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

# Watchlist default usada quando o usuário passa --symbols all
DEFAULT_WATCHLIST = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
]


def formatar_par(symbol: str) -> str:
    """Converte 'GBPUSD' -> 'GBP/USD' para exibição."""
    if len(symbol) == 6:
        return f"{symbol[:3]}/{symbol[3:]}"
    return symbol


def conectar_mt5() -> None:
    if not mt5.initialize():
        erro = mt5.last_error()
        print(f"Falha ao conectar ao MetaTrader5: {erro}")
        print("Verifique se o terminal MT5 está aberto e logado.")
        sys.exit(1)


def converter_para_utc(date_str: str, hora_str: str, tz_name: str) -> datetime:
    """Converte data+hora informadas em um timezone local para um datetime UTC
    (o MT5 sempre espera os horários de copy_rates_range em UTC, sem tzinfo)."""
    dia = datetime.strptime(date_str, "%Y-%m-%d").date()
    hora = datetime.strptime(hora_str, "%H:%M").time()
    dt_local = datetime.combine(dia, hora, tzinfo=ZoneInfo(tz_name))
    dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
    return dt_utc.replace(tzinfo=None)


def buscar_velas(symbol: str, timeframe_str: str, date_str: str,
                  start_str: str, end_str: str, tz_name: str) -> "pd.DataFrame | None":
    """Busca as velas do símbolo no intervalo de data/hora informado.

    date_str/start_str/end_str são interpretados no timezone `tz_name`
    (padrão: America/Sao_Paulo) e convertidos para UTC antes de consultar o MT5.
    """
    timeframe = TIMEFRAME_MAP[timeframe_str]

    dt_inicio = converter_para_utc(date_str, start_str, tz_name)
    dt_fim = converter_para_utc(date_str, end_str, tz_name)

    # Garante que o símbolo está selecionado no Market Watch
    if not mt5.symbol_select(symbol, True):
        print(f"  [aviso] Não foi possível selecionar o símbolo {symbol}, pulando.")
        return None

    rates = mt5.copy_rates_range(symbol, timeframe, dt_inicio, dt_fim)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def calcular_percentual_pavios(df: "pd.DataFrame", threshold_pct: float) -> tuple[float, int, int]:
    """Retorna (percentual, qtd_com_pavio, total_velas)."""
    range_total = df["high"] - df["low"]
    corpo_topo = df[["open", "close"]].max(axis=1)
    corpo_base = df[["open", "close"]].min(axis=1)

    pavio_superior = df["high"] - corpo_topo
    pavio_inferior = corpo_base - df["low"]
    maior_pavio = pavio_superior.combine(pavio_inferior, max)

    # Evita divisão por zero em velas doji perfeitas (range 0)
    proporcao = (maior_pavio / range_total).replace([np.inf, -np.inf], 0).fillna(0)

    threshold = threshold_pct / 100.0
    com_pavio = (proporcao >= threshold).sum()
    total = len(df)
    percentual = (com_pavio / total * 100) if total > 0 else 0.0

    return percentual, int(com_pavio), total


def escanear_mercado(symbols: list[str], timeframe_str: str, date_str: str,
                      start_str: str, end_str: str, threshold_pct: float,
                      tz_name: str) -> list[dict]:
    resultados = []
    for symbol in symbols:
        df = buscar_velas(symbol, timeframe_str, date_str, start_str, end_str, tz_name)
        if df is None or df.empty:
            print(f"  [aviso] Sem dados para {symbol} no período informado.")
            continue

        percentual, qtd, total = calcular_percentual_pavios(df, threshold_pct)
        resultados.append({
            "symbol": symbol,
            "par": formatar_par(symbol),
            "percentual": percentual,
            "qtd_com_pavio": qtd,
            "total_velas": total,
        })

    resultados.sort(key=lambda r: r["percentual"], reverse=True)
    return resultados


def main():
    parser = argparse.ArgumentParser(description="Analisador de pavios em velas de forex (MetaTrader5)")
    parser.add_argument("--symbols", type=str, default="all",
                         help="Pares separados por vírgula (ex: GBPUSD,EURJPY) ou 'all' para a watchlist padrão")
    parser.add_argument("--timeframe", type=str, required=True, choices=TIMEFRAME_MAP.keys(),
                         help="Timeframe: M1, M5, M15, M30, H1, H4, D1")
    parser.add_argument("--date", type=str, required=True, help="Data no formato AAAA-MM-DD")
    parser.add_argument("--start", type=str, required=True, help="Hora inicial HH:MM")
    parser.add_argument("--end", type=str, required=True, help="Hora final HH:MM")
    parser.add_argument("--threshold", type=float, default=40.0,
                         help="Percentual mínimo do range para considerar 'pavio grande' (padrão: 40)")
    parser.add_argument("--tz", type=str, default="America/Sao_Paulo",
                         help="Timezone dos horários --start/--end informados (padrão: America/Sao_Paulo)")

    args = parser.parse_args()

    symbols = DEFAULT_WATCHLIST if args.symbols.lower() == "all" else [
        s.strip().upper() for s in args.symbols.split(",")
    ]

    print(f"Conectando ao MetaTrader5...")
    conectar_mt5()

    print(f"Escaneando {len(symbols)} par(es) | {args.timeframe} | {args.date} "
          f"{args.start}-{args.end} ({args.tz}) | threshold={args.threshold}%\n")

    resultados = escanear_mercado(symbols, args.timeframe, args.date, args.start, args.end,
                                   args.threshold, args.tz)

    mt5.shutdown()

    if not resultados:
        print("Nenhum resultado encontrado. Verifique símbolos, datas e conexão com o terminal.")
        return

    print("Resultado (ordenado do maior para o menor % de pavios):\n")
    for r in resultados:
        print(f"  {r['par']:<10} -> {r['percentual']:.1f}% das velas de {args.timeframe} "
              f"têm pavios ({r['qtd_com_pavio']}/{r['total_velas']} velas)")


if __name__ == "__main__":
    main()
