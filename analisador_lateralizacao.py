"""
Analisador de Lateralizacao / Alternancia de Cor - MetaTrader5
=================================================================

Objetivo
--------
Complementa o analisador de pavios: em vez de medir pavios grandes,
mede o quanto o mercado ficou "lateralizado" (velas alternando cor
sim/nao seguidamente) em uma janela de tempo, para um ou varios
pares. Isso ajuda a responder duas perguntas praticas da estrategia
de fluxo de vela:

    1) Dentro do horario que eu opero, a partir de que hora o
       mercado costuma perder liquidez e comecar a lateralizar?
    2) Entre varios pares candidatos, qual deles esta com o
       comportamento mais "limpo" (tendencia/impulso) agora,
       e qual esta mais "sujo" (lateralizado)?

Requisitos
----------
    pip install MetaTrader5 pandas pytz

O terminal MetaTrader5 precisa estar instalado e logado (mesma
conta/corretora usada no analisador de pavios), rodando no Windows
ou via Wine, pois o pacote MetaTrader5 se conecta ao terminal local.

Conceitos usados
-----------------
- "Alternancia de cor": comparamos a cor de cada vela com a da vela
  anterior. Se mudou (alta -> baixa ou baixa -> alta), conta como
  uma alternancia. Um mercado saudavel/tendencial tende a ter varias
  velas seguidas da mesma cor; um mercado lateralizado alterna quase
  vela a vela.
- "Corpo relativo": corpo da vela (abertura-fechamento) dividido
  pelo range total da vela (maxima-minima). Velas de lateralizacao
  costumam ter corpo pequeno e pavios dos dois lados (indecisao).
- "Sequencia maxima de alternancia": maior trecho consecutivo de
  velas alternando cor sem quebrar o padrao. E o que voce descreveu
  como "8 velas cor sim cor nao seguidas".
- "Score de propicio": combina baixa alternancia + corpo saudavel
  em um unico numero, para ranquear pares. Quanto maior, melhor.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import MetaTrader5 as mt5
import pandas as pd
import pytz

# ---------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}

# Corpo abaixo disso (em % do range da vela) e tratado como
# "vela de indecisao" (quase doji) para fins de diagnostico.
CORPO_INDECISAO_PCT = 15.0


# ---------------------------------------------------------------
# Conexao e coleta de dados
# ---------------------------------------------------------------

def conectar_mt5() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"Falha ao conectar no MetaTrader5: {mt5.last_error()}")


def obter_velas(
    par: str,
    timeframe: str,
    dia: dt.date,
    hora_inicio: int,
    hora_fim: int,
    tz_name: str = "America/Sao_Paulo",
) -> pd.DataFrame:
    """
    Busca as velas de um par no dia e janela de horario informados.

    hora_inicio / hora_fim: inteiros de 0 a 23, no fuso `tz_name`
    (padrao horario de Brasilia).
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Timeframe invalido: {timeframe}. Use um de {list(TIMEFRAMES)}")

    tz = pytz.timezone(tz_name)
    inicio_local = tz.localize(dt.datetime.combine(dia, dt.time(hour=hora_inicio)))
    # hora_fim e INCLUSIVA: pega a hora inteira ate xx:59:59, nao so o instante xx:00:00
    fim_local = tz.localize(dt.datetime.combine(dia, dt.time(hour=hora_fim, minute=59, second=59)))

    inicio_utc = inicio_local.astimezone(pytz.utc)
    fim_utc = fim_local.astimezone(pytz.utc)

    rates = mt5.copy_rates_range(par, TIMEFRAMES[timeframe], inicio_utc, fim_utc)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Sem dados para {par} nesse periodo (erro: {mt5.last_error()})")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(tz_name)
    return df


def amostra_para_validacao(df: pd.DataFrame, n: int = 5) -> None:
    """
    Imprime algumas linhas (time + OHLC) para voce validar manualmente
    no grafico do MT5. IMPORTANTE: o campo 'time' que a MetaTrader5
    devolve normalmente esta no horario DO SERVIDOR da corretora, que
    quase nunca e UTC de verdade (muitas usam EET, UTC+2/+3). A
    conversao de fuso aqui assume que o dado veio em UTC puro - se o
    seu servidor nao for UTC, o horario mostrado vai ficar deslocado.

    Como validar:
      1) Rode esta funcao e anote o horario + os precos (open/close)
         de 2-3 velas.
      2) Abra o mesmo par no MT5, va ate o horario mostrado no
         GRAFICO (que usa o horario do servidor, exibido no eixo X)
         e compare os precos daquela vela com os impressos aqui.
      3) Se os precos batem mas o horario exibido no grafico for
         diferente do horario aqui, a diferenca em horas E o offset
         real do seu servidor - ajuste tz_name ou aplique esse offset
         manualmente antes de interpretar hora_inicio/hora_fim.
    """
    print(df[["time", "open", "high", "low", "close"]].head(n).to_string(index=False))


# ---------------------------------------------------------------
# Metricas de lateralizacao
# ---------------------------------------------------------------

@dataclass
class MetricasLateralizacao:
    par: str
    total_velas: int
    pct_alternancia: float
    maior_sequencia_alternada: int
    corpo_medio_pct: float
    pct_velas_indecisao: float
    score_propicio: float
    por_hora: pd.DataFrame = field(repr=False)


def _cor_vela(df: pd.DataFrame) -> pd.Series:
    # True = alta (fechou acima da abertura), False = baixa/neutra
    return df["close"] >= df["open"]


def calcular_metricas(df: pd.DataFrame, par: str) -> MetricasLateralizacao:
    df = df.copy()
    df["cor_alta"] = _cor_vela(df)
    df["range"] = (df["high"] - df["low"]).replace(0, pd.NA)
    df["corpo"] = (df["close"] - df["open"]).abs()
    df["corpo_pct"] = (df["corpo"] / df["range"] * 100).fillna(0)

    # alternancia: cor diferente da vela anterior
    df["alternou"] = df["cor_alta"] != df["cor_alta"].shift(1)
    df.loc[df.index[0], "alternou"] = False  # primeira vela nao conta

    total_velas = len(df)
    total_alternancias = int(df["alternou"].sum())
    pct_alternancia = round(100 * total_alternancias / max(total_velas - 1, 1), 2)

    # maior sequencia consecutiva de alternancia (streak de True em 'alternou')
    maior_seq = 0
    seq_atual = 0
    for alternou in df["alternou"]:
        if alternou:
            seq_atual += 1
            maior_seq = max(maior_seq, seq_atual)
        else:
            seq_atual = 0
    # +1 porque a sequencia de alternancia envolve (streak+1) velas
    maior_sequencia_alternada = maior_seq + 1 if maior_seq > 0 else 1

    corpo_medio_pct = round(df["corpo_pct"].mean(), 2)
    pct_velas_indecisao = round(100 * (df["corpo_pct"] < CORPO_INDECISAO_PCT).mean(), 2)

    # score: quanto maior, mais "propicio" (baixa alternancia + corpo saudavel)
    score_propicio = round(corpo_medio_pct * (1 - pct_alternancia / 100), 2)

    # quebra por hora, para achar o horario de corte
    df["hora"] = df["time"].dt.hour
    por_hora = (
        df.groupby("hora")
        .apply(lambda g: pd.Series({
            "velas": len(g),
            "pct_alternancia": round(100 * g["alternou"].sum() / max(len(g) - 1, 1), 2),
            "corpo_medio_pct": round(g["corpo_pct"].mean(), 2),
        }))
        .reset_index()
    )

    return MetricasLateralizacao(
        par=par,
        total_velas=total_velas,
        pct_alternancia=pct_alternancia,
        maior_sequencia_alternada=maior_sequencia_alternada,
        corpo_medio_pct=corpo_medio_pct,
        pct_velas_indecisao=pct_velas_indecisao,
        score_propicio=score_propicio,
        por_hora=por_hora,
    )


# ---------------------------------------------------------------
# Analise de multiplos pares (ranking)
# ---------------------------------------------------------------

def analisar_pares(
    pares: list[str],
    timeframe: str,
    dia: dt.date,
    hora_inicio: int,
    hora_fim: int,
) -> pd.DataFrame:
    """
    Roda a analise para uma lista de pares e devolve um DataFrame
    ranqueado do mais propicio (menos lateralizado) para o menos
    propicio, com base no score_propicio.
    """
    linhas = []
    for par in pares:
        try:
            df = obter_velas(par, timeframe, dia, hora_inicio, hora_fim)
            m = calcular_metricas(df, par)
            linhas.append({
                "par": m.par,
                "velas": m.total_velas,
                "% alternancia": m.pct_alternancia,
                "maior sequencia alternada": m.maior_sequencia_alternada,
                "corpo medio %": m.corpo_medio_pct,
                "% velas indecisao": m.pct_velas_indecisao,
                "score propicio": m.score_propicio,
            })
        except RuntimeError as e:
            print(f"[aviso] {par}: {e}")

    resultado = pd.DataFrame(linhas).sort_values("score propicio", ascending=False)
    return resultado.reset_index(drop=True)


def encontrar_horario_de_corte(m: MetricasLateralizacao) -> str:
    """
    Em vez de usar um limite fixo (que e arbitrario e sensivel a
    ruido de um unico dia), compara cada hora com a MEDIA do proprio
    dia analisado e procura o primeiro ponto de piora SUSTENTADA
    (2 horas seguidas acima da media) - isso evita apontar corte por
    causa de um pico isolado, como aconteceria em um dia sem
    tendencia clara ao longo das horas.
    """
    ph = m.por_hora.copy()
    if len(ph) < 3:
        return "Poucas horas no periodo para identificar um padrao confiavel."

    media_dia = ph["pct_alternancia"].mean()
    ph["acima_da_media"] = ph["pct_alternancia"] > media_dia

    for i in range(len(ph) - 1):
        if ph.iloc[i]["acima_da_media"] and ph.iloc[i + 1]["acima_da_media"]:
            hora = int(ph.iloc[i]["hora"])
            return (
                f"Piora sustentada a partir das {hora}h "
                f"(2h seguidas acima da media do dia, que foi {media_dia:.1f}%)."
            )

    return (
        f"Sem piora sustentada de 2h seguidas no periodo - a alternancia "
        f"oscilou em torno de {media_dia:.1f}% o dia inteiro, sem um "
        f"horario de corte claro hoje. Rode em mais dias (analisar_varios_dias) "
        f"para ver se esse patamar geral e normal para este par ou se hoje foi atipico."
    )


def analisar_varios_dias(
    par: str,
    timeframe: str,
    dias: list[dt.date],
    hora_inicio: int,
    hora_fim: int,
) -> pd.DataFrame:
    """
    Roda a analise por hora em varios dias e tira a MEDIA de cada
    hora entre os dias. Um unico dia tem so ~60 velas por hora, o
    que e pouco e ruidoso; agregando varios dias, o padrao de horario
    de baixa liquidez (se existir de verdade) fica bem mais claro e
    confiavel do que olhando um dia isolado.
    """
    tabelas = []
    for dia in dias:
        try:
            df = obter_velas(par, timeframe, dia, hora_inicio, hora_fim)
            m = calcular_metricas(df, par)
            tabelas.append(m.por_hora.assign(dia=dia))
        except RuntimeError as e:
            print(f"[aviso] {dia}: {e}")

    if not tabelas:
        raise RuntimeError("Nenhum dia retornou dados.")

    todas = pd.concat(tabelas, ignore_index=True)
    media_por_hora = (
        todas.groupby("hora")
        .agg(
            dias_amostrados=("dia", "nunique"),
            pct_alternancia_media=("pct_alternancia", "mean"),
            pct_alternancia_desvio=("pct_alternancia", "std"),
            corpo_medio_pct=("corpo_medio_pct", "mean"),
        )
        .round(2)
        .reset_index()
    )
    return media_por_hora


# ---------------------------------------------------------------
# Exemplo de uso
# ---------------------------------------------------------------

if __name__ == "__main__":
    conectar_mt5()

    dia = dt.date.today()
    pares_candidatos = ["GBPUSD" , "AUDUSD"]

    print(f"\n=== Ranking de pares mais propicios - {dia} ===\n")
    ranking = analisar_pares(
        pares=pares_candidatos,
        timeframe="M1",
        dia=dia,
        hora_inicio=9,
        hora_fim=13,
    )
    print(ranking.to_string(index=False))

    # Detalhe do melhor colocado: em que hora ele comeca a lateralizar
    if not ranking.empty:
        melhor_par = ranking.iloc[0]["par"]
        df_melhor = obter_velas(melhor_par, "M1", dia, 9, 13)
        m_melhor = calcular_metricas(df_melhor, melhor_par)
        print(f"\n=== Quebra por hora - {melhor_par} ===\n")
        print(m_melhor.por_hora.to_string(index=False))
        print("\n" + encontrar_horario_de_corte(m_melhor))

    # ---- Analise de varios dias (padrao real de horario, nao ruido de 1 dia) ----
    dias = []
    d = dt.date.today()
    while len(dias) < 30:
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:  # pula sabado(5) e domingo(6)
            dias.append(d)

    par_para_historico = melhor_par if not ranking.empty else "EURUSD"
    media = analisar_varios_dias(par_para_historico, "M1", dias, 9, 13)
    print(f"\n=== Media de alternancia por hora - {par_para_historico} (ultimos {len(dias)} dias uteis) ===\n")
    print(media.to_string(index=False))

    mt5.shutdown()