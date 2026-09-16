"""
Trabalho Pratico 01 - Classificacao KNN
GCC 128 - Inteligencia Artificial - UFLA
Ahmed Ali Abdalla Esmin / Anna Paula Figueiredo

Script principal: executa os dois classificadores (hardcore e sklearn)
sobre a base Iris, imprime as matrizes de confusao e as metricas de
avaliacao, e compara o desempenho das duas implementacoes.

Uso:  python main.py
"""

import json
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from knn_hardcore import KNNHardcore, relatorio_classificacao  # noqa: E402
from knn_sklearn import relatorio_classificacao_sklearn, treinar_e_prever  # noqa: E402
from utils import (  # noqa: E402
    COLUNAS_ATRIBUTOS,
    COLUNA_ALVO,
    DIR_RESULTADOS,
    VALORES_K,
    carregar_iris,
    dividir_treino_teste,
    imprimir_matriz_confusao,
    imprimir_relatorio,
    plotar_comparacao_k,
    plotar_matriz_confusao,
)

LARGURA = 78


def titulo(texto, caractere="="):
    print("\n" + caractere * LARGURA)
    print(texto.center(LARGURA))
    print(caractere * LARGURA)


def medir(funcao, repeticoes=10):
    """Executa `funcao` medindo tempo (melhor de N) e pico de memoria.

    O tempo reportado e o menor entre as repeticoes, que e a estimativa mais
    estavel do custo real (menos sujeita a ruido do sistema operacional).
    """
    tempos = []
    resultado = None
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        resultado = funcao()
        tempos.append(time.perf_counter() - inicio)

    tracemalloc.start()
    funcao()
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return resultado, {
        "tempo_min_s": float(min(tempos)),
        "tempo_medio_s": float(np.mean(tempos)),
        "memoria_pico_kb": pico / 1024.0,
    }


def main():
    os.makedirs(DIR_RESULTADOS, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Carga e preparacao dos dados
    # -----------------------------------------------------------------
    titulo("TRABALHO PRATICO 01 - CLASSIFICACAO KNN (BASE IRIS)")

    df = carregar_iris()
    X = df[COLUNAS_ATRIBUTOS].values
    y = df[COLUNA_ALVO].values
    classes = sorted(np.unique(y).tolist())

    X_treino, X_teste, y_treino, y_teste = dividir_treino_teste(X, y)

    print(f"\nAmostras totais .......: {len(df)}")
    print(f"Atributos .............: {len(COLUNAS_ATRIBUTOS)} -> {COLUNAS_ATRIBUTOS}")
    print(f"Classes ...............: {classes}")
    print(f"Treino / Teste ........: {len(X_treino)} / {len(X_teste)} (split 70/30)")
    print("Particao ..............: estratificada, semente fixa (42)")
    print("Distancia .............: euclidiana")

    # -----------------------------------------------------------------
    # 2. Taxa de reconhecimento para k = {1, 3, 5, 7}
    # -----------------------------------------------------------------
    titulo("TAXA DE RECONHECIMENTO PARA k = {1, 3, 5, 7}")

    acuracias_hc, acuracias_sk = {}, {}
    tempos_hc, tempos_sk = {}, {}
    relatorios_hc, relatorios_sk = {}, {}

    print(f"\n{'k':>4}{'KNN hardcore':>18}{'KNN sklearn':>16}{'diferenca':>14}")
    print("-" * 52)

    for k in VALORES_K:
        # --- implementacao hardcore ---
        def rodar_hardcore(k=k):
            modelo = KNNHardcore(k=k, metrica="euclidiana")
            modelo.fit(X_treino, y_treino)
            return modelo.predict(X_teste)

        y_pred_hc, perf_hc = medir(rodar_hardcore)
        rel_hc = relatorio_classificacao(y_teste, y_pred_hc, classes)

        # --- implementacao sklearn ---
        def rodar_sklearn(k=k):
            _, predicoes = treinar_e_prever(X_treino, y_treino, X_teste, k)
            return predicoes

        y_pred_sk, perf_sk = medir(rodar_sklearn)
        rel_sk = relatorio_classificacao_sklearn(y_teste, y_pred_sk, classes)

        acuracias_hc[k] = rel_hc["acuracia"]
        acuracias_sk[k] = rel_sk["acuracia"]
        tempos_hc[k] = perf_hc
        tempos_sk[k] = perf_sk
        relatorios_hc[k] = rel_hc
        relatorios_sk[k] = rel_sk

        diferenca = rel_hc["acuracia"] - rel_sk["acuracia"]
        print(
            f"{k:>4}{rel_hc['acuracia']:>18.4f}{rel_sk['acuracia']:>16.4f}"
            f"{diferenca:>14.4f}"
        )

    # concordancia entre as predicoes das duas implementacoes
    titulo("CONCORDANCIA ENTRE AS DUAS IMPLEMENTACOES", "-")
    for k in VALORES_K:
        modelo = KNNHardcore(k=k).fit(X_treino, y_treino)
        pred_hc = modelo.predict(X_teste)
        _, pred_sk = treinar_e_prever(X_treino, y_treino, X_teste, k)
        iguais = int(np.sum(pred_hc == pred_sk))
        print(
            f"  k={k}: {iguais}/{len(y_teste)} predicoes identicas "
            f"({100.0 * iguais / len(y_teste):.2f}%)"
        )

    # -----------------------------------------------------------------
    # 3. Matrizes de confusao e metricas detalhadas
    # -----------------------------------------------------------------
    for k in VALORES_K:
        titulo(f"RESULTADOS DETALHADOS PARA k = {k}")

        print("\n--- KNN HARDCORE (implementacao propria) ---\n")
        imprimir_matriz_confusao(relatorios_hc[k]["matriz_confusao"], classes)
        print()
        imprimir_relatorio(relatorios_hc[k], classes)

        print("\n--- KNN SKLEARN (biblioteca) ---\n")
        imprimir_matriz_confusao(relatorios_sk[k]["matriz_confusao"], classes)
        print()
        imprimir_relatorio(relatorios_sk[k], classes)

        plotar_matriz_confusao(
            relatorios_hc[k]["matriz_confusao"],
            classes,
            f"Matriz de confusao - KNN hardcore (k={k})",
            os.path.join(DIR_RESULTADOS, f"matriz_confusao_hardcore_k{k}.png"),
        )
        plotar_matriz_confusao(
            relatorios_sk[k]["matriz_confusao"],
            classes,
            f"Matriz de confusao - KNN sklearn (k={k})",
            os.path.join(DIR_RESULTADOS, f"matriz_confusao_sklearn_k{k}.png"),
            cmap="Greens",
        )

    # -----------------------------------------------------------------
    # 4. Analise de desempenho
    # -----------------------------------------------------------------
    titulo("ANALISE DE DESEMPENHO: HARDCORE x SKLEARN")

    print("\nMetricas de avaliacao (medias macro sobre o conjunto de teste)\n")
    print(
        f"{'k':>3} | {'acuracia HC':>11} {'acuracia SK':>11} | "
        f"{'precisao HC':>11} {'precisao SK':>11} | "
        f"{'revocacao HC':>12} {'revocacao SK':>12}"
    )
    print("-" * LARGURA)
    for k in VALORES_K:
        hc, sk = relatorios_hc[k], relatorios_sk[k]
        print(
            f"{k:>3} | {hc['acuracia']:>11.4f} {sk['acuracia']:>11.4f} | "
            f"{hc['macro']['precisao']:>11.4f} {sk['macro']['precisao']:>11.4f} | "
            f"{hc['macro']['revocacao']:>12.4f} {sk['macro']['revocacao']:>12.4f}"
        )

    print("\n\nTempo de execucao (fit + predict, melhor de 10 execucoes)\n")
    print(
        f"{'k':>3} | {'hardcore (ms)':>15} {'sklearn (ms)':>14} "
        f"{'razao HC/SK':>13}"
    )
    print("-" * LARGURA)
    for k in VALORES_K:
        t_hc = tempos_hc[k]["tempo_min_s"] * 1000
        t_sk = tempos_sk[k]["tempo_min_s"] * 1000
        print(f"{k:>3} | {t_hc:>15.3f} {t_sk:>14.3f} {t_hc / t_sk:>13.2f}x")

    print("\n\nPico de memoria alocada durante fit + predict\n")
    print(f"{'k':>3} | {'hardcore (KB)':>15} {'sklearn (KB)':>14}")
    print("-" * LARGURA)
    for k in VALORES_K:
        print(
            f"{k:>3} | {tempos_hc[k]['memoria_pico_kb']:>15.1f} "
            f"{tempos_sk[k]['memoria_pico_kb']:>14.1f}"
        )

    # -----------------------------------------------------------------
    # 5. Graficos e exportacao dos resultados
    # -----------------------------------------------------------------
    plotar_comparacao_k(
        acuracias_hc,
        "Taxa de reconhecimento por k - KNN hardcore",
        os.path.join(DIR_RESULTADOS, "taxa_reconhecimento_hardcore.png"),
    )
    plotar_comparacao_k(
        acuracias_sk,
        "Taxa de reconhecimento por k - KNN sklearn",
        os.path.join(DIR_RESULTADOS, "taxa_reconhecimento_sklearn.png"),
    )

    resumo = {
        "configuracao": {
            "base": "Iris (150 amostras, 4 atributos, 3 classes)",
            "split": "70% treino / 30% teste, estratificado, semente 42",
            "distancia": "euclidiana",
            "valores_k": VALORES_K,
        },
        "hardcore": {
            str(k): {
                "acuracia": relatorios_hc[k]["acuracia"],
                "macro": relatorios_hc[k]["macro"],
                "ponderada": relatorios_hc[k]["ponderada"],
                "por_classe": relatorios_hc[k]["por_classe"],
                "matriz_confusao": relatorios_hc[k]["matriz_confusao"].tolist(),
                "desempenho": tempos_hc[k],
            }
            for k in VALORES_K
        },
        "sklearn": {
            str(k): {
                "acuracia": relatorios_sk[k]["acuracia"],
                "macro": relatorios_sk[k]["macro"],
                "ponderada": relatorios_sk[k]["ponderada"],
                "por_classe": relatorios_sk[k]["por_classe"],
                "matriz_confusao": relatorios_sk[k]["matriz_confusao"].tolist(),
                "desempenho": tempos_sk[k],
            }
            for k in VALORES_K
        },
    }

    caminho_json = os.path.join(DIR_RESULTADOS, "resultados.json")
    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, indent=2, ensure_ascii=False)

    titulo("ARQUIVOS GERADOS")
    for nome in sorted(os.listdir(DIR_RESULTADOS)):
        print(f"  resultados/{nome}")
    print()


if __name__ == "__main__":
    main()
