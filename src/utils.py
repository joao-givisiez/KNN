"""
Funcoes auxiliares compartilhadas: carga da base Iris, particao treino/teste
e geracao dos graficos das matrizes de confusao.

GCC 128 - Inteligencia Artificial - Trabalho Pratico 01
"""

import os

import matplotlib

matplotlib.use("Agg")  # backend sem janela, para salvar os PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_CSV = os.path.join(RAIZ, "data", "iris.csv")
DIR_RESULTADOS = os.path.join(RAIZ, "resultados")

COLUNAS_ATRIBUTOS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
COLUNA_ALVO = "species"

SEMENTE = 42
PROPORCAO_TESTE = 0.3
VALORES_K = [1, 3, 5, 7]


# ---------------------------------------------------------------------------
# Carga dos dados
# ---------------------------------------------------------------------------
def carregar_iris():
    """Le a base Iris do CSV local; se nao existir, busca no sklearn.

    Retorna um DataFrame do pandas com os 4 atributos + a coluna `species`.
    O sklearn e usado apenas como fonte dos dados -- nunca do algoritmo.
    """
    if os.path.exists(CAMINHO_CSV):
        return pd.read_csv(CAMINHO_CSV)

    from sklearn.datasets import load_iris

    dados = load_iris(as_frame=True)
    df = dados.frame.copy()
    df.columns = COLUNAS_ATRIBUTOS + ["target"]
    df[COLUNA_ALVO] = df["target"].map(dict(enumerate(dados.target_names)))
    df = df.drop(columns=["target"])

    os.makedirs(os.path.dirname(CAMINHO_CSV), exist_ok=True)
    df.to_csv(CAMINHO_CSV, index=False)
    return df


# ---------------------------------------------------------------------------
# Particao treino/teste estratificada (implementada do zero)
# ---------------------------------------------------------------------------
def dividir_treino_teste(X, y, proporcao_teste=PROPORCAO_TESTE, semente=SEMENTE):
    """Separa treino/teste mantendo a proporcao de cada classe (estratificado).

    Implementado manualmente para que a mesma particao possa ser reproduzida
    nos dois classificadores (hardcore e sklearn), garantindo comparacao justa.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(semente)

    idx_treino, idx_teste = [], []
    for classe in np.unique(y):
        indices = np.flatnonzero(y == classe)
        indices = rng.permutation(indices)
        n_teste = int(round(proporcao_teste * len(indices)))
        idx_teste.extend(indices[:n_teste])
        idx_treino.extend(indices[n_teste:])

    idx_treino = rng.permutation(np.array(idx_treino))
    idx_teste = rng.permutation(np.array(idx_teste))
    return X[idx_treino], X[idx_teste], y[idx_treino], y[idx_teste]


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def plotar_matriz_confusao(matriz, classes, titulo, caminho_saida, cmap="Blues"):
    """Salva a matriz de confusao como imagem PNG anotada."""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matriz, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Classe predita",
        ylabel="Classe real",
        title=titulo,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    limite = matriz.max() / 2.0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(
                j,
                i,
                format(matriz[i, j], "d"),
                ha="center",
                va="center",
                fontsize=12,
                color="white" if matriz[i, j] > limite else "black",
            )

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return caminho_saida


def plotar_comparacao_k(resultados_por_k, titulo, caminho_saida):
    """Grafico de barras da taxa de reconhecimento para cada valor de k."""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    ks = list(resultados_por_k)
    acuracias = [resultados_por_k[k] for k in ks]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    barras = ax.bar([str(k) for k in ks], acuracias, color="#4C72B0", width=0.55)
    ax.set(
        xlabel="k (numero de vizinhos)",
        ylabel="Taxa de reconhecimento",
        title=titulo,
        ylim=(0, 1.08),
    )
    for barra, valor in zip(barras, acuracias):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 0.02,
            f"{valor:.4f}",
            ha="center",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return caminho_saida


# ---------------------------------------------------------------------------
# Impressao formatada
# ---------------------------------------------------------------------------
def imprimir_matriz_confusao(matriz, classes, titulo=""):
    if titulo:
        print(titulo)
    largura = max(12, max(len(c) for c in classes) + 2)
    cabecalho = " " * largura + "".join(f"{c:>{largura}}" for c in classes)
    print("  (linha = real | coluna = predito)")
    print(cabecalho)
    for i, classe in enumerate(classes):
        linha = f"{classe:>{largura}}" + "".join(
            f"{matriz[i, j]:>{largura}d}" for j in range(len(classes))
        )
        print(linha)


def imprimir_relatorio(relatorio, classes, titulo=""):
    if titulo:
        print(titulo)
    largura = max(12, max(len(c) for c in classes) + 2)
    print(
        f"{'classe':>{largura}}{'precisao':>12}{'revocacao':>12}"
        f"{'f1-score':>12}{'suporte':>10}"
    )
    for classe in classes:
        m = relatorio["por_classe"][classe]
        print(
            f"{classe:>{largura}}{m['precisao']:>12.4f}{m['revocacao']:>12.4f}"
            f"{m['f1']:>12.4f}{m['suporte']:>10d}"
        )
    macro, ponderada = relatorio["macro"], relatorio["ponderada"]
    print(
        f"{'media macro':>{largura}}{macro['precisao']:>12.4f}"
        f"{macro['revocacao']:>12.4f}{macro['f1']:>12.4f}"
    )
    print(
        f"{'media pond.':>{largura}}{ponderada['precisao']:>12.4f}"
        f"{ponderada['revocacao']:>12.4f}{ponderada['f1']:>12.4f}"
    )
    print(f"\n  Acuracia (taxa de reconhecimento): {relatorio['acuracia']:.4f}")
