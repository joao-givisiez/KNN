"""
Implementacao "hardcore" (do zero) do algoritmo KNN.

Nenhuma biblioteca com o KNN pronto e usada aqui. Apenas numpy para
operacoes vetoriais basicas (soma, raiz, ordenacao) e estruturas nativas
do Python. A logica do algoritmo -- calculo de distancia, selecao dos k
vizinhos e votacao -- e toda escrita a mao.

GCC 128 - Inteligencia Artificial - Trabalho Pratico 01
"""

from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Metricas de distancia
# ---------------------------------------------------------------------------
def distancia_euclidiana(a, b):
    """Distancia euclidiana entre o vetor `a` e cada linha da matriz `b`.

    d(a, b) = sqrt( sum_i (a_i - b_i)^2 )
    """
    return np.sqrt(np.sum((b - a) ** 2, axis=1))


def distancia_manhattan(a, b):
    """Distancia de Manhattan entre o vetor `a` e cada linha da matriz `b`."""
    return np.sum(np.abs(b - a), axis=1)


METRICAS = {
    "euclidiana": distancia_euclidiana,
    "manhattan": distancia_manhattan,
}


# ---------------------------------------------------------------------------
# Classificador
# ---------------------------------------------------------------------------
class KNNHardcore:
    """Classificador K-Nearest Neighbors implementado do zero.

    Parametros
    ----------
    k : int
        Quantidade de vizinhos considerados na votacao.
    metrica : str
        "euclidiana" (padrao) ou "manhattan".
    """

    def __init__(self, k=3, metrica="euclidiana"):
        if k < 1:
            raise ValueError("k deve ser >= 1")
        if metrica not in METRICAS:
            raise ValueError(f"metrica deve ser uma de {list(METRICAS)}")
        self.k = k
        self.metrica = metrica
        self._funcao_distancia = METRICAS[metrica]
        self.X_treino = None
        self.y_treino = None
        self.classes_ = None

    # -- KNN e um lazy learner: "treinar" e apenas memorizar a base -------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y devem ter o mesmo numero de amostras")
        if self.k > X.shape[0]:
            raise ValueError("k nao pode ser maior que o numero de amostras de treino")
        self.X_treino = X
        self.y_treino = y
        self.classes_ = np.unique(y)
        return self

    def _classificar_amostra(self, amostra):
        """Classifica uma unica amostra pela votacao dos k vizinhos."""
        # 1) distancia da amostra para todos os pontos de treino
        distancias = self._funcao_distancia(amostra, self.X_treino)

        # 2) indices dos k menores valores (argpartition e O(n), mais barato
        #    que ordenar tudo); depois ordenamos apenas esses k
        idx_parciais = np.argpartition(distancias, self.k - 1)[: self.k]
        idx_vizinhos = idx_parciais[np.argsort(distancias[idx_parciais])]

        rotulos_vizinhos = self.y_treino[idx_vizinhos]

        # 3) votacao majoritaria simples
        votos = Counter(rotulos_vizinhos)
        maior = max(votos.values())
        empatados = [rotulo for rotulo, n in votos.items() if n == maior]

        if len(empatados) == 1:
            return empatados[0]

        # Desempate: vence a classe cujo vizinho mais proximo aparece antes
        # na lista ja ordenada por distancia.
        for rotulo in rotulos_vizinhos:
            if rotulo in empatados:
                return rotulo
        return empatados[0]  # inalcancavel, mantido por seguranca

    def predict(self, X):
        if self.X_treino is None:
            raise RuntimeError("chame fit() antes de predict()")
        X = np.asarray(X, dtype=float)
        return np.array([self._classificar_amostra(amostra) for amostra in X])

    def score(self, X, y):
        """Taxa de reconhecimento (acuracia) sobre o conjunto informado."""
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------------------
# Metricas de avaliacao implementadas do zero
# ---------------------------------------------------------------------------
def matriz_confusao(y_true, y_pred, classes):
    """Matriz de confusao com linhas = valor real e colunas = predito."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    indice = {c: i for i, c in enumerate(classes)}
    matriz = np.zeros((len(classes), len(classes)), dtype=int)
    for real, predito in zip(y_true, y_pred):
        matriz[indice[real], indice[predito]] += 1
    return matriz


def acuracia(y_true, y_pred):
    """(VP + VN) / total -- fracao de acertos sobre todas as amostras."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def metricas_por_classe(y_true, y_pred, classes):
    """Precisao, revocacao e F1 por classe, calculados da matriz de confusao.

    precisao_c  = VP_c / (VP_c + FP_c)   -> coluna c da matriz
    revocacao_c = VP_c / (VP_c + FN_c)   -> linha  c da matriz
    f1_c        = 2 * P * R / (P + R)
    """
    matriz = matriz_confusao(y_true, y_pred, classes)
    resultado = {}
    for i, classe in enumerate(classes):
        vp = matriz[i, i]
        fp = matriz[:, i].sum() - vp
        fn = matriz[i, :].sum() - vp
        precisao = vp / (vp + fp) if (vp + fp) > 0 else 0.0
        revocacao = vp / (vp + fn) if (vp + fn) > 0 else 0.0
        f1 = (
            2 * precisao * revocacao / (precisao + revocacao)
            if (precisao + revocacao) > 0
            else 0.0
        )
        resultado[classe] = {
            "precisao": float(precisao),
            "revocacao": float(revocacao),
            "f1": float(f1),
            "suporte": int(matriz[i, :].sum()),
        }
    return resultado


def medias_macro_ponderada(por_classe):
    """Media macro (simples) e ponderada (pelo suporte) das metricas."""
    classes = list(por_classe)
    total = sum(por_classe[c]["suporte"] for c in classes)

    macro = {
        chave: float(np.mean([por_classe[c][chave] for c in classes]))
        for chave in ("precisao", "revocacao", "f1")
    }
    ponderada = {
        chave: float(
            sum(por_classe[c][chave] * por_classe[c]["suporte"] for c in classes) / total
        )
        for chave in ("precisao", "revocacao", "f1")
    }
    return macro, ponderada


def relatorio_classificacao(y_true, y_pred, classes):
    """Agrega todas as metricas de avaliacao em um unico dicionario."""
    por_classe = metricas_por_classe(y_true, y_pred, classes)
    macro, ponderada = medias_macro_ponderada(por_classe)
    return {
        "acuracia": acuracia(y_true, y_pred),
        "por_classe": por_classe,
        "macro": macro,
        "ponderada": ponderada,
        "matriz_confusao": matriz_confusao(y_true, y_pred, classes),
    }
