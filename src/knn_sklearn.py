"""
Classificador KNN usando a biblioteca Sklearn (algoritmo ja implementado
por terceiros), usado como referencia de comparacao com a versao hardcore.

GCC 128 - Inteligencia Artificial - Trabalho Pratico 01
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier


def treinar_e_prever(X_treino, y_treino, X_teste, k):
    """Treina o KNeighborsClassifier do sklearn e retorna as predicoes."""
    modelo = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    modelo.fit(X_treino, y_treino)
    return modelo, modelo.predict(X_teste)


def relatorio_classificacao_sklearn(y_true, y_pred, classes):
    """Mesmo formato de relatorio da versao hardcore, porem via sklearn.

    Permite comparar os dois classificadores numero a numero.
    """
    precisao, revocacao, f1, suporte = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    por_classe = {
        classe: {
            "precisao": float(precisao[i]),
            "revocacao": float(revocacao[i]),
            "f1": float(f1[i]),
            "suporte": int(suporte[i]),
        }
        for i, classe in enumerate(classes)
    }

    macro = {}
    ponderada = {}
    for chave, valores in (
        ("precisao", precisao),
        ("revocacao", revocacao),
        ("f1", f1),
    ):
        macro[chave] = float(np.mean(valores))
        ponderada[chave] = float(np.average(valores, weights=suporte))

    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "por_classe": por_classe,
        "macro": macro,
        "ponderada": ponderada,
        "matriz_confusao": confusion_matrix(y_true, y_pred, labels=classes),
    }
