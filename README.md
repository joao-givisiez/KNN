# Trabalho Pratico 01 — Classificacao KNN

**GCC 128 - Inteligencia Artificial — UFLA**
Profs. Ahmed Ali Abdalla Esmin / Anna Paula Figueiredo

Implementacao do algoritmo K-Nearest Neighbors **do zero (hardcore)** e comparacao
com a implementacao do **Scikit-learn**, usando a base de dados **Iris**.

## Motivacao

Usar um classificador pronto e simples; entender o que ele faz por dentro exige
reconstrui-lo. A proposta deste trabalho e justamente essa: escrever o KNN sem
recorrer a nenhuma biblioteca que ja o implemente — calculo de distancias,
selecao dos k vizinhos, votacao majoritaria, criterio de desempate e ate as
metricas de avaliacao — e so entao confrontar o resultado com o do Scikit-learn.

A comparacao serve a dois propositos. Primeiro, **validar**: como o KNN e
deterministico, fixadas a mesma particao, a mesma metrica de distancia e a mesma
regra de desempate, as duas implementacoes precisam concordar. Qualquer
divergencia apontaria um erro na versao propria — e, de fato, a concordancia foi
de 100% das amostras em todos os valores de k.

Segundo, **medir o custo real da abstracao**. Na base Iris a implementacao
propria chega a ser mais rapida que a da biblioteca, resultado contraintuitivo
que, investigado, se explica pelo overhead fixo do Scikit-learn sobre uma base de
apenas 105 amostras de treino. Um teste com bases maiores mostra a inversao e
deixa claro onde cada abordagem faz sentido — o tipo de conclusao que so aparece
quando se tem as duas implementacoes lado a lado.

## Requisitos

Python 3.8+ com `numpy`, `pandas`, `matplotlib` e `scikit-learn`:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Execucao

```bash
python main.py
```

Imprime, para k = {1, 3, 5, 7}, a taxa de reconhecimento, as matrizes de confusao
e as metricas de avaliacao das duas implementacoes, alem da analise de desempenho.
Salva os graficos e o `resultados.json` em `resultados/`.

Versao em notebook (ja executada, com as saidas salvas): `knn_iris.ipynb`.

## Estrutura

| Arquivo | Conteudo |
|---|---|
| `main.py` | script principal: roda tudo e imprime os resultados |
| `knn_iris.ipynb` | versao notebook, com graficos e explicacoes |
| `src/knn_hardcore.py` | **KNN implementado do zero** + metricas do zero |
| `src/knn_sklearn.py` | classificador usando `KNeighborsClassifier` |
| `src/utils.py` | carga dos dados, particao estratificada, graficos |
| `data/iris.csv` | base Iris exportada (funciona offline) |
| `resultados/` | graficos gerados + `resultados.json` |
| `relatorio.pdf` | relatorio de 1 pagina (entrega) |
| `doc de desenvolvimento.md` | documentacao do desenvolvimento e analise completa |

## Resultados

| k | Taxa de reconhecimento (hardcore) | Taxa (sklearn) |
|---|---|---|
| 1 | 0,9333 | 0,9333 |
| 3 | 0,9333 | 0,9333 |
| 5 | **0,9556** | **0,9556** |
| 7 | **0,9556** | **0,9556** |

As duas implementacoes produzem predicoes **identicas** (45/45 amostras) para
todos os valores de k — o que valida a implementacao feita do zero.

Detalhes da analise em [doc de desenvolvimento.md](doc%20de%20desenvolvimento.md)
e em `relatorio.pdf`.

## Observacao sobre o uso do Scikit-learn

O enunciado proibe usar bibliotecas **com o algoritmo KNN ja implementado** na
versao hardcore. No projeto, o sklearn aparece apenas:

1. como **fonte da base Iris** (`load_iris`) — permitido pelo enunciado;
2. como **classificador de comparacao** — etapa explicitamente pedida.

O modulo `src/knn_hardcore.py` nao importa nada do sklearn: distancias, selecao
de vizinhos, votacao e metricas de avaliacao sao todas escritas a mao.
