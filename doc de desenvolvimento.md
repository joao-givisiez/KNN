# Documentacao do desenvolvimento — Trabalho Pratico 01 (KNN)

**Disciplina:** GCC 128 - Inteligencia Artificial - UFLA
**Profs.:** Ahmed Ali Abdalla Esmin / Anna Paula Figueiredo
**Base:** Iris (150 amostras, 4 atributos, 3 classes)

Este arquivo registra o processo de desenvolvimento: decisoes tomadas, problemas
encontrados e como foram resolvidos, e o resultado final de cada etapa.

---

## 1. Ambiente

Verificado o que ja estava disponivel na maquina antes de instalar qualquer coisa:

| Pacote | Versao | Uso no trabalho |
|---|---|---|
| Python | 3.11.9 | — |
| numpy | 2.3.4 | operacoes vetoriais na versao hardcore |
| pandas | 2.3.3 | leitura e tratamento dos dados (permitido pelo enunciado) |
| scikit-learn | 1.8.0 | **fonte dos dados** + classificador de comparacao |
| matplotlib | 3.10.7 | matrizes de confusao e graficos |

Instalado depois, apenas para ferramentas de apoio (nao para o algoritmo):
`nbconvert`, `nbclient`, `ipykernel` (executar o notebook) e `pymupdf`
(conferir visualmente o PDF gerado).

**Sobre a base de dados:** o enunciado indica o Kaggle ou bibliotecas que
disponibilizam o Iris. Optei pelo `sklearn.datasets.load_iris`, que ja vem
embutido no pacote — nao exige download nem cadastro no Kaggle, e e a mesma
base (Fisher, 1936). Os dados sao exportados para `data/iris.csv` na primeira
execucao, de modo que o projeto funciona offline e a leitura passa a ser feita
com pandas, como sugere o enunciado.

> Restricao respeitada: o sklearn e usado como **fonte de dados** e como
> **classificador de comparacao** (etapa explicitamente pedida). O algoritmo KNN
> da versao hardcore nao usa nada dele.

---

## 2. Estrutura criada

```
KNN/
├── main.py                    # executa tudo e imprime os resultados
├── knn_iris.ipynb             # versao notebook (ja executada, com saidas)
├── relatorio.pdf              # relatorio de 1 pagina (entrega)
├── doc de desenvolvimento.md  # este arquivo
├── README.md                  # instrucoes de uso
├── data/
│   └── iris.csv               # base exportada
├── src/
│   ├── knn_hardcore.py        # KNN do zero + metricas do zero
│   ├── knn_sklearn.py         # KNN via biblioteca
│   └── utils.py               # carga, particao estratificada, graficos
└── resultados/
    ├── matriz_confusao_{hardcore,sklearn}_k{1,3,5,7}.png
    ├── taxa_reconhecimento_{hardcore,sklearn}.png
    ├── escalabilidade.png
    └── resultados.json
```

---

## 3. Decisoes de implementacao

### 3.1 Particao treino/teste
Implementada do zero (`dividir_treino_teste` em `src/utils.py`), 70/30,
**estratificada** e com semente fixa (42).

A estratificacao preserva as 3 classes em proporcoes iguais nos dois conjuntos
(35 de treino e 15 de teste por classe). A semente fixa garante que os dois
classificadores recebam **exatamente a mesma particao** — sem isso, qualquer
diferenca observada nas metricas poderia vir do sorteio, e nao do algoritmo.

### 3.2 Selecao dos k vizinhos
Em vez de ordenar todas as n distancias (O(n log n)), uso `np.argpartition`
para achar os k menores em O(n) e ordeno apenas esses k. O resultado e
identico ao da ordenacao completa, o que foi verificado comparando as
predicoes das duas abordagens.

### 3.3 Criterio de desempate
Quando duas classes empatam em votos (possivel com k par ou com 3 classes),
vence aquela cujo vizinho **mais proximo** aparece primeiro na lista ordenada
por distancia. Adotei o mesmo criterio do sklearn justamente para que a
comparacao entre as duas implementacoes seja direta.

### 3.4 Metricas
Precisao, revocacao, F1 e acuracia sao calculadas a mao a partir da matriz de
confusao (`metricas_por_classe`), sem usar `sklearn.metrics` na versao hardcore.
As formulas estao documentadas no codigo e no notebook.

---

## 4. Problemas encontrados e correcoes

### 4.1 Diretorio `data/` inexistente
A primeira tentativa de exportar o CSV falhou (`OSError: Cannot save file into a
non-existent directory`). Resolvido criando o diretorio antes da escrita; a
funcao `carregar_iris` hoje ja faz isso com `os.makedirs(..., exist_ok=True)`.

### 4.2 Falso negativo no teste de validacao
Ao comparar a versao hardcore com uma busca por forca bruta, **todos** os k
apareceram como divergentes — inclusive k=1, onde empate e impossivel. Isso
indicava erro no teste, nao no classificador (o sklearn concordava 100% com a
minha implementacao).

Causa: no codigo de referencia eu usei `Counter(...).most_common(1)[0][1]`, que
retorna a **contagem** do rotulo mais comum, e nao o rotulo (`[0][0]`). A funcao
de referencia estava devolvendo numeros em vez de nomes de classe.

Corrigido o indice, a equivalencia passou a ser confirmada para todos os k.

### 4.3 Celulas do notebook coladas em uma linha so
O gerador do `.ipynb` quebrava o codigo com `split("\n")`, que **remove** o
`\n` de cada linha. O formato `.ipynb` exige que cada item da lista `source`
mantenha a quebra de linha ao final — sem isso, o Jupyter concatena tudo numa
linha unica e a primeira celula quebrava com `SyntaxError`.

Detectado ao executar o notebook com `nbconvert --execute`. Corrigido trocando
por `splitlines(keepends=True)`. O notebook foi entao executado do inicio ao fim
com sucesso (40 celulas, 0 erros).

### 4.4 Sobreposicao no PDF
Na primeira versao do relatorio o titulo da secao 4 sobrepunha os rotulos das
matrizes de confusao, e o grafico de escalabilidade ficou pequeno demais.
Ajustadas as coordenadas dos eixos e conferido o resultado renderizando o PDF
como imagem.

---

## 5. Validacao da implementacao

A implementacao foi submetida a 22 verificacoes, todas passando:

1. **Hardcore x forca bruta** — as predicoes batem com uma busca ingenua que
   ordena todas as distancias, validando a otimizacao com `argpartition`.
2. **Metricas proprias x sklearn** — matriz de confusao, precisao, revocacao e
   F1 identicos aos do `sklearn.metrics` (tolerancia 1e-12).
3. **Hardcore x KNeighborsClassifier** — 100% de concordancia nas predicoes.
4. **Casos de borda** — k=1 reclassifica o treino com 100% de acerto; acuracia
   dentro de [0,1]; matriz de confusao soma o total de amostras; `k<=0`, `k`
   maior que a base e `predict` antes de `fit` levantam as excecoes corretas.

---

## 6. Resultados

### Taxa de reconhecimento (conjunto de teste, 45 amostras)

| k | Hardcore | Sklearn | Diferenca |
|---|---|---|---|
| 1 | 0,9333 | 0,9333 | 0,0000 |
| 3 | 0,9333 | 0,9333 | 0,0000 |
| 5 | **0,9556** | **0,9556** | 0,0000 |
| 7 | **0,9556** | **0,9556** | 0,0000 |

Concordancia entre as duas implementacoes: **45/45 (100%) para todos os k**.

### Metricas (medias macro)

| k | Precisao HC / SK | Revocacao HC / SK | F1 HC / SK |
|---|---|---|---|
| 1 | 0,9345 / 0,9345 | 0,9333 / 0,9333 | 0,9333 / 0,9333 |
| 3 | 0,9345 / 0,9345 | 0,9333 / 0,9333 | 0,9333 / 0,9333 |
| 5 | 0,9556 / 0,9556 | 0,9556 / 0,9556 | 0,9556 / 0,9556 |
| 7 | 0,9556 / 0,9556 | 0,9556 / 0,9556 | 0,9556 / 0,9556 |

### Matriz de confusao (k=5, identica nas duas versoes)

```
             setosa  versicolor  virginica
setosa           15           0          0
versicolor        0          14          1
virginica         0           1         14
```

Todos os erros ficam entre *versicolor* e *virginica* — as classes que se
sobrepoem no espaco de atributos. *Setosa* e linearmente separavel e tem
precisao e revocacao de 1,00 em todas as configuracoes.

### Desempenho (base Iris)

| k | Tempo HC (ms) | Tempo SK (ms) | Razao | Mem. HC (KB) | Mem. SK (KB) |
|---|---|---|---|---|---|
| 1 | 0,654 | 1,093 | 0,60x | 8,8 | 26,5 |
| 3 | 0,674 | 1,092 | 0,62x | 8,7 | 29,5 |
| 5 | 0,692 | 1,099 | 0,63x | 8,7 | 33,3 |
| 7 | 0,685 | 1,102 | 0,62x | 8,7 | 36,9 |

### Escalabilidade (k=5, dados sinteticos)

| n treino | Hardcore (ms) | Sklearn (ms) | Razao HC/SK |
|---|---|---|---|
| 150 | 0,611 | 1,287 | 0,48x |
| 500 | 2,827 | 1,745 | 1,62x |
| 1.000 | 7,312 | 2,072 | 3,53x |
| 2.500 | 34,204 | 4,253 | 8,04x |
| 5.000 | 111,307 | 7,839 | 14,20x |
| 10.000 | 397,773 | 17,173 | 23,16x |

---

## 7. Analise

**Por que os resultados sao identicos.** O KNN e deterministico. Fixadas a mesma
particao, a mesma metrica de distancia e a mesma regra de desempate, as duas
implementacoes **tem** que chegar ao mesmo resultado. A coincidencia perfeita
(45/45 em todos os k) nao e redundancia: e a evidencia de que a implementacao
propria esta correta.

**Efeito de k.** A acuracia sobe de 93,33% (k=1, k=3) para 95,56% (k=5, k=7).
Com k=1 a decisao depende de um unico vizinho, o que deixa o modelo sensivel a
ruido perto da fronteira; valores maiores suavizam a fronteira de decisao. O
ganho veio da correcao de exatamente um erro (uma *virginica* que era
classificada como *versicolor*).

**Sobre o tempo de execucao.** A versao hardcore ficou ~1,6x **mais rapida** que
a do sklearn na base Iris. Esse resultado e contraintuitivo e seria facil
apresenta-lo como se a implementacao propria fosse superior — mas a explicacao e
outra: com apenas 105 amostras de treino, o tempo do sklearn e dominado por
**overhead fixo** (validacao de entrada, checagem de parametros, construcao da
KD-tree), que nao chega a ser amortizado.

Por isso fiz o teste de escalabilidade: ele mostra o cruzamento das curvas. A
vantagem da versao hardcore desaparece ja entre 150 e 500 amostras, e com 10.000
amostras o sklearn e **23x mais rapido**. A implementacao propria e O(n) por
consulta num laco Python; o sklearn usa indexacao espacial e codigo compilado.

**Conclusao.** Em qualidade de classificacao os dois classificadores sao
equivalentes. A diferenca real esta no custo computacional e na robustez: a
versao propria so vence no regime de base muito pequena, enquanto a biblioteca
escala muito melhor e ainda oferece tratamento de casos de borda, outras
metricas de distancia e paralelizacao. Para aprender o algoritmo, implementar do
zero e o que torna o funcionamento explicito; para uso real, a biblioteca e a
escolha adequada.

---

## 8. Como reproduzir

```bash
python main.py                  # resultados completos + graficos + JSON
```

O notebook `knn_iris.ipynb` ja esta com todas as saidas salvas, mas pode ser
reexecutado:

```bash
python -m nbconvert --to notebook --execute --inplace knn_iris.ipynb
```

---

## 9. Checklist do enunciado

- [x] Base Iris como referencia
- [x] KNN implementado do zero, sem biblioteca com o algoritmo pronto
- [x] Taxa de reconhecimento para k = {1, 3, 5, 7}
- [x] Matriz de confusao plotada
- [x] Metricas de avaliacao (precisao, revocacao, acuracia)
- [x] Segundo classificador usando Sklearn
- [x] Matriz de confusao e metricas do classificador da biblioteca
- [x] Comparacao das metricas entre os dois classificadores *(obrigatorio)*
- [x] Comparacao de tempo de execucao e memoria *(opcional)*
- [x] Codigo em `.py` e `.ipynb`
- [x] Relatorio de 1 pagina em PDF

**Antes de entregar:** conferir o nome da dupla no cabecalho do relatorio e
compactar a pasta como `nome1_nome2.zip`.
