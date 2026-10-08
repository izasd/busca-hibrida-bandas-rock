# Trabalho 4 — recuperação lexical, embeddings e busca híbrida

**Tema do corpus:** bandas de rock  
**Data da apresentação:** 09/10/2026

## 1. Objetivo

O trabalho compara três maneiras de localizar frases relevantes: coincidência
de palavras por BM25, similaridade semântica com BGE-M3 e similaridade semântica
com paraphrase-multilingual-MiniLM-L12-v2. Depois, combina BM25 com cada modelo
por Reciprocal Rank Fusion (RRF) e por média ponderada de escores normalizados.

O código do Trabalho 3 serviu como ponto de partida conceitual: leitura do
corpus, geração de embeddings, consulta e apresentação dos primeiros
resultados. A nova implementação acrescenta um índice lexical, avaliação com
rótulos, cache de vetores, duas fusões e varredura de hiperparâmetros.

## 2. Dataset

O arquivo `dados/dataset_bandas_rock_100k.txt` contém exatamente 100.000 frases
UTF-8, uma por linha. As perguntas usam 20 bandas reais e 25 tópicos. Cada tópico
tem 4.000 frases, formadas por quatro variantes temáticas, cinco formas de
referir-se à banda e dez estruturas de pergunta.

As frases são perguntas de busca sintéticas. Elas pedem informações sobre
formação, integrantes, discografia e outros temas, sem afirmar acontecimentos
biográficos sobre as bandas. Assim, o gerador pode usar nomes reais sem inventar
datas, cidades, integrantes ou apresentações. O CSV auxiliar acrescenta `id` e
`topico`, usados como rótulo de relevância depois da recuperação.

## 3. Métodos implementados

### 3.1 BM25

O texto é convertido para minúsculas e tokenizado com expressão regular
Unicode. O BM25 valoriza termos frequentes na consulta e raros no corpus,
compensando o tamanho do documento. Ele é especialmente forte quando consulta
e frase compartilham as mesmas palavras.

### 3.2 Embeddings

Os textos e as consultas são convertidos em vetores normalizados. A pontuação é
o produto escalar, equivalente ao cosseno quando os vetores têm norma unitária.

- **BGE-M3:** vetor denso de 1.024 dimensões; o modelo também foi projetado para
  recuperação multilíngue e contextos longos.
- **paraphrase-multilingual-MiniLM-L12-v2:** vetor de 384 dimensões, treinado
  para representar textos semanticamente próximos em um espaço multilíngue.

### 3.3 Reciprocal Rank Fusion

O RRF usa apenas as posições dos documentos:

`RRF(d) = Σ 1 / (k + rank_i(d))`

Valores pequenos de `k` aumentam a influência das primeiras posições. Valores
maiores distribuem a contribuição por uma parte mais ampla de cada ranking. O
teste usa `k = 10, 30, 60 e 100`.

### 3.4 Média ponderada com normalização

Como BM25 e cosseno têm escalas diferentes, cada vetor de pontuações é
normalizado por min–max em cada consulta:

`norm(s) = (s - min(s)) / (max(s) - min(s))`

Depois:

`score(d) = w_lex × norm(BM25(d)) + (1 - w_lex) × norm(cosseno(d))`

Foram testados pesos lexicais de 0,2; 0,4; 0,5; 0,6 e 0,8.

## 4. Protocolo de teste

O corpus completo foi criado e o código aceita `--sample-per-topic 0` para
indexá-lo. As medições entregues foram executadas em CPU, sem GPU, sobre uma
amostra estratificada e determinística de 10.000 frases: 400 de cada um dos 25
tópicos. A amostra mantém a mesma proporção do corpus e permite comparar os
métodos em uma base maior sem indexar todas as 100 mil frases.

Foram usadas 16 consultas: oito literais, com palavras presentes nas perguntas,
e oito semânticas, formuladas como paráfrases. Para cada consulta, uma pergunta
é relevante quando possui o tópico esperado.

As métricas principais são Precision@10, nDCG@10, MRR@10 e Recall@100. A análise
prioriza nDCG@10 porque a tarefa apresenta uma lista curta e a posição dos
resultados importa.

## 5. Resultados medidos

### 5.1 Métodos isolados

| Método | Precision@10 | nDCG@10 | MRR@10 | Recall@100 |
|---|---:|---:|---:|---:|
| BM25 | 0,719 | 0,720 | 0,780 | 0,175 |
| MiniLM multilíngue | 0,931 | **0,938** | **0,969** | 0,193 |
| BGE-M3 | **0,938** | 0,937 | 0,950 | **0,228** |

O MiniLM teve o maior nDCG@10 isolado, por uma diferença pequena em relação ao
BGE-M3. Nas consultas semânticas, MiniLM alcançou 0,903 e BGE-M3 0,881; nas
literais, BGE-M3 chegou a 0,992, MiniLM a 0,973 e BM25 a 0,944. O BM25 teve
desempenho menor nas consultas semânticas (0,496), como esperado para buscas
formuladas como paráfrases.

### 5.2 Média ponderada

Resultados com BGE-M3:

| Peso BM25 | Peso BGE-M3 | nDCG@10 | Precision@10 |
|---:|---:|---:|---:|
| 0,2 | 0,8 | **0,996** | **0,994** |
| 0,4 | 0,6 | 0,934 | 0,931 |
| 0,5 | 0,5 | 0,923 | 0,919 |
| 0,6 | 0,4 | 0,889 | 0,887 |
| 0,8 | 0,2 | 0,803 | 0,794 |

A configuração 0,2/0,8 superou os métodos isolados em nDCG@10: 0,996 contra
0,937 do BGE-M3 e 0,938 do MiniLM. Nesta amostra, aumentar o peso lexical
reduziu continuamente o nDCG. A melhor média com MiniLM também usou peso
lexical 0,2 e atingiu nDCG@10 de 0,928.

### 5.3 RRF

Resultados com BGE-M3:

| k | nDCG@10 | Precision@10 |
|---:|---:|---:|
| 10 | 0,861 | 0,844 |
| 30 | 0,924 | 0,900 |
| 60 | 0,963 | 0,950 |
| 100 | **0,980** | **0,975** |

Com BGE-M3, o maior `k` testado foi o melhor: `k = 100` atingiu nDCG@10 de
0,980, acima do BGE-M3 isolado (0,937), mas abaixo da média ponderada 0,2/0,8
(0,996). O aumento de `k` melhorou o RRF nesta amostra.

Com MiniLM, o melhor RRF usou `k = 100` e chegou a nDCG@10 de 0,890, abaixo do
MiniLM isolado (0,938).

### 5.4 Custo medido

| Etapa | Tempo |
|---|---:|
| BM25: preparação e 16 consultas | 0,345 s |
| BGE-M3: 10.000 embeddings | 1.139,923 s (19 min) |
| BGE-M3: 16 consultas e busca exata | 1,067 s |
| MiniLM: 10.000 embeddings | 107,305 s (1 min 47 s) |
| MiniLM: 16 consultas e busca exata | 0,115 s |

O BGE-M3 ofereceu a melhor fusão, mas a geração de seus embeddings foi
aproximadamente 10,6 vezes mais lenta que a do MiniLM neste teste em CPU. O
cache reutiliza os embeddings nas execuções seguintes com a mesma amostra. Uma
extrapolação linear para 100 mil frases indica cerca de 3,2 horas de indexação
do BGE-M3 nesta máquina; GPU e busca vetorial aproximada são recomendadas para
o corpus completo.

## 6. Conclusões

1. A melhor configuração foi a média normalizada com 20% BM25 e 80% BGE-M3,
   com nDCG@10 de 0,996.
2. Isoladamente, MiniLM e BGE-M3 ficaram praticamente empatados em nDCG@10
   (0,938 e 0,937, respectivamente); o BGE-M3 foi melhor nas consultas literais.
3. A fusão melhora o resultado, mas depende dos pesos e do método: o RRF com
   MiniLM não superou o modelo isolado nesta amostra.
4. O MiniLM indexou as 10 mil frases cerca de 10,6 vezes mais rápido que o
   BGE-M3 em CPU.
5. Os valores pertencem a perguntas sintéticas e a uma amostra de 10 mil frases.
   As estruturas geradas podem elevar as métricas; não se deve generalizá-las
   para buscas em documentos reais.

## 7. Limitações e próximos passos

O rótulo por tópico simplifica a relevância: uma pergunta de outro tópico pode
ser útil mesmo sendo marcada como não relevante. O corpus usa nomes de bandas
reais, mas consiste em perguntas sintéticas, não em biografias ou respostas
factuais. Com 400 frases relevantes por tópico, o Recall@100 tem teto de 0,25;
por isso, compare essa métrica com cautela entre amostras de tamanhos diferentes.
As estruturas repetidas facilitam a classificação temática e podem inflar as
outras métricas. Um próximo experimento deve usar documentos factuais verificados,
perguntas independentes do gerador, relevância julgada por pessoas, o corpus
completo e um índice ANN como FAISS ou HNSW. Também seria útil medir memória,
latência por percentil e estabilidade em novas consultas.

## 8. Fontes técnicas

- BGE-M3: https://huggingface.co/BAAI/bge-m3
- paraphrase-multilingual-MiniLM-L12-v2:
  https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Código e resultados deste diretório, executados em 08/10/2026.
