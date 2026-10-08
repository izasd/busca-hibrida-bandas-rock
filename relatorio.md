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
amostra estratificada e determinística de 1.000 frases: 40 de cada um dos 25
tópicos. Essa decisão torna o experimento reproduzível na máquina disponível e
mantém a mesma proporção do corpus.

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
| BM25 | 0,644 | 0,656 | 0,763 | 0,448 |
| MiniLM multilíngue | 0,788 | 0,823 | 0,969 | 0,639 |
| BGE-M3 | **0,900** | **0,895** | 0,906 | **0,845** |

O BGE-M3 foi o melhor método isolado, com nDCG@10 de 0,895. Nas consultas
semânticas, BGE-M3 alcançou 0,854 e MiniLM 0,828, enquanto BM25 caiu para 0,435.
Nas consultas literais, BM25 chegou a 0,878, MiniLM a 0,818 e BGE-M3 a 0,935.
As diferenças seguem o desenho do conjunto: as perguntas literais mantêm
vocabulário próximo às frases, enquanto as paráfrases mudam a formulação.

### 5.2 Média ponderada

Resultados com BGE-M3:

| Peso BM25 | Peso BGE-M3 | nDCG@10 | Precision@10 |
|---:|---:|---:|---:|
| 0,2 | 0,8 | **0,942** | **0,937** |
| 0,4 | 0,6 | 0,868 | 0,844 |
| 0,5 | 0,5 | 0,840 | 0,825 |
| 0,6 | 0,4 | 0,800 | 0,787 |
| 0,8 | 0,2 | 0,704 | 0,681 |

A configuração 0,2/0,8 superou o BGE-M3 isolado em nDCG@10: 0,942 contra
0,895. Pesos lexicais maiores reduziram continuamente a qualidade. O BM25
ajudou nas consultas literais, mas prejudicou mais as consultas semânticas.

Com MiniLM, a melhor média ponderada também usou peso lexical 0,2 e atingiu
nDCG@10 de 0,882, acima do MiniLM isolado (0,823).

### 5.3 RRF

Resultados com BGE-M3:

| k | nDCG@10 | Precision@10 |
|---:|---:|---:|
| 10 | 0,853 | 0,835 |
| 30 | 0,879 | **0,868** |
| 60 | 0,890 | 0,850 |
| 100 | **0,902** | 0,838 |

Com BGE-M3, o maior `k` testado foi o melhor: `k = 100` atingiu nDCG@10 de
0,902, acima do BGE-M3 isolado (0,895), mas abaixo da média ponderada 0,2/0,8
(0,942). O aumento de `k` tornou o RRF menos concentrado nas primeiras posições
e mudou a contribuição relativa dos dois rankings.

Com MiniLM, o melhor RRF usou `k = 30` e chegou a nDCG@10 de 0,868, acima do
MiniLM isolado (0,823), embora abaixo da melhor média ponderada (0,882).

### 5.4 Custo medido

| Etapa | Tempo |
|---|---:|
| BM25: preparação e 16 consultas | 0,030 s |
| BGE-M3: 1.000 embeddings | 110,328 s |
| BGE-M3: 16 consultas e busca exata | 1,149 s |
| MiniLM: 1.000 embeddings | 11,236 s |
| MiniLM: 16 consultas e busca exata | 0,137 s |

O BGE-M3 ofereceu a melhor qualidade, mas a geração de seus embeddings foi
aproximadamente 9,8 vezes mais lenta que a do MiniLM neste teste em CPU. O
cache elimina esse custo nas execuções seguintes. Para as 100 mil frases, GPU e
busca vetorial aproximada são recomendadas.

## 6. Conclusões

1. O BGE-M3 foi a melhor escolha quando qualidade é a prioridade, sobretudo em
   perguntas com paráfrases.
2. O MiniLM oferece uma relação interessante entre qualidade e custo: indexou a
   amostra cerca de dez vezes mais rápido.
3. A fusão não melhora automaticamente o melhor modelo. Ela depende da qualidade
   relativa dos sinais e dos hiperparâmetros.
4. A média normalizada foi melhor com BGE-M3 quando o embedding recebeu 80% do
   peso. O RRF foi mais útil com MiniLM e mostrou baixa sensibilidade a `k`.
5. Os valores encontrados pertencem a este corpus sintético de perguntas e à
   amostra medida. As frases e consultas seguem estruturas geradas, então as
   métricas não devem ser generalizadas para buscas em documentos reais.

## 7. Limitações e próximos passos

O rótulo por tópico simplifica a relevância: uma pergunta de outro tópico pode
ser útil mesmo sendo marcada como não relevante. O corpus usa nomes de bandas
reais, mas consiste em perguntas sintéticas, não em biografias ou respostas
factuais. As estruturas repetidas facilitam a classificação temática e podem
inflar a semelhança entre consultas e documentos. Um próximo experimento deve
usar documentos factuais verificados, perguntas independentes do gerador,
relevância julgada por pessoas, o corpus completo e um índice ANN como FAISS ou
HNSW. Também seria útil medir memória, latência por percentil e estabilidade em
novas consultas.

## 8. Fontes técnicas

- BGE-M3: https://huggingface.co/BAAI/bge-m3
- paraphrase-multilingual-MiniLM-L12-v2:
  https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Código e resultados deste diretório, executados em 08/10/2026.
