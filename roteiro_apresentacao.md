# Roteiro enxuto dos slides — Trabalho 4

**Total: 6 slides** · **Duração sugerida: 6–8 minutos**

Pouco texto na tela. Deixe detalhes nas falas e mostre o código diretamente no
editor durante a apresentação.

## 1. Apresentação da proposta

### Na tela

**Busca híbrida sobre bandas de rock**  
BM25 + embeddings · RRF · média ponderada

### Fala

Comparar busca lexical, busca semântica e a combinação dos dois resultados.

## 2. Metodologia

### Na tela

`100 mil perguntas` · `20 bandas reais` · `25 tópicos`  
Avaliação: `1.000 perguntas` · `16 consultas`

BM25 + BGE-M3 / MiniLM → RRF ou média normalizada → nDCG@10

### Visual

Um fluxo simples com os métodos e a avaliação. Sem parágrafos.

### Fala

O corpus contém perguntas sintéticas sobre nomes de bandas reais; não afirma
fatos biográficos. Os resultados foram medidos em uma amostra balanceada de
1.000 itens, 40 por tópico.

## 3. Valores

### Na tela

**nDCG@10 — métodos isolados**

| BM25 | MiniLM | BGE-M3 |
|---:|---:|---:|
| 0,656 | 0,823 | **0,895** |

**Melhores fusões**

| Combinação | nDCG@10 |
|---|---:|
| BM25 + BGE-M3, média 20/80 | **0,942** |
| BM25 + BGE-M3, RRF k=100 | 0,902 |
| BM25 + MiniLM, média 20/80 | 0,882 |
| BM25 + MiniLM, RRF k=30 | 0,868 |

### Visual

Use barras para os três métodos isolados. Mantenha a tabela de fusões pequena.

## 4. Como os valores foram alterados

### Na tela

**Média:** peso BM25 `0,2 → 0,8`  
**RRF:** `k = 10, 30, 60, 100`

### Gráficos

Dois gráficos de linha, lado a lado. Eixo y: nDCG@10.

**Peso BM25 → nDCG@10, fusão com BGE-M3**

| Peso BM25 | 0,2 | 0,4 | 0,5 | 0,6 | 0,8 |
|---|---:|---:|---:|---:|---:|
| nDCG@10 | 0,942 | 0,868 | 0,840 | 0,800 | 0,704 |

**k do RRF → nDCG@10, fusão com BGE-M3**

| k | 10 | 30 | 60 | 100 |
|---|---:|---:|---:|---:|
| nDCG@10 | 0,853 | 0,879 | 0,890 | 0,902 |

## 5. Impactos e diferenças

### Na tela

- Melhor resultado: **média 20/80 com BGE-M3 — 0,942**
- Ganho sobre BGE-M3 isolado: **+0,047**
- Mais peso lexical → menor nDCG neste teste
- Melhor RRF com BGE-M3: **k=100**

### Visual

Um destaque numérico grande: `0,942`. Use no máximo quatro chamadas curtas.

### Fala

Com paráfrases, os embeddings foram mais fortes que BM25. A fusão melhorou o
BGE-M3 quando o embedding recebeu mais peso. O RRF também melhorou com k maior,
mas ficou abaixo da média ponderada.

## 6. Código ao vivo

### Na tela

**Demonstração da implementação**

### Reservado para apresentação

Não colocar código no slide. Abrir `trabalho4.py` no editor e mostrar rapidamente:

1. BM25 e cálculo dos embeddings.
2. Fórmula RRF.
3. Média ponderada com normalização min–max.
4. Métricas e varredura de pesos / k.

Encerrar com: **“A configuração escolhida foi 20% BM25 e 80% BGE-M3.”**

---

## Preparação rápida

- Usar vírgula decimal.
- Não dizer que os resultados foram medidos nas 100 mil perguntas: a amostra
  avaliada tinha 1.000.
- Gráficos e valores vêm de `resultados/resumo_metricas.csv`.
- Tempos vêm de `resultados/tempos.json`.
- Código para a demonstração: `trabalho4.py`.
