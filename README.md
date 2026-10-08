# Trabalho 4 — busca híbrida em frases sobre bandas de rock

O projeto adapta o código do Trabalho 3 para comparar recuperação lexical com
BM25, recuperação semântica com BGE-M3 e paraphrase-multilingual-MiniLM-L12-v2,
e duas formas de fusão: RRF e média ponderada após normalização min–max.

## Arquivos principais

- `gerar_dataset.py`: gera 100.000 perguntas sintéticas, únicas e balanceadas sobre 20 bandas reais.
- `trabalho4.py`: executa os buscadores, as fusões e a avaliação.
- `dados/dataset_bandas_rock_100k.txt`: corpus solicitado, uma frase por linha.
- `dados/dataset_bandas_rock_100k.csv`: o corpus com ID e tópico de referência.
- `dados/fontes_dataset.json`: descrição da composição do dataset e das bandas usadas.
- `dados/consultas_avaliacao.json`: 16 consultas, metade literal e metade semântica.
- `resultados/`: métricas, exemplos de resultados, tempos e gráficos medidos.
- `relatorio.md`: metodologia, análise e conclusões.
- `roteiro_apresentacao.md`: conteúdo completo para montar a apresentação de 09/10/2026.

As frases são perguntas de busca geradas por combinação de tópicos, estruturas
linguísticas e nomes de bandas reais. Elas não afirmam fatos biográficos, datas,
formações ou eventos sobre as bandas.

## Instalação

```powershell
python -m pip install -r requirements.txt
```

Na primeira execução, o Sentence Transformers baixa os pesos dos modelos.

## Reprodução

Gerar novamente o corpus:

```powershell
python gerar_dataset.py
```

Reproduzir o benchmark entregue, com 400 frases por tópico (10.000 no total):

```powershell
python trabalho4.py --sample-per-topic 400 --batch-size 8 --output-dir resultados
```

Executar nas 100.000 frases:

```powershell
python trabalho4.py --sample-per-topic 0 --batch-size 16 --output-dir resultados-100k
```

O BGE-M3 é consideravelmente mais pesado em CPU. Os vetores ficam em
`.cache_embeddings` e são reutilizados. Em uma máquina com GPU, aumente o
`--batch-size` conforme a memória disponível.

## Métricas

- **Precision@10**: proporção de itens relevantes entre os dez primeiros.
- **nDCG@10**: qualidade da ordenação, dando mais peso aos primeiros lugares.
- **MRR@10**: posição do primeiro resultado relevante.
- **Recall@100**: cobertura dos itens relevantes entre os cem primeiros.

O tópico gerado para cada frase funciona como julgamento de relevância. O
buscador não recebe esse rótulo; ele é usado apenas depois da recuperação.

## Configurações de fusão

No RRF, cada documento recebe `1 / (k + rank)` de cada lista. O experimento
avalia `k = 10, 30, 60 e 100`.

Na média ponderada, os escores de BM25 e cosseno são normalizados por consulta
para o intervalo `[0, 1]`. O experimento avalia pesos lexicais de `0,2`, `0,4`,
`0,5`, `0,6` e `0,8`; o peso do embedding é `1 - peso lexical`.
