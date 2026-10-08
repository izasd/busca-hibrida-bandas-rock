"""Trabalho 4 - recuperação lexical, semântica e híbrida sobre bandas de rock.

Exemplo de execução do experimento usado no relatório:

    python gerar_dataset.py
    python trabalho4.py --sample-per-topic 400 --batch-size 8 --output-dir resultados

Para indexar as 100.000 frases, use ``--sample-per-topic 0``. Em CPU, o BGE-M3
é demorado; os embeddings são salvos em cache e reaproveitados nas execuções.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


@dataclass
class QueryMetric:
    query_id: str
    query_type: str
    topic: str
    method: str
    configuration: str
    precision_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_100: float


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_corpus(csv_path: Path, sample_per_topic: int) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    expected = {"id", "topico", "frase"}
    if set(frame.columns) != expected:
        raise ValueError(f"Colunas esperadas: {sorted(expected)}")
    if sample_per_topic > 0:
        frame = (
            frame.groupby("topico", group_keys=False)
            .sample(n=sample_per_topic, random_state=42)
            .sort_values("id")
            .reset_index(drop=True)
        )
    return frame


def load_queries(path: Path) -> list[dict[str, str]]:
    queries = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "tipo", "topico", "consulta"}
    for query in queries:
        if not required.issubset(query):
            raise ValueError(f"Consulta incompleta: {query}")
    return queries


def corpus_signature(sentences: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sentence in sentences:
        digest.update(sentence.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def minmax(scores: np.ndarray) -> np.ndarray:
    minimum = float(scores.min())
    maximum = float(scores.max())
    if math.isclose(minimum, maximum):
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - minimum) / (maximum - minimum)).astype(np.float32)


def order_desc(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")


def ranks_from_order(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return ranks


def evaluate_ranking(
    order: np.ndarray,
    topics: np.ndarray,
    relevant_topic: str,
) -> tuple[float, float, float, float]:
    relevant = topics == relevant_topic
    top10 = relevant[order[:10]].astype(np.float32)
    precision = float(top10.mean())
    discounts = 1.0 / np.log2(np.arange(2, 12))
    dcg = float((top10 * discounts).sum())
    ideal_count = min(10, int(relevant.sum()))
    idcg = float(discounts[:ideal_count].sum()) if ideal_count else 0.0
    ndcg = dcg / idcg if idcg else 0.0
    positions = np.flatnonzero(top10)
    mrr = 1.0 / float(positions[0] + 1) if len(positions) else 0.0
    recall100 = float(relevant[order[:100]].sum() / max(1, relevant.sum()))
    return precision, ndcg, mrr, recall100


def metric_row(
    query: dict[str, str], method: str, configuration: str,
    order: np.ndarray, topics: np.ndarray,
) -> QueryMetric:
    p10, ndcg10, mrr10, r100 = evaluate_ranking(order, topics, query["topico"])
    return QueryMetric(
        query_id=query["id"], query_type=query["tipo"], topic=query["topico"],
        method=method, configuration=configuration,
        precision_at_10=p10, ndcg_at_10=ndcg10,
        mrr_at_10=mrr10, recall_at_100=r100,
    )


def encode_cached(
    model_key: str,
    model_id: str,
    sentences: list[str],
    cache_dir: Path,
    batch_size: int,
) -> tuple[SentenceTransformer, np.ndarray, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = corpus_signature(sentences)
    cache_path = cache_dir / f"{model_key}-{len(sentences)}-{signature}.npy"
    model = SentenceTransformer(model_id)
    start = time.perf_counter()
    if cache_path.exists():
        embeddings = np.load(cache_path, mmap_mode="r")
        elapsed = time.perf_counter() - start
        print(f"Cache carregado: {cache_path}")
    else:
        print(f"Gerando {len(sentences):,} embeddings com {model_id}...")
        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(cache_path, embeddings)
        elapsed = time.perf_counter() - start
        print(f"Embeddings salvos: {cache_path}")
    return model, embeddings, elapsed


def write_top_results(
    output_path: Path,
    queries: list[dict[str, str]],
    corpus: pd.DataFrame,
    rankings: dict[str, list[np.ndarray]],
) -> None:
    lines: list[str] = []
    for query_index, query in enumerate(queries):
        lines.extend([f"[{query['id']}] {query['consulta']}", f"Tópico esperado: {query['topico']}"])
        for method, method_rankings in rankings.items():
            lines.append(f"  {method}")
            for rank, index in enumerate(method_rankings[query_index][:5], start=1):
                row = corpus.iloc[int(index)]
                marker = "✓" if row["topico"] == query["topico"] else "×"
                lines.append(f"    {rank}. {marker} [{row['topico']}] {row['frase']}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def create_charts(summary: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"bge-m3": "#E45756", "minilm": "#4C78A8"}

    isolated = summary[
        ((summary["method"] == "BM25") | (summary["configuration"] == "isolado"))
        & (summary["query_type"] == "todos")
    ].copy()
    labels = ["BM25" if m == "BM25" else m for m in isolated["method"]]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, isolated["ndcg_at_10"], color=["#72B7B2", "#E45756", "#4C78A8"][:len(labels)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("nDCG@10")
    ax.set_title("Qualidade dos métodos isolados")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "metodos_isolados.png", dpi=180, transparent=False)
    plt.close(fig)

    for model_key in MODELS:
        weighted = summary[
            (summary["method"] == f"BM25+{model_key}")
            & summary["configuration"].str.startswith("media_peso_lex=")
            & (summary["query_type"] == "todos")
        ].copy()
        weighted["peso"] = weighted["configuration"].str.split("=").str[-1].astype(float)
        rrf = summary[
            (summary["method"] == f"BM25+{model_key}")
            & summary["configuration"].str.startswith("rrf_k=")
            & (summary["query_type"] == "todos")
        ].copy()
        rrf["k"] = rrf["configuration"].str.split("=").str[-1].astype(int)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].plot(weighted["peso"], weighted["ndcg_at_10"], marker="o", color=colors[model_key])
        axes[0].set_xlabel("Peso do BM25")
        axes[0].set_ylabel("nDCG@10")
        axes[0].set_title("Média ponderada")
        axes[0].set_ylim(0, 1.05)
        axes[1].plot(rrf["k"], rrf["ndcg_at_10"], marker="o", color=colors[model_key])
        axes[1].set_xlabel("k do RRF")
        axes[1].set_ylabel("nDCG@10")
        axes[1].set_title("Reciprocal Rank Fusion")
        axes[1].set_ylim(0, 1.05)
        fig.suptitle(f"Sensibilidade da fusão com {model_key}")
        fig.tight_layout()
        fig.savefig(output_dir / f"sensibilidade_{model_key}.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("dados/dataset_bandas_rock_100k.csv"))
    parser.add_argument("--queries", type=Path, default=Path("dados/consultas_avaliacao.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("resultados"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_embeddings"))
    parser.add_argument(
        "--sample-per-topic", type=int, default=400,
        help="0 usa as 100 mil frases; N seleciona N frases de cada tópico.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(args.corpus, args.sample_per_topic)
    queries = load_queries(args.queries)
    sentences = corpus["frase"].astype(str).tolist()
    topics = corpus["topico"].astype(str).to_numpy()
    query_texts = [query["consulta"] for query in queries]
    print(f"Corpus do experimento: {len(corpus):,} frases; {len(queries)} consultas")

    timings: dict[str, float | int | str] = {
        "corpus_total": 100_000,
        "corpus_experimento": len(corpus),
        "sample_per_topic": args.sample_per_topic,
    }
    start = time.perf_counter()
    tokenized = [tokenize(sentence) for sentence in sentences]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = np.vstack([bm25.get_scores(tokenize(query)) for query in query_texts]).astype(np.float32)
    timings["bm25_segundos"] = time.perf_counter() - start
    bm25_orders = [order_desc(scores) for scores in bm25_scores]

    metrics: list[QueryMetric] = []
    display_rankings: dict[str, list[np.ndarray]] = {"BM25": bm25_orders}
    for query, order in zip(queries, bm25_orders):
        metrics.append(metric_row(query, "BM25", "isolado", order, topics))

    for model_key in args.models:
        model, embeddings, encode_seconds = encode_cached(
            model_key, MODELS[model_key], sentences, args.cache_dir, args.batch_size
        )
        timings[f"{model_key}_indexacao_segundos"] = encode_seconds
        start = time.perf_counter()
        query_embeddings = model.encode(
            query_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        dense_scores = np.asarray(query_embeddings @ np.asarray(embeddings).T, dtype=np.float32)
        timings[f"{model_key}_consultas_segundos"] = time.perf_counter() - start
        dense_orders = [order_desc(scores) for scores in dense_scores]
        display_rankings[model_key] = dense_orders

        for query, order in zip(queries, dense_orders):
            metrics.append(metric_row(query, model_key, "isolado", order, topics))

        for k in (10, 30, 60, 100):
            fused_orders: list[np.ndarray] = []
            for lexical_order, dense_order in zip(bm25_orders, dense_orders):
                lexical_ranks = ranks_from_order(lexical_order)
                dense_ranks = ranks_from_order(dense_order)
                fused_score = 1.0 / (k + lexical_ranks) + 1.0 / (k + dense_ranks)
                fused_orders.append(order_desc(fused_score))
            for query, order in zip(queries, fused_orders):
                metrics.append(metric_row(query, f"BM25+{model_key}", f"rrf_k={k}", order, topics))
            if k == 60:
                display_rankings[f"RRF BM25+{model_key} (k=60)"] = fused_orders

        for lexical_weight in (0.2, 0.4, 0.5, 0.6, 0.8):
            fused_orders = []
            for lexical_score, dense_score in zip(bm25_scores, dense_scores):
                fused_score = lexical_weight * minmax(lexical_score) + (1.0 - lexical_weight) * minmax(dense_score)
                fused_orders.append(order_desc(fused_score))
            config = f"media_peso_lex={lexical_weight:.1f}"
            for query, order in zip(queries, fused_orders):
                metrics.append(metric_row(query, f"BM25+{model_key}", config, order, topics))
            if math.isclose(lexical_weight, 0.5):
                display_rankings[f"Média BM25+{model_key} (0,5/0,5)"] = fused_orders

        del model, embeddings, dense_scores, query_embeddings

    metric_frame = pd.DataFrame(asdict(item) for item in metrics)
    metric_frame.to_csv(args.output_dir / "metricas_por_consulta.csv", index=False, encoding="utf-8-sig")

    overall = (
        metric_frame.groupby(["method", "configuration"], as_index=False)
        [["precision_at_10", "ndcg_at_10", "mrr_at_10", "recall_at_100"]]
        .mean()
    )
    overall["query_type"] = "todos"
    by_type = (
        metric_frame.groupby(["method", "configuration", "query_type"], as_index=False)
        [["precision_at_10", "ndcg_at_10", "mrr_at_10", "recall_at_100"]]
        .mean()
    )
    summary = pd.concat([overall, by_type], ignore_index=True)
    summary.to_csv(args.output_dir / "resumo_metricas.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "tempos.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_top_results(args.output_dir / "top_5_resultados.txt", queries, corpus, display_rankings)
    create_charts(summary, args.output_dir)

    best = summary[summary["query_type"] == "todos"].sort_values("ndcg_at_10", ascending=False)
    print("\nMelhores configurações por nDCG@10:")
    print(best.head(12).to_string(index=False))
    print(f"\nArquivos gravados em {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
