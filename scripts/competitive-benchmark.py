#!/usr/bin/env python3
"""Competitive single-query latency bake-off for MetalANNS vs embedded vector libraries.

Protocol (identical for every backend unless noted):
  - Synthetic corpus: float32, dim 384, Gaussian mixture over min(256, n/64)
    cluster centers (mimics the topical clustering of agent-memory embeddings),
    rows L2-normalized (cosine). Deterministic under numpy seed 42.
  - Queries: 200 fresh draws from the same mixture (seed 7).
  - k = 10, warmup 20 queries, then 200 individually timed queries.
  - Reported: p50/p95 single-query latency, QPS (= 1e6 / p50_us), recall@10
    against brute-force ground truth, index build time.
  - Approximate backends are tuned upward (doubling ef) until recall@10 >= 0.99
    on a 50-query probe, then measured at that operating point on all queries.
  - MetalANNS runs through its own Swift harness on a uniform-random corpus of
    the same shape; exact-scan latency is content-independent, and quoting the
    harder distribution keeps its column conservative.

Usage:
  uv pip install faiss-cpu hnswlib usearch sqlite-vec matplotlib numpy
  python3 scripts/competitive-benchmark.py --scales 10000,50000,100000 \
      --metalanns-dir .build/bakeoff --out .build/bakeoff/results.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

DIM = 384
K = 10
QUERY_COUNT = 200
WARMUP = 20
TUNE_TARGET_RECALL = 0.99


def make_corpus(n, seed):
    rng = np.random.default_rng(seed)
    n_clusters = max(8, min(256, n // 64))
    centers = rng.standard_normal((n_clusters, DIM), dtype=np.float32)
    assignment = rng.integers(0, n_clusters, size=n)
    x = centers[assignment] + 0.35 * rng.standard_normal((n, DIM), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def make_queries(seed):
    rng = np.random.default_rng(seed)
    n_clusters = 256
    centers = rng.standard_normal((n_clusters, DIM), dtype=np.float32)
    assignment = rng.integers(0, n_clusters, size=QUERY_COUNT)
    q = centers[assignment] + 0.35 * rng.standard_normal(
        (QUERY_COUNT, DIM), dtype=np.float32
    )
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def ground_truth(corpus, queries):
    scores = queries @ corpus.T
    return np.argsort(-scores, axis=1)[:, :K]


def recall_at_k(pred, truth):
    hits = sum(len(set(row_p) & set(row_t)) for row_p, row_t in zip(pred, truth))
    return hits / (len(truth) * K)


def time_queries(fn):
    for i in range(WARMUP):
        fn(i % QUERY_COUNT)
    samples = []
    for i in range(QUERY_COUNT):
        t0 = time.perf_counter_ns()
        fn(i)
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    arr = np.array(samples)
    return {
        "p50_us": float(np.percentile(arr, 50)),
        "p95_us": float(np.percentile(arr, 95)),
    }


def tune_ef(set_ef, predict, probe_truth, start_ef=64, cap_ef=4096):
    ef = start_ef
    best_recall = 0.0
    while ef <= cap_ef:
        set_ef(ef)
        pred = [predict(i) for i in range(len(probe_truth))]
        best_recall = recall_at_k(pred, probe_truth)
        if best_recall >= TUNE_TARGET_RECALL:
            return ef, best_recall
        ef *= 2
    return ef // 2, best_recall


class BenchResult(dict):
    pass


def bench_numpy_exact(corpus, queries, truth):
    def query(i):
        scores = corpus @ queries[i]
        idx = np.argpartition(-scores, K)[:K]
        return idx[np.argsort(-scores[idx])]

    timing = time_queries(lambda i: query(i))
    pred = [query(i) for i in range(QUERY_COUNT)]
    return BenchResult(
        backend="numpy (exact scan)",
        kind="exact",
        build_s=0.0,
        tuned_ef=None,
        recall=recall_at_k(pred, truth),
        **timing,
    )


def bench_faiss_flat(corpus, queries, truth):
    import faiss

    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(DIM)
    index.add(corpus)
    build_s = time.perf_counter() - t0

    pred = []
    for i in range(QUERY_COUNT):
        _, ids = index.search(queries[i : i + 1], K)
        pred.append(ids[0])
    rec = recall_at_k(pred, truth)

    timing = time_queries(lambda i: index.search(queries[i : i + 1], K))
    return BenchResult(
        backend="FAISS IndexFlatIP",
        kind="exact",
        build_s=build_s,
        tuned_ef=None,
        recall=rec,
        **timing,
    )


def bench_faiss_hnsw(corpus, queries, truth):
    import faiss

    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(DIM, 16)
    index.hnsw.efConstruction = 200
    index.add(corpus)
    build_s = time.perf_counter() - t0

    probe_n = 50

    def predict(i):
        _, ids = index.search(queries[i : i + 1], K)
        return ids[0]

    ef, _ = tune_ef(
        lambda v: setattr(index.hnsw, "efSearch", v),
        predict,
        truth[:probe_n],
    )

    pred = [predict(i) for i in range(QUERY_COUNT)]
    rec = recall_at_k(pred, truth)

    timing = time_queries(predict)
    return BenchResult(
        backend="FAISS HNSW",
        kind="approx",
        build_s=build_s,
        tuned_ef=ef,
        recall=rec,
        **timing,
    )


def bench_hnswlib(corpus, queries, truth):
    import hnswlib

    t0 = time.perf_counter()
    index = hnswlib.Index(space="ip", dim=DIM)
    index.init_index(max_elements=len(corpus), ef_construction=200, M=16)
    index.add_items(corpus, np.arange(len(corpus)))
    build_s = time.perf_counter() - t0

    probe_n = 50

    def predict(i):
        labels, _ = index.knn_query(queries[i : i + 1], k=K)
        return labels[0]

    ef, _ = tune_ef(index.set_ef, predict, truth[:probe_n])

    pred = [predict(i) for i in range(QUERY_COUNT)]
    rec = recall_at_k(pred, truth)

    timing = time_queries(predict)
    return BenchResult(
        backend="hnswlib",
        kind="approx",
        build_s=build_s,
        tuned_ef=ef,
        recall=rec,
        **timing,
    )


def bench_usearch(corpus, queries, truth):
    from usearch.index import Index, MetricKind, ScalarKind

    t0 = time.perf_counter()
    index = Index(
        ndim=DIM,
        metric=MetricKind.Cos,
        dtype=ScalarKind.F32,
        connectivity=16,
        expansion_add=128,
        expansion_search=64,
    )
    index.add(np.arange(len(corpus)), corpus)
    build_s = time.perf_counter() - t0

    probe_n = 50

    def predict(i):
        return [int(key) for key in index.search(queries[i], count=K).keys]

    ef, _ = tune_ef(
        lambda v: setattr(index, "expansion_search", v),
        predict,
        truth[:probe_n],
    )

    pred = [predict(i) for i in range(QUERY_COUNT)]
    rec = recall_at_k(pred, truth)

    timing = time_queries(predict)
    return BenchResult(
        backend="USearch",
        kind="approx",
        build_s=build_s,
        tuned_ef=ef,
        recall=rec,
        **timing,
    )


def bench_sqlite_vec(corpus, queries, truth):
    import sqlite3

    import sqlite_vec

    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    t0 = time.perf_counter()
    db.execute(
        f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{DIM}])"
    )
    db.executemany(
        "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
        [(i, corpus[i].tobytes()) for i in range(len(corpus))],
    )
    build_s = time.perf_counter() - t0

    sql = "SELECT rowid FROM vec_items WHERE embedding MATCH ? AND k = ?"

    def query(i):
        rows = db.execute(sql, (queries[i].tobytes(), K)).fetchall()
        return [r[0] for r in rows]

    pred = [query(i) for i in range(QUERY_COUNT)]
    rec = recall_at_k(pred, truth)

    timing = time_queries(query)
    return BenchResult(
        backend="sqlite-vec",
        kind="exact",
        build_s=build_s,
        tuned_ef=None,
        recall=rec,
        **timing,
    )


def load_metalanns(dir_path, n):
    path = Path(dir_path) / f"metalanns_{n}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    row = report["rows"][0]
    return BenchResult(
        backend="MetalANNS",
        kind="exact",
        build_s=row["buildTimeMs"] / 1000.0,
        tuned_ef=None,
        recall=row["recallAt10"],
        p50_us=row["p50Ms"] * 1000.0,
        p95_us=row["p95Ms"] * 1000.0,
    )


BACKENDS = [
    bench_numpy_exact,
    bench_sqlite_vec,
    bench_faiss_flat,
    bench_hnswlib,
    bench_usearch,
    bench_faiss_hnsw,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", default="10000,50000,100000")
    parser.add_argument("--metalanns-dir", default=".build/bakeoff")
    parser.add_argument("--out", default=".build/bakeoff/results.json")
    args = parser.parse_args()

    scales = [int(s) for s in args.scales.split(",")]
    results = []
    for n in scales:
        print(f"\n=== scale {n:,} ===")
        corpus = make_corpus(n, seed=42)
        queries = make_queries(seed=7)
        truth = ground_truth(corpus, queries)

        ma = load_metalanns(args.metalanns_dir, n)
        if ma:
            results.append({**ma, "scale": n})
            print(
                f"  {'MetalANNS':24s} {ma['kind']:6s} "
                f"p50 {ma['p50_us']:9.1f} us  recall@10 {ma['recall']:.3f}"
            )

        for bench_fn in BACKENDS:
            try:
                r = bench_fn(corpus, queries, truth)
                r["scale"] = n
                results.append(r)
                ef_note = f" ef={r['tuned_ef']}" if r["tuned_ef"] else ""
                print(
                    f"  {r['backend']:24s} {r['kind']:6s} "
                    f"p50 {r['p50_us']:9.1f} us  recall@10 {r['recall']:.3f}{ef_note}"
                )
            except Exception as exc:
                print(f"  {bench_fn.__name__}: FAILED ({exc})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
