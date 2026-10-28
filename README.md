# vector

An embeddable vector database in pure Go: no cgo and no dependencies outside
the standard library. It is built to be linked into other programs to store
and search **text embeddings** for semantic search, RAG, deduplication and
recommendations.

- **HNSW** approximate index (parallel bulk build) and exact **Flat** index
- **Cosine**, **Euclidean** and **DotProduct** metrics, with unrolled pure-Go distance kernels
- Records carry an ID, the source text (`Content`) and string `Metadata`
- **Metadata filtering**, with automatic choice of exact scan vs. graph traversal by selectivity
- **`Embedder` interface** plus `UpsertDocuments` / `SearchText` to plug in any embedding model
- **Single-file persistence**: crash-safe, checksummed snapshots that include the HNSW graph, so loads don't rebuild
- **OS-level file locking** (flock / LockFileEx, no cgo) during writes, with conflict detection between processes
- Safe for concurrent use: parallel searches, serialised writes
- Tested with unit, recall, race-detector and fuzz tests, plus a benchmark suite

## Install

```sh
go get github.com/brimstone/go-vector
```

The package name is `vector`:

```go
import "github.com/brimstone/go-vector" // used as vector.Open, vector.Options, ...
```

## Quick start

```go
db, err := vector.Open("app.vdb")         // single file; vector.Open("") = memory only
if err != nil { log.Fatal(err) }
defer db.Close()                          // saves changes to disk

docs, err := db.GetOrCreateCollection("docs", vector.Options{
    Dimension: 768,                       // your embedding model's dimension
    // Metric: vector.Cosine (default), Index: vector.HNSW (default)
})

err = docs.Upsert("handbook.md#0", embedding, "chunk text…",
    vector.Metadata{"path": "handbook.md"})

hits, err := docs.Search(queryEmbedding, 5,
    vector.WithFilter(vector.Where("path", "handbook.md")))
for _, h := range hits {
    fmt.Printf("%.3f %s %s\n", h.Similarity(), h.ID, h.Content)
}
```

### With an embedding model

Implement the one-method `Embedder` interface around your model: an HTTP API,
a local server such as Ollama, or an in-process model. Then work with text
directly:

```go
var emb vector.Embedder = myEmbedder{}    // Embed(ctx, []string) ([][]float32, error)

err := docs.UpsertDocuments(ctx, emb, []vector.Document{
    {ID: "faq#1", Content: "Refunds are processed within 5 days.", Metadata: vector.Metadata{"src": "faq"}},
})
hits, err := docs.SearchText(ctx, emb, "how long do refunds take?", 3)
```

### With the `embedding` package

The companion package [`github.com/brimstone/go-embedding`](https://github.com/brimstone/go-embedding)
handles the text side, also in pure Go with no dependencies. It chunks
documents along paragraph and sentence boundaries, embeds them, and runs a
concurrent chunk → embed → store `Pipeline`. Its embedders work offline,
with no model server:

| Embedder | Matches | Adds to your binary |
|---|---|---|
| `glove.Embedder()` | Meaning ("automobile" finds "car"), via bundled GloVe word vectors (100k words, 300 dims); English only | ~32 MB |
| `embedding.NewHashEmbedder(…)` | Shared words and word fragments; good for names, codes and typos | nothing |

This package doesn't depend on `go-embedding`, so you add it yourself.
Only programs that import `glove` carry its vectors, but every `go get` of
the module downloads the ~30 MB data file once into the module cache.

An `embedding.Embedder` also satisfies `vector.Embedder`, so it works with
`SearchText` and `UpsertDocuments` as it is. To have a pipeline write into a
collection, adapt the collection to `embedding.Store`:

```go
import (
    "github.com/brimstone/go-embedding"
    "github.com/brimstone/go-embedding/glove"
    "github.com/brimstone/go-vector"
)

// collectionStore lets an embedding.Pipeline write into a vector.Collection.
type collectionStore struct{ c *vector.Collection }

func (s collectionStore) UpsertBatch(recs []embedding.Record) error {
    out := make([]vector.Record, len(recs))
    for i, r := range recs {
        out[i] = vector.Record{ID: r.ID, Vector: r.Vector,
            Content: r.Metadata[embedding.MetaText], Metadata: r.Metadata}
    }
    return s.c.UpsertBatch(out)
}

chunker, _ := embedding.NewRecursiveChunker(800, 100)  // runes, overlap
emb, err := glove.Embedder()                           // decodes the bundled vectors on first use (~0.1 s)

docs, err := db.GetOrCreateCollection("docs-glove", vector.Options{Dimension: glove.Dimension}) // 300
p, err := embedding.NewPipeline(chunker, emb, embedding.PipelineOptions{Store: collectionStore{docs}})

// Records get IDs "handbook.md#0", "#1", … and metadata doc_id, chunk, start, end.
docs.DeleteWhere(vector.Where(embedding.MetaDocID, "handbook.md")) // drop stale chunks when re-indexing
_, err = p.Ingest(ctx, "handbook.md", text, map[string]string{"team": "hr"})

hits, err := docs.SearchText(ctx, emb, "how many vacation days do I get?", 3)
```

Keep each embedder's vectors in their own collection, and name the
collection after the embedder, as `docs-glove` is here. GloVe, hashing and
API models produce vectors that can't be compared, even when the dimensions
match. To switch embedders, re-ingest into a new collection.

GloVe averages word meanings (SIF weighting). It is a large step up from
hashing for paraphrased queries, but a neural model served over HTTP does
better still on long or subtle queries. Any model can sit behind the
`Embedder` interface. A remote model's `Dimension()` may return 0 until the
first call, so embed a probe string to size the collection.
`embedding.Handler` serves any embedder, GloVe included, over an
OpenAI-compatible `/v1/embeddings` endpoint.

## Sample program

[`examples/semantic-search`](examples/semantic-search) is a complete program
that embeds the library. It chunks `.txt`/`.md` files with
`embedding.RecursiveChunker`, ingests them through an `embedding.Pipeline`
into a collection on disk, and answers natural-language queries. It includes
an OpenAI-compatible HTTP embedder (OpenAI, Ollama, vLLM, llama.cpp, LM
Studio…). By default it uses the bundled `glove` vectors, so it searches
by meaning with no setup and no network. `-embedder hash` switches to
`embedding.HashEmbedder`. Its `serve` command puts both offline embedders
behind `embedding.Handler`.

The example is a separate Go module, so the library keeps no dependencies.
Its `go.mod` points at local checkouts of `go-vector` and `go-embedding`
side by side; edit the `replace` lines if yours live elsewhere.

```sh
cd examples/semantic-search
go run . demo                                         # offline demo, GloVe (by meaning)
go run . -embedder hash demo                          # offline demo, hashing (by shared words)

go run . serve localhost:8080 &                       # glove-300 and hash-512 as an OpenAI-style API
go run . -embed-url http://localhost:8080/v1 -model glove-300 demo

ollama pull nomic-embed-text                          # or any embeddings API
go run . -embed-url http://localhost:11434/v1 -model nomic-embed-text index ./docs
go run . -embed-url http://localhost:11434/v1 -model nomic-embed-text \
    -k 3 search "how do I persist a collection?"
```

## Storage and multiple processes

The whole database is one file. `Save` writes the new snapshot beside the
current one, fsyncs it, then switches a small double-buffered header over to
it. A crash at any point leaves the previous snapshot readable.

While `Save` writes, it holds an exclusive OS lock on the file, so another
process can't write at the same time; `Open` takes a shared lock while
reading. Because every process works on its own in-memory copy, `Save` also
checks a counter in the header: if another process saved after you loaded,
you get `ErrConflict` and nothing is overwritten.

## Choosing settings

| Situation | Recommendation |
|---|---|
| Text embeddings | `Metric: Cosine` (default); one collection per embedding model |
| Offline semantic search | `go-embedding/glove` with `Dimension: glove.Dimension` (300) |
| < ~50k vectors, or exact results needed | `Index: Flat` |
| Larger collections | `Index: HNSW` (default) |
| Recall too low | raise `EfSearch` or pass `WithEf(n)` per query; no rebuild needed |
| Bulk import | `UpsertBatch` / `UpsertDocuments`, or an `embedding.Pipeline` writing via `UpsertBatch` (builds the graph on all CPUs) |
| Many deletes or updates | call `Compact()` when `Stats().Tombstones` grows large |
| Durability | call `db.Save()` periodically; `Close()` also saves |
| Several processes | one writer, any number of readers; readers reopen to see new data. A second writer's `Save` returns `ErrConflict` instead of overwriting |

The [package documentation](doc.go) covers all of this in more depth, with
runnable examples (`go doc -all github.com/brimstone/go-vector`).

## Benchmarks

Benchmarks were run with `go test -run '^$' -bench . -benchtime 2s` on an
i9-13900HK laptop under WSL2. That machine's mix of performance and
efficiency cores makes run-to-run variance large (±40%), so treat the numbers
as ballpark and run the suite on your own hardware. The search data has the
geometry of real text embeddings (a low-dimensional manifold embedded in 768
dims; see `embeddingLikeVecs`). Recall@10 is measured against exact search.

| Benchmark (n = 10,000, dim = 768) | Latency / throughput | Recall@10 |
|---|---|---|
| Distance kernel, 768-d (dot / L2) | ~180–220 ns | — |
| HNSW search, default ef = 64 | ~0.4–0.55 ms | 0.998 |
| HNSW search, ef = 16 | ~0.15 ms | 0.990 |
| HNSW search, 20 parallel goroutines | ~55 µs/query aggregate (~18k QPS) | 0.998 |
| Flat (exact) search, parallel scan | ~0.9–1 ms | 1.000 |
| Filtered search, 10% selectivity | ~0.5–0.7 ms | exact |
| Bulk build (`UpsertBatch`), 768-d | ~9–15k vectors/s | — |
| Bulk build, 1536-d | ~5–6k vectors/s | — |
| Snapshot write / read | ~2.2 GB/s / ~0.7–0.9 GB/s | — |

Flags `-benchn` and `-benchdim` change the dataset size and dimension:

```sh
go test -run '^$' -bench 'Search' -benchn 100000 -benchdim 1536
```

## Development

```sh
make check   # gofmt check + go vet + tests + race detector (library and example)
make bench   # benchmark suite
make fuzz    # fuzz the snapshot decoder
```

## Layout

| File | Contents |
|---|---|
| `collection.go` | `Collection`: upsert/get/delete/search, filters, compaction |
| `hnsw.go` | HNSW graph index, including the parallel batch builder |
| `flat.go` | exact (parallel) scan index |
| `distance.go`, `store.go`, `heap.go` | distance kernels, vector slab, top-k heaps |
| `embed.go` | `Embedder`, `UpsertDocuments`, `SearchText`, `Normalize` |
| `persist.go` | snapshot format, `WriteTo` / `ReadCollection` / `SaveFile` / `LoadFile` |
| `db.go` | `DB`: named collections in one crash-safe file |
| `lock_*.go` | OS file locking (flock on Unix, LockFileEx on Windows) |
| `examples/semantic-search` | sample host program using `go-embedding` (separate module) |

## Limitations

- In-memory with snapshot persistence; there is no write-ahead log, so
  changes since the last `Save` are lost on a crash.
- Vectors are stored as float32; there is no quantisation yet.
- Each process keeps its own in-memory copy, so concurrent writers are
  detected (`ErrConflict`) rather than merged. File locks are advisory and
  may not work on network filesystems.
- Every save rewrites the whole database file.
