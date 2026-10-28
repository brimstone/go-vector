// Package vector is an embeddable vector database written in pure Go (no
// cgo, no dependencies outside the standard library). It is designed to be
// linked into other programs to index and search text embeddings for
// semantic search, retrieval-augmented generation (RAG), deduplication,
// recommendation and clustering.
//
// # Overview
//
// A [Collection] holds vectors of one fixed dimension. Each vector has a
// unique string ID and may carry the text it was computed from (Content) and
// string key/value [Metadata]. Collections are searched for the k nearest
// neighbours of a query vector, optionally restricted by a [Filter] over
// metadata.
//
// A [DB] is a set of named collections persisted together in a single file.
// Collections can also be used on their own, in memory or persisted with
// [Collection.WriteTo] / [ReadCollection].
//
// # Quick start
//
//	db, err := vector.Open("app.vdb") // or vector.Open("") for memory only
//	if err != nil { ... }
//	defer db.Close() // saves changes
//
//	docs, err := db.GetOrCreateCollection("docs", vector.Options{
//		Dimension: 768, // must match your embedding model
//	})
//
//	err = docs.Upsert("readme#0", embedding, "chunk text…", vector.Metadata{"path": "README.md"})
//
//	hits, err := docs.Search(queryEmbedding, 5)
//	for _, h := range hits {
//		fmt.Println(h.ID, h.Similarity(), h.Content)
//	}
//
// # Text embeddings
//
// The package does not compute embeddings itself. Programs supply them,
// either directly as []float32 or by implementing [Embedder], a one-method
// interface that wraps whatever model you use: a hosted API such as OpenAI,
// a local server such as Ollama, or an in-process model. With an Embedder,
// [Collection.UpsertDocuments] embeds and stores text in batches and
// [Collection.SearchText] embeds a query and searches in one call. The
// examples/semantic-search program contains a complete OpenAI-compatible
// HTTP embedder, and shows how to chunk and embed documents with the
// companion github.com/brimstone/go-embedding package. Its glove
// subpackage provides an offline Embedder that matches by meaning, using
// bundled GloVe word vectors (300 dimensions).
//
// Guidelines for text embeddings:
//
//   - Use [Cosine], the default metric. It is what embedding models are
//     trained for, and vectors need not be pre-normalised.
//   - Use one collection per embedding model. Vectors from different models
//     (or model versions) are not comparable. A dimension mismatch is
//     reported as [ErrDimensionMismatch], but two models with the same
//     dimension will silently give meaningless results.
//   - Split long documents into passages of a few hundred tokens before
//     embedding. Give each chunk an ID such as "path#3" and store its source
//     in metadata, so a re-indexed document's old chunks can be removed with
//     [Collection.DeleteWhere].
//   - Store the chunk text as Content, so search results can be fed straight
//     to an LLM without a second lookup.
//   - Some models (E5, BGE, nomic-embed) expect different prefixes for
//     queries and documents, such as "query: " and "passage: ". Add them in
//     your Embedder.
//   - Raw similarity scores are model-specific. Calibrate any relevance
//     threshold against [Result.Similarity] on your own data.
//
// # Choosing an index
//
// [HNSW] (the default) is an approximate graph index. Search time grows
// roughly logarithmically with collection size. Recall, the fraction of the
// true nearest neighbours returned, is typically 0.95–0.99 and can be traded
// for latency per query with [WithEf] or per collection with
// [HNSWConfig.EfSearch]. Build cost is significant: insert in batches with
// [Collection.UpsertBatch], which builds the graph on all CPUs.
//
// [Flat] scans every vector on every query, so results are exact. It is
// faster than HNSW to build and, below a few tens of thousands of vectors,
// often fast enough to search. Large scans use all CPUs.
//
// [WithExact] runs an exact scan on an HNSW collection, which is useful for
// measuring recall on your own data.
//
// # Filtering
//
// [WithFilter] restricts results to records accepted by a [Filter]. [Where],
// [WhereIn] and [And] cover common cases, and any func(id string, meta
// Metadata) bool works. Filters never reduce result counts: if k matching
// records exist, k are returned. On HNSW collections, a filter that matches
// only a small fraction of records is answered by an exact scan of the
// matches, and a broad filter is applied during graph traversal. The choice
// is made per query from a sample.
//
// # Persistence and durability
//
// Collections live in memory. A [DB] persists all of them to one file using
// snapshots. [DB.Save] (also called by [DB.Close]) writes the whole database
// when anything changed. Saves are crash-safe: the new snapshot is written
// beside the current one and fsynced, and only then does a small header
// switch over to it, so a crash or power loss leaves the previous snapshot
// intact. Changes made since the last save are lost on a crash, so programs
// that ingest continuously should call Save periodically, for example after
// each batch or on a timer. Snapshots include the HNSW graph, so loading
// does not rebuild it, and they are checksummed and verified on load.
//
// # Multiple processes
//
// The database file is locked at the OS level (flock on Unix, LockFileEx on
// Windows; no cgo). Save holds an exclusive lock while it writes, so two
// processes never write the file at the same time, and Open holds a shared
// lock while reading, so it never reads a half-finished save. Since each
// process has its own in-memory copy, Save also refuses, with
// [ErrConflict], to overwrite a file that another writer saved after this
// process loaded it. The intended pattern is one writing process with any
// number of reading processes, which reopen the file to see new data. See
// [DB] for details.
//
// # Concurrency
//
// All types are safe for concurrent use by multiple goroutines. A collection allows any number of
// concurrent searches and reads. Writes are serialised and wait for
// in-flight searches. Each search runs on the calling goroutine, except
// large Flat scans, which fan out across CPUs.
//
// # Deletes, updates and compaction
//
// Deleting a record, or upserting a new vector under an existing ID, leaves
// a tombstone. Tombstoned vectors still help navigate the HNSW graph but are
// never returned. They cost memory, which [Collection.Compact] reclaims.
// [Collection.Stats] reports how many there are. Updating only Content or
// Metadata (same vector) is done in place and leaves no tombstone.
//
// # Memory
//
// Each record costs 4×Dimension bytes for its vector, plus about 8×M to 16×M
// bytes of HNSW links, plus its ID, content and metadata. One million
// 768-dimensional vectors need roughly 3.1 GB for vectors and 0.2 GB for the
// graph.
//
// # Performance notes
//
// Distance kernels are unrolled, bounds-check-free pure Go. On a modern x86
// core a 768-dimensional distance takes about 250–400 ns. Run the benchmark
// suite with
//
//	go test -run '^$' -bench . -benchdim 768 -benchn 20000
//
// to measure your own hardware and data shape. Benchmarks report
// recall@10 alongside latency.
package vector
