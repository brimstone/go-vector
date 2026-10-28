// Command semantic-search shows how to embed the vector package in a program
// to index and search text by meaning, using the companion
// github.com/brimstone/go-embedding package for the text side.
//
// It stores text chunks and their embeddings in a single vector.DB file and
// answers natural-language queries against them. The file is locked while
// it is written, so several instances (say, an indexer and searchers) can
// share it safely.
//
// Documents are split with embedding.RecursiveChunker and ingested with an
// embedding.Pipeline, which embeds chunks in concurrent batches and writes
// them to the collection through a small embedding.Store adapter
// (collectionStore). Embeddings come from one of:
//
//   - glove (the default): an embedding.VectorEmbedder over the 300-dimension
//     GloVe word vectors bundled in github.com/brimstone/go-embedding/glove.
//     It works offline and matches by meaning ("automobile" finds "car"),
//     at the cost of about 32 MB of binary size. English only.
//   - hash (-embedder hash): embedding.HashEmbedder, an offline
//     feature-hashing embedder that adds nothing to the binary. It matches
//     shared words and word fragments, so it suits names, codes and typos
//     but not paraphrases.
//   - any OpenAI-compatible /v1/embeddings endpoint (OpenAI, Ollama, vLLM,
//     llama.cpp server, LM Studio, ...), selected with -embed-url.
//
// The serve command exposes both offline embedders as an OpenAI-compatible
// endpoint with embedding.Handler, so other programs (or another instance
// of this one, via -embed-url) can use them.
//
// Usage:
//
//	semantic-search [flags] demo                 # index a built-in corpus and run sample queries
//	semantic-search [flags] index FILE|DIR...    # chunk, embed and store .txt/.md files
//	semantic-search [flags] search QUERY         # search the index
//	semantic-search [flags] stats                # show collection statistics
//	semantic-search [flags] serve [ADDR]         # serve the offline embedders at ADDR/v1/embeddings
//
// Examples, run from this directory (it is a separate Go module):
//
//	go run . demo                  # GloVe: matches by meaning
//	go run . -embedder hash demo   # feature hashing: matches shared words
//
//	# With a local Ollama server:
//	ollama pull nomic-embed-text
//	go run . -embed-url http://localhost:11434/v1 \
//	    -model nomic-embed-text index ./docs
//	go run . -embed-url http://localhost:11434/v1 \
//	    -model nomic-embed-text search "how do I persist a collection?"
//
//	# With OpenAI (reads OPENAI_API_KEY):
//	go run . -embed-url https://api.openai.com/v1 \
//	    -model text-embedding-3-small search "..."
//
//	# Serve the offline embedders, then use them like any other API:
//	go run . serve localhost:8080 &
//	go run . -embed-url http://localhost:8080/v1 \
//	    -model glove-300 demo
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/brimstone/go-embedding"
	"github.com/brimstone/go-embedding/glove"
	"github.com/brimstone/go-vector"
)

// chunkSize and chunkOverlap, in runes, suit most embedding models: about
// 200 tokens per chunk, with a sentence or so repeated across boundaries.
const (
	chunkSize    = 800
	chunkOverlap = 100
)

func main() {
	dbPath := flag.String("db", "semantic-search.vdb", "database file")
	embedderName := flag.String("embedder", "glove", "offline embedder when -embed-url is empty: glove (by meaning) or hash (by shared words)")
	embedURL := flag.String("embed-url", "", "base URL of an OpenAI-compatible embeddings API (overrides -embedder)")
	model := flag.String("model", "text-embedding-3-small", "embedding model name (with -embed-url)")
	apiKey := flag.String("api-key", os.Getenv("OPENAI_API_KEY"), "API key (with -embed-url); defaults to $OPENAI_API_KEY")
	k := flag.Int("k", 5, "number of results")
	pathFilter := flag.String("path", "", "only search chunks from this file path")
	minSim := flag.Float64("min-similarity", 0, "drop results below this cosine similarity")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s [flags] demo | index FILE|DIR... | search QUERY | stats | serve [ADDR]\n\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() == 0 {
		flag.Usage()
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if flag.Arg(0) == "serve" {
		addr := "localhost:8080"
		if flag.NArg() > 1 {
			addr = flag.Arg(1)
		}
		if err := serve(addr); err != nil {
			fatal(err)
		}
		return
	}

	// One collection per model: vectors from different models are not
	// comparable, and their dimensions usually differ.
	var emb embedding.Embedder
	var collName string
	if *embedURL != "" {
		emb = &OpenAIEmbedder{BaseURL: *embedURL, Model: *model, APIKey: *apiKey}
		collName = "chunks-" + sanitize(*model)
	} else {
		var err error
		if emb, collName, err = localEmbedder(*embedderName); err != nil {
			fatal(err)
		}
	}

	app, err := openApp(ctx, *dbPath, collName, emb)
	if err != nil {
		fatal(err)
	}
	// Close saves every modified collection. Without it, changes are lost.
	defer func() {
		if err := app.db.Close(); err != nil {
			fatal(err)
		}
	}()

	switch cmd, args := flag.Arg(0), flag.Args()[1:]; cmd {
	case "demo":
		err = app.demo(ctx)
	case "index":
		err = app.index(ctx, args)
	case "search":
		var opts []vector.SearchOption
		if *pathFilter != "" {
			opts = append(opts, vector.WithFilter(vector.Where("path", *pathFilter)))
		}
		err = app.search(ctx, strings.Join(args, " "), *k, float32(*minSim), opts...)
	case "stats":
		s := app.coll.Stats()
		fmt.Printf("collection %s: %d chunks, %d tombstones, dimension %d, %.1f MiB of vectors\n",
			collName, s.Live, s.Tombstones, s.Dimension, float64(s.VectorBytes)/(1<<20))
	default:
		flag.Usage()
		os.Exit(2)
	}
	if err != nil {
		fatal(err)
	}
}

type app struct {
	db   *vector.DB
	coll *vector.Collection
	emb  embedding.Embedder
	pipe *embedding.Pipeline
}

// localEmbedder returns the named offline embedder and the collection that
// holds its vectors.
func localEmbedder(name string) (embedding.Embedder, string, error) {
	switch name {
	case "glove":
		// Decodes the bundled vectors on first use (about 0.1 s). The
		// collection name predates the package's rename from glove300; the
		// vectors are the same, so existing databases stay valid.
		e, err := glove.Embedder()
		return e, "chunks-glove300", err
	case "hash":
		return newHashEmbedder(), "chunks-hash512-v2", nil // v2: embedding.HashEmbedder, not the old built-in hasher
	}
	return nil, "", fmt.Errorf("unknown -embedder %q; use glove or hash", name)
}

// newHashEmbedder returns the feature-hashing embedder.
func newHashEmbedder() *embedding.HashEmbedder {
	e, err := embedding.NewHashEmbedder(embedding.DefaultHashOptions())
	if err != nil {
		panic(err) // the default options are valid
	}
	return e
}

func openApp(ctx context.Context, path, collName string, emb embedding.Embedder) (*app, error) {
	db, err := vector.Open(path)
	if err != nil {
		return nil, err
	}
	coll, ok := db.Collection(collName)
	if !ok {
		// The dimension is a property of the embedding model. Local
		// embedders report it; for remote ones, embed a probe string rather
		// than hard-coding it.
		dim := emb.Dimension()
		if dim == 0 {
			probe, err := emb.Embed(ctx, []string{"dimension probe"})
			if err != nil {
				return nil, fmt.Errorf("probing embedding model: %w", err)
			}
			if len(probe) != 1 || len(probe[0]) == 0 {
				return nil, errors.New("embedding model returned no vector")
			}
			dim = len(probe[0])
		}
		coll, err = db.CreateCollection(collName, vector.Options{
			Dimension: dim,
			Metric:    vector.Cosine,
		})
		if err != nil {
			return nil, err
		}
	}
	chunker, err := embedding.NewRecursiveChunker(chunkSize, chunkOverlap)
	if err != nil {
		return nil, err
	}
	pipe, err := embedding.NewPipeline(chunker, emb, embedding.PipelineOptions{Store: collectionStore{coll}})
	if err != nil {
		return nil, err
	}
	return &app{db: db, coll: coll, emb: emb, pipe: pipe}, nil
}

// index chunks every .txt / .md file under the given paths and stores the
// chunks. Re-indexing a file first deletes its old chunks, so edits and
// shrinking files leave no stale results behind.
func (a *app) index(ctx context.Context, paths []string) error {
	if len(paths) == 0 {
		return errors.New("index: no files given")
	}
	var files []string
	for _, p := range paths {
		err := filepath.WalkDir(p, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if ext := strings.ToLower(filepath.Ext(path)); !d.IsDir() && (ext == ".txt" || ext == ".md") {
				files = append(files, path)
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	start := time.Now()
	total := 0
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return err
		}
		n, err := a.indexText(ctx, filepath.ToSlash(f), string(data))
		if err != nil {
			return fmt.Errorf("%s: %w", f, err)
		}
		fmt.Printf("indexed %-50s %4d chunks\n", f, n)
		total += n
	}
	fmt.Printf("indexed %d chunks from %d files in %v\n", total, len(files), time.Since(start).Round(time.Millisecond))
	return nil
}

// indexText chunks, embeds and stores one document, using its path as the
// document ID. The pipeline gives each chunk the ID "path#N" and metadata
// with the document ID (embedding.MetaDocID), chunk index and byte offsets,
// plus the "path" key added here for -path filtering.
func (a *app) indexText(ctx context.Context, path, text string) (int, error) {
	// Re-ingesting overwrites chunks #0..#n but leaves any old trailing
	// chunks, so remove the document's previous chunks first.
	a.coll.DeleteWhere(vector.Where(embedding.MetaDocID, path))
	recs, err := a.pipe.Ingest(ctx, path, text, map[string]string{"path": path})
	return len(recs), err
}

func (a *app) search(ctx context.Context, query string, k int, minSim float32, opts ...vector.SearchOption) error {
	if strings.TrimSpace(query) == "" {
		return errors.New("search: empty query")
	}
	start := time.Now()
	res, err := a.coll.SearchText(ctx, a.emb, query, k, opts...)
	if err != nil {
		return err
	}
	fmt.Printf("%q (%v)\n", query, time.Since(start).Round(time.Microsecond))
	shown := 0
	for _, r := range res {
		if r.Similarity() < minSim {
			continue
		}
		shown++
		fmt.Printf("  %.3f  %s\n         %s\n", r.Similarity(), r.ID, preview(r.Content, 100))
	}
	if shown == 0 {
		fmt.Println("  (no results)")
	}
	return nil
}

func (a *app) demo(ctx context.Context) error {
	for path, text := range demoCorpus {
		if _, err := a.indexText(ctx, path, text); err != nil {
			return err
		}
	}
	fmt.Printf("indexed %d demo documents (%d chunks)\n\n", len(demoCorpus), a.coll.Len())
	for _, q := range []string{
		"how do I bake bread at home",
		"orbit of planets around the sun",
		"which language has goroutines and channels",
		// Shares no content words with demo/bread.md: glove should still
		// rank it first, the hash embedder will not.
		"making a loaf in the kitchen",
	} {
		if err := a.search(ctx, q, 3, 0); err != nil {
			return err
		}
	}
	fmt.Println()
	fmt.Println("filtered to one document:")
	return a.search(ctx, "compiled language", 2, 0, vector.WithFilter(vector.Where("path", "demo/go.md")))
}

// serve exposes the offline embedders over the OpenAI embeddings API at
// http://ADDR/v1/embeddings, as models "glove-300" and "hash-512".
func serve(addr string) error {
	gloveEmb, err := glove.Embedder()
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	mux.Handle("POST /v1/embeddings", &embedding.Handler{
		Models: map[string]embedding.Embedder{
			"glove-300": gloveEmb,
			"hash-512":  newHashEmbedder(),
		},
	})
	log.Printf("serving models glove-300 and hash-512 at http://%s/v1/embeddings", addr)
	return http.ListenAndServe(addr, mux)
}

func preview(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "…"
	}
	return s
}

func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' || r == '.' {
			return r
		}
		return '_'
	}, s)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

var demoCorpus = map[string]string{
	"demo/bread.md": `Baking bread at home needs flour, water, salt and yeast.
Knead the dough until smooth, let it rise until doubled, shape it, and bake in a hot oven.

A sourdough starter replaces commercial yeast with wild yeast and bacteria, giving a tangy flavour.`,
	"demo/solar.md": `The solar system has eight planets that orbit the sun.
Mercury is closest to the sun; Neptune is farthest.

Each planet's orbit is an ellipse, as described by Kepler's laws of planetary motion.`,
	"demo/go.md": `Go is a statically typed, compiled programming language designed at Google.
Its concurrency model is built on goroutines and channels.

Go compiles quickly to a single static binary, which makes deployment simple.`,
	"demo/python.md": `Python is a dynamically typed, interpreted programming language.
It is popular for data science, scripting and web development.`,
}
