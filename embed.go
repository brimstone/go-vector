package vector

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// Embedder turns text into embedding vectors. It is how the database is
// connected to an embedding model, whether that is a hosted API (OpenAI,
// Cohere, Voyage...), a local server (Ollama, llama.cpp, vLLM, TEI) or an
// in-process model.
//
// The package itself never calls the network. Implement Embedder in your
// program (see examples/semantic-search for an OpenAI-compatible HTTP
// implementation) and pass it to UpsertDocuments and SearchText.
//
// Embed must return exactly one vector per input text, in order, each of
// the collection's dimension. It may be called concurrently.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// EmbedderFunc adapts an ordinary function to the Embedder interface.
type EmbedderFunc func(ctx context.Context, texts []string) ([][]float32, error)

// Embed calls f(ctx, texts).
func (f EmbedderFunc) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return f(ctx, texts)
}

// Document is a piece of text to embed and store with UpsertDocuments.
type Document struct {
	// ID uniquely identifies the document (or chunk) in the collection.
	ID string
	// Content is the text to embed. It is stored and returned in results.
	Content string
	// Metadata is optional key/value data.
	Metadata Metadata
}

// DefaultEmbedBatchSize is the number of documents UpsertDocuments sends to
// the Embedder per call. Most embedding APIs accept batches of this size.
const DefaultEmbedBatchSize = 64

// ErrEmbedder is returned (wrapped) when an Embedder misbehaves, for
// example by returning the wrong number of vectors.
var ErrEmbedder = errors.New("vector: embedder error")

// UpsertDocuments embeds the documents' Content with e, in batches of
// DefaultEmbedBatchSize, and upserts them. Each batch is written as soon as
// it is embedded, so if an error occurs (including ctx cancellation) the
// batches before it have already been stored. Upserting is idempotent, so
// the call can simply be retried.
func (c *Collection) UpsertDocuments(ctx context.Context, e Embedder, docs []Document) error {
	for start := 0; start < len(docs); start += DefaultEmbedBatchSize {
		if err := ctx.Err(); err != nil {
			return err
		}
		batch := docs[start:min(start+DefaultEmbedBatchSize, len(docs))]
		texts := make([]string, len(batch))
		for i, d := range batch {
			texts[i] = d.Content
		}
		vecs, err := embedChecked(ctx, e, texts)
		if err != nil {
			return fmt.Errorf("embedding documents %d-%d: %w", start, start+len(batch)-1, err)
		}
		recs := make([]Record, len(batch))
		for i, d := range batch {
			recs[i] = Record{ID: d.ID, Vector: vecs[i], Content: d.Content, Metadata: d.Metadata}
		}
		if err := c.UpsertBatch(recs); err != nil {
			return err
		}
	}
	return nil
}

// SearchText embeds query with e and returns the k nearest records. It is
// equivalent to calling e.Embed and then Search.
func (c *Collection) SearchText(ctx context.Context, e Embedder, query string, k int, opts ...SearchOption) ([]Result, error) {
	vecs, err := embedChecked(ctx, e, []string{query})
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}
	return c.Search(vecs[0], k, opts...)
}

func embedChecked(ctx context.Context, e Embedder, texts []string) ([][]float32, error) {
	vecs, err := e.Embed(ctx, texts)
	if err != nil {
		return nil, err
	}
	if len(vecs) != len(texts) {
		return nil, fmt.Errorf("%w: got %d vectors for %d texts", ErrEmbedder, len(vecs), len(texts))
	}
	return vecs, nil
}

// Normalize scales v in place to unit length and returns it. The zero vector
// is returned unchanged.
//
// Cosine collections do not need normalised input. Normalising is useful
// before storing vectors in a DotProduct collection, which then ranks
// exactly like Cosine but skips the norm computation.
func Normalize(v []float32) []float32 {
	n := dot(v, v)
	if n == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(float64(n)))
	for i := range v {
		v[i] *= inv
	}
	return v
}
