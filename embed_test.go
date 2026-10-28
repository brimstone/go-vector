package vector

import (
	"context"
	"errors"
	"hash/fnv"
	"math"
	"strings"
	"sync/atomic"
	"testing"
)

// hashEmbedder is a deterministic bag-of-words embedder for tests.
type hashEmbedder struct {
	dim   int
	calls atomic.Int32
}

func (h *hashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	h.calls.Add(1)
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, h.dim)
		for _, w := range strings.Fields(strings.ToLower(t)) {
			f := fnv.New32a()
			f.Write([]byte(w))
			v[f.Sum32()%uint32(h.dim)]++
		}
		out[i] = v
	}
	return out, nil
}

func TestUpsertDocumentsAndSearchText(t *testing.T) {
	ctx := context.Background()
	e := &hashEmbedder{dim: 256}
	c := mustCollection(t, Options{Dimension: 256})
	docs := []Document{
		{ID: "go", Content: "go is a statically typed compiled language", Metadata: Metadata{"topic": "lang"}},
		{ID: "py", Content: "python is a dynamically typed interpreted language", Metadata: Metadata{"topic": "lang"}},
		{ID: "cat", Content: "the cat sat on the mat", Metadata: Metadata{"topic": "pets"}},
	}
	for i := 0; i < 200; i++ { // enough to span several embed batches
		docs = append(docs, Document{ID: id(i), Content: "filler document number " + id(i)})
	}
	if err := c.UpsertDocuments(ctx, e, docs); err != nil {
		t.Fatal(err)
	}
	if want := int32((len(docs) + DefaultEmbedBatchSize - 1) / DefaultEmbedBatchSize); e.calls.Load() != want {
		t.Fatalf("embed calls=%d want %d", e.calls.Load(), want)
	}
	if c.Len() != len(docs) {
		t.Fatalf("Len=%d", c.Len())
	}
	res, err := c.SearchText(ctx, e, "compiled typed language", 1)
	if err != nil {
		t.Fatal(err)
	}
	if res[0].ID != "go" || !strings.Contains(res[0].Content, "compiled") {
		t.Fatalf("SearchText top hit %+v", res[0])
	}
	res, _ = c.SearchText(ctx, e, "cat", 1, WithFilter(Where("topic", "lang")))
	if res[0].Metadata["topic"] != "lang" {
		t.Fatalf("filtered SearchText %+v", res[0])
	}
}

func TestEmbedderErrors(t *testing.T) {
	ctx := context.Background()
	c := mustCollection(t, Options{Dimension: 4})
	docs := []Document{{ID: "a", Content: "x"}, {ID: "b", Content: "y"}}

	short := EmbedderFunc(func(context.Context, []string) ([][]float32, error) {
		return [][]float32{{1, 2, 3, 4}}, nil
	})
	if err := c.UpsertDocuments(ctx, short, docs); !errors.Is(err, ErrEmbedder) {
		t.Fatalf("short embedder: %v", err)
	}
	wrongDim := EmbedderFunc(func(_ context.Context, in []string) ([][]float32, error) {
		return make([][]float32, len(in)), nil // zero-length vectors
	})
	if err := c.UpsertDocuments(ctx, wrongDim, docs); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("wrong dim: %v", err)
	}
	if _, err := c.SearchText(ctx, wrongDim, "q", 1); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("wrong dim query: %v", err)
	}
	boom := errors.New("boom")
	failing := EmbedderFunc(func(context.Context, []string) ([][]float32, error) { return nil, boom })
	if _, err := c.SearchText(ctx, failing, "q", 1); !errors.Is(err, boom) {
		t.Fatalf("failing embedder: %v", err)
	}
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if err := c.UpsertDocuments(cctx, short, docs); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled: %v", err)
	}
	if c.Len() != 0 {
		t.Fatalf("Len=%d after failures", c.Len())
	}
}

func TestNormalize(t *testing.T) {
	v := Normalize([]float32{3, 4})
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Fatalf("Normalize=%v", v)
	}
	z := Normalize([]float32{0, 0})
	if z[0] != 0 || z[1] != 0 {
		t.Fatalf("Normalize zero=%v", z)
	}
}
