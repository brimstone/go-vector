package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/brimstone/go-embedding"
)

func TestOpenAIEmbedder(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" || r.Header.Get("Authorization") != "Bearer k" {
			http.Error(w, "bad request", 400)
			return
		}
		var req struct{ Input []string }
		json.NewDecoder(r.Body).Decode(&req)
		type item struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}
		var data []item
		for i := len(req.Input) - 1; i >= 0; i-- { // out of order on purpose
			data = append(data, item{i, []float32{float32(i), 1}})
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer srv.Close()

	e := &OpenAIEmbedder{BaseURL: srv.URL + "/v1/", Model: "m", APIKey: "k"}
	vecs, err := e.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vecs {
		if v[0] != float32(i) {
			t.Fatalf("vecs[%d]=%v", i, v)
		}
	}
	e.APIKey = "wrong"
	if _, err := e.Embed(context.Background(), []string{"a"}); err == nil {
		t.Fatal("expected error")
	}
}

// TestOpenAIEmbedderHandler checks that the client and the serve command's
// embedding.Handler interoperate: vectors fetched over HTTP must match the
// local embedder's.
func TestOpenAIEmbedderHandler(t *testing.T) {
	local := newHashEmbedder()
	srv := httptest.NewServer(&embedding.Handler{Models: map[string]embedding.Embedder{"hash-512": local}})
	defer srv.Close()

	ctx := context.Background()
	texts := []string{"goroutines and channels", "sourdough starter"}
	remote, err := (&OpenAIEmbedder{BaseURL: srv.URL, Model: "hash-512"}).Embed(ctx, texts)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := local.Embed(ctx, texts)
	for i := range texts {
		if !slices.Equal(remote[i], want[i]) {
			t.Fatalf("text %d: remote and local vectors differ", i)
		}
	}
	if _, err := (&OpenAIEmbedder{BaseURL: srv.URL, Model: "nope"}).Embed(ctx, texts); err == nil {
		t.Fatal("expected error for unknown model")
	}
}

// TestGloVeParaphrase checks that the default embedder matches by meaning:
// the query shares no content words with the document it should find.
func TestGloVeParaphrase(t *testing.T) {
	ctx := context.Background()
	emb, coll, err := localEmbedder("glove")
	if err != nil {
		t.Fatal(err)
	}
	a, err := openApp(ctx, filepath.Join(t.TempDir(), "glove.vdb"), coll, emb)
	if err != nil {
		t.Fatal(err)
	}
	defer a.db.Close()
	if a.coll.Dimension() != 300 {
		t.Fatalf("dimension %d", a.coll.Dimension())
	}
	for p, text := range demoCorpus {
		if _, err := a.indexText(ctx, p, text); err != nil {
			t.Fatal(err)
		}
	}
	res, err := a.coll.SearchText(ctx, a.emb, "making a loaf in the kitchen", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].Metadata["path"] != "demo/bread.md" {
		t.Fatalf("top hit %+v", res)
	}
}

func TestDemoEndToEnd(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "test.vdb")
	a, err := openApp(ctx, path, "c", newHashEmbedder())
	if err != nil {
		t.Fatal(err)
	}
	for p, text := range demoCorpus {
		if _, err := a.indexText(ctx, p, text); err != nil {
			t.Fatal(err)
		}
	}
	res, _ := a.coll.SearchText(ctx, a.emb, "goroutines and channels", 1)
	if len(res) == 0 || res[0].Metadata["path"] != "demo/go.md" {
		t.Fatalf("top hit %+v", res)
	}
	// Re-indexing must replace, not duplicate, a file's chunks.
	before := a.coll.Len()
	a.indexText(ctx, "demo/go.md", "Go is short now.\n\n"+strings.Repeat("filler ", 200))
	a.indexText(ctx, "demo/go.md", "Go is short now.")
	if after := a.coll.Len(); after != before {
		t.Fatalf("re-index: %d -> %d chunks", before, after)
	}
	r, _ := a.coll.Get("demo/go.md#0")
	if r.Content != "Go is short now." {
		t.Fatalf("content not replaced: %q", r.Content)
	}
	if r.Metadata[embedding.MetaDocID] != "demo/go.md" || r.Metadata[embedding.MetaChunk] != "0" {
		t.Fatalf("pipeline metadata missing: %v", r.Metadata)
	}
	if _, dup := r.Metadata[embedding.MetaText]; dup {
		t.Fatal("chunk text stored in metadata as well as Content")
	}
	if err := a.db.Close(); err != nil {
		t.Fatal(err)
	}
	// And everything survives a restart.
	b, err := openApp(ctx, path, "c", newHashEmbedder())
	if err != nil {
		t.Fatal(err)
	}
	if b.coll.Len() != a.coll.Len() {
		t.Fatalf("reload: %d != %d", b.coll.Len(), a.coll.Len())
	}
}
