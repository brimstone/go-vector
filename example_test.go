package vector_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/brimstone/go-vector"
)

func Example() {
	db, err := vector.Open("") // "" = memory only; pass a file path to persist
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	fruit, _ := db.CreateCollection("fruit", vector.Options{Dimension: 3})
	fruit.Upsert("apple", []float32{0.9, 0.1, 0.0}, "a crisp apple", vector.Metadata{"color": "red"})
	fruit.Upsert("cherry", []float32{0.8, 0.2, 0.1}, "a sweet cherry", vector.Metadata{"color": "red"})
	fruit.Upsert("banana", []float32{0.1, 0.9, 0.0}, "a ripe banana", vector.Metadata{"color": "yellow"})
	fruit.Upsert("lemon", []float32{0.0, 0.8, 0.3}, "a sour lemon", vector.Metadata{"color": "yellow"})

	res, _ := fruit.Search([]float32{1, 0, 0}, 2)
	for _, r := range res {
		fmt.Printf("%s %.3f %q\n", r.ID, r.Similarity(), r.Content)
	}
	// Output:
	// apple 0.994 "a crisp apple"
	// cherry 0.963 "a sweet cherry"
}

// toyEmbedder stands in for a real embedding model. It maps each text to
// counts of a few keywords. In a real program, call your model here (see
// examples/semantic-search for an OpenAI-compatible HTTP client).
var toyEmbedder = vector.EmbedderFunc(func(_ context.Context, texts []string) ([][]float32, error) {
	keywords := []string{"go", "python", "bread", "oven", "planet"}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, len(keywords))
		for j, k := range keywords {
			v[j] = float32(strings.Count(strings.ToLower(t), k))
		}
		out[i] = v
	}
	return out, nil
})

// Index text through an Embedder and search it with a natural-language
// query.
func ExampleCollection_SearchText() {
	ctx := context.Background()
	coll, _ := vector.NewCollection(vector.Options{Dimension: 5})

	err := coll.UpsertDocuments(ctx, toyEmbedder, []vector.Document{
		{ID: "go-intro", Content: "Go is a compiled language; Go has goroutines.", Metadata: vector.Metadata{"lang": "en"}},
		{ID: "py-intro", Content: "Python is an interpreted language.", Metadata: vector.Metadata{"lang": "en"}},
		{ID: "baking", Content: "Bake bread in a hot oven.", Metadata: vector.Metadata{"lang": "en"}},
	})
	if err != nil {
		log.Fatal(err)
	}

	res, _ := coll.SearchText(ctx, toyEmbedder, "how hot should the oven be for bread?", 1)
	fmt.Println(res[0].ID, "-", res[0].Content)
	// Output:
	// baking - Bake bread in a hot oven.
}

// Restrict results with metadata filters.
func ExampleWithFilter() {
	coll, _ := vector.NewCollection(vector.Options{Dimension: 2, Index: vector.Flat})
	coll.Upsert("a", []float32{1, 0}, "", vector.Metadata{"tenant": "acme", "kind": "faq"})
	coll.Upsert("b", []float32{0.9, 0.1}, "", vector.Metadata{"tenant": "globex", "kind": "faq"})
	coll.Upsert("c", []float32{0.5, 0.5}, "", vector.Metadata{"tenant": "globex", "kind": "manual"})

	res, _ := coll.Search([]float32{1, 0}, 10, vector.WithFilter(
		vector.And(vector.Where("tenant", "globex"), vector.WhereIn("kind", "faq", "manual")),
	))
	for _, r := range res {
		fmt.Println(r.ID)
	}
	// Output:
	// b
	// c
}

// Re-index a document: remove its old chunks, then insert the new ones.
func ExampleCollection_DeleteWhere() {
	coll, _ := vector.NewCollection(vector.Options{Dimension: 2})
	for i := 0; i < 3; i++ {
		coll.Upsert(fmt.Sprintf("guide.md#%d", i), []float32{1, float32(i)}, "old text", vector.Metadata{"path": "guide.md"})
	}
	coll.Upsert("other.md#0", []float32{0, 1}, "", vector.Metadata{"path": "other.md"})

	removed := coll.DeleteWhere(vector.Where("path", "guide.md"))
	coll.Upsert("guide.md#0", []float32{1, 1}, "new text", vector.Metadata{"path": "guide.md"})
	fmt.Println("removed", removed, "now", coll.Len())
	// Output:
	// removed 3 now 2
}

// Persist collections in a single database file and reopen them.
func ExampleOpen() {
	dir, _ := os.MkdirTemp("", "vector-example")
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "notes.vdb")

	db, _ := vector.Open(path)
	notes, _ := db.GetOrCreateCollection("notes", vector.Options{Dimension: 2})
	notes.Upsert("n1", []float32{1, 2}, "remember the milk", nil)
	if err := db.Close(); err != nil { // Close saves; so does db.Save()
		log.Fatal(err)
	}

	db2, _ := vector.Open(path)
	defer db2.Close()
	notes2, _ := db2.Collection("notes")
	r, _ := notes2.Get("n1")
	fmt.Println(db2.Collections(), r.Content)
	// Output:
	// [notes] remember the milk
}

// Measure HNSW recall against an exact search on your own data.
func ExampleWithExact() {
	coll, _ := vector.NewCollection(vector.Options{Dimension: 2})
	for i := 0; i < 100; i++ {
		coll.Upsert(fmt.Sprint(i), []float32{float32(i), 1}, "", nil)
	}
	q := []float32{42, 1}
	approx, _ := coll.Search(q, 5)
	exact, _ := coll.Search(q, 5, vector.WithExact())
	hits := 0
	for _, a := range approx {
		for _, e := range exact {
			if a.ID == e.ID {
				hits++
			}
		}
	}
	fmt.Printf("recall@5 = %.2f\n", float64(hits)/5)
	// Output:
	// recall@5 = 1.00
}

// Handle a save that would overwrite another process's changes.
func ExampleDB_Save() {
	dir, _ := os.MkdirTemp("", "vector-example")
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "shared.vdb")

	setup, _ := vector.Open(path)
	setup.CreateCollection("c", vector.Options{Dimension: 2})
	setup.Close()

	a, _ := vector.Open(path) // e.g. two processes sharing one file
	b, _ := vector.Open(path)
	ca, _ := a.Collection("c")
	cb, _ := b.Collection("c")
	ca.Upsert("x", []float32{1, 0}, "", nil)
	cb.Upsert("y", []float32{0, 1}, "", nil)

	fmt.Println("a:", a.Save())
	err := b.Save()
	fmt.Println("b conflict:", errors.Is(err, vector.ErrConflict))
	// b must reopen to see a's data, then re-apply its own change.
	b.Close()
	// Output:
	// a: <nil>
	// b conflict: true
}
