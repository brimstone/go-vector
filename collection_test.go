package vector

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"
)

func TestOptionsValidation(t *testing.T) {
	bad := []Options{
		{Dimension: 0},
		{Dimension: -1},
		{Dimension: 1 << 17},
		{Dimension: 4, Metric: 99},
		{Dimension: 4, Index: 99},
	}
	for _, o := range bad {
		if _, err := NewCollection(o); !errors.Is(err, ErrInvalidOptions) {
			t.Errorf("NewCollection(%+v) err=%v, want ErrInvalidOptions", o, err)
		}
	}
}

func TestUpsertValidation(t *testing.T) {
	c := mustCollection(t, Options{Dimension: 3})
	if err := c.Upsert("a", []float32{1, 2}, "", nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("short vector: %v", err)
	}
	if err := c.Upsert("a", []float32{1, float32(math.NaN()), 3}, "", nil); !errors.Is(err, ErrInvalidVector) {
		t.Errorf("NaN: %v", err)
	}
	if err := c.Upsert("a", []float32{1, float32(math.Inf(1)), 3}, "", nil); !errors.Is(err, ErrInvalidVector) {
		t.Errorf("Inf: %v", err)
	}
	if err := c.Upsert("", []float32{1, 2, 3}, "", nil); !errors.Is(err, ErrEmptyID) {
		t.Errorf("empty id: %v", err)
	}
	// A bad record anywhere in a batch must leave the collection untouched.
	err := c.UpsertBatch([]Record{{ID: "ok", Vector: []float32{1, 2, 3}}, {ID: "bad", Vector: []float32{1}}})
	if err == nil || c.Len() != 0 {
		t.Errorf("batch err=%v len=%d, want error and 0", err, c.Len())
	}
	if _, err := c.Search([]float32{1}, 1); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("search dim: %v", err)
	}
}

func TestCRUD(t *testing.T) {
	for _, it := range []IndexType{HNSW, Flat} {
		t.Run(it.String(), func(t *testing.T) {
			c := mustCollection(t, Options{Dimension: 2, Metric: Euclidean, Index: it})
			meta := Metadata{"k": "v"}
			vec := []float32{1, 2}
			if err := c.Upsert("a", vec, "", meta); err != nil {
				t.Fatal(err)
			}
			// Mutating inputs after Upsert must not affect stored data.
			vec[0], meta["k"] = 100, "changed"

			r, ok := c.Get("a")
			if !ok || !slices.Equal(r.Vector, []float32{1, 2}) || r.Metadata["k"] != "v" {
				t.Fatalf("Get = %+v, %v", r, ok)
			}
			// Mutating outputs must not affect stored data either.
			r.Vector[0], r.Metadata["k"] = 100, "changed"
			if r2, _ := c.Get("a"); r2.Vector[0] != 1 || r2.Metadata["k"] != "v" {
				t.Fatalf("Get returned aliased data")
			}

			// Metadata-only update.
			c.Upsert("a", []float32{1, 2}, "", Metadata{"k": "w"})
			if r, _ := c.Get("a"); r.Metadata["k"] != "w" {
				t.Fatalf("metadata not updated: %+v", r)
			}
			if s := c.Stats(); s.Tombstones != 0 {
				t.Fatalf("metadata-only update created tombstone: %+v", s)
			}

			// Vector update.
			c.Upsert("a", []float32{5, 5}, "", nil)
			c.Upsert("b", []float32{0, 0}, "", nil)
			res, _ := c.Search([]float32{5, 5}, 10)
			if len(res) != 2 || res[0].ID != "a" || res[0].Distance != 0 {
				t.Fatalf("after update: %+v", res)
			}
			if c.Len() != 2 {
				t.Fatalf("Len=%d", c.Len())
			}

			if !c.Delete("a") || c.Delete("a") {
				t.Fatal("Delete semantics")
			}
			if _, ok := c.Get("a"); ok {
				t.Fatal("Get after delete")
			}
			res, _ = c.Search([]float32{5, 5}, 10)
			if len(res) != 1 || res[0].ID != "b" {
				t.Fatalf("search after delete: %+v", res)
			}
			if s := c.Stats(); s.Live != 1 || s.Tombstones != 2 {
				t.Fatalf("Stats=%+v", s)
			}
		})
	}
}

func TestSearchEdgeCases(t *testing.T) {
	c := mustCollection(t, Options{Dimension: 4})
	if res, err := c.Search([]float32{1, 0, 0, 0}, 5); err != nil || len(res) != 0 {
		t.Fatalf("empty collection: %v %v", res, err)
	}
	fill(t, c, randVecs(1, 3, 4), nil)
	if res, _ := c.Search([]float32{1, 0, 0, 0}, 0); len(res) != 0 {
		t.Fatalf("k=0 returned %d", len(res))
	}
	if res, _ := c.Search([]float32{1, 0, 0, 0}, 50); len(res) != 3 {
		t.Fatalf("k>n returned %d", len(res))
	}
	// Zero vectors are allowed under cosine (distance 1 to everything).
	if err := c.Upsert("zero", []float32{0, 0, 0, 0}, "", nil); err != nil {
		t.Fatal(err)
	}
	res, _ := c.Search([]float32{0, 0, 0, 0}, 10)
	for _, r := range res {
		if r.Distance != 1 {
			t.Fatalf("zero query distance=%v", r.Distance)
		}
	}
}

// Flat search must match a float64 reference exactly for every metric.
func TestFlatExact(t *testing.T) {
	vecs := randVecs(7, 500, 37)
	queries := randVecs(8, 20, 37)
	for _, m := range allMetrics {
		t.Run(m.String(), func(t *testing.T) {
			c := mustCollection(t, Options{Dimension: 37, Metric: m, Index: Flat})
			fill(t, c, vecs, nil)
			for _, q := range queries {
				res, err := c.Search(q, 10)
				if err != nil {
					t.Fatal(err)
				}
				want := bruteForce(m, vecs, q, 10, nil)
				for i := range want {
					if res[i].ID != want[i] {
						t.Fatalf("rank %d: got %s want %s", i, res[i].ID, want[i])
					}
					var vi int
					fmtSscan(res[i].ID, &vi)
					wd := naiveDist(m, q, vecs[vi])
					if math.Abs(float64(res[i].Distance)-wd) > 1e-4*(1+math.Abs(wd)) {
						t.Fatalf("distance %v want %v", res[i].Distance, wd)
					}
				}
				if !slices.IsSortedFunc(res, func(a, b Result) int { return cmpF(a.Distance, b.Distance) }) {
					t.Fatal("results not sorted")
				}
			}
		})
	}
}

// Large flat collections take the parallel scan path.
func TestFlatParallelScan(t *testing.T) {
	n := parallelScanFloats/8 + 1000
	vecs := randVecs(3, n, 8)
	c := mustCollection(t, Options{Dimension: 8, Metric: Euclidean, Index: Flat})
	fill(t, c, vecs, nil)
	q := randVecs(4, 1, 8)[0]
	res, _ := c.Search(q, 25)
	want := bruteForce(Euclidean, vecs, q, 25, nil)
	if r := recall(want, res); r != 1 {
		t.Fatalf("parallel flat recall=%v", r)
	}
}

func TestFilter(t *testing.T) {
	vecs := clusteredVecs(11, 2000, 16, 10)
	meta := func(i int) Metadata {
		if i%7 == 0 {
			return Metadata{"kind": "rare"}
		}
		return Metadata{"kind": "common"}
	}
	for _, it := range []IndexType{HNSW, Flat} {
		t.Run(it.String(), func(t *testing.T) {
			c := mustCollection(t, Options{Dimension: 16, Index: it})
			fill(t, c, vecs, meta)
			var total float64
			queries := randVecs(12, 20, 16)
			for _, q := range queries {
				res, err := c.Search(q, 10, WithFilter(Where("kind", "rare")))
				if err != nil {
					t.Fatal(err)
				}
				if len(res) != 10 {
					t.Fatalf("got %d results", len(res))
				}
				for _, r := range res {
					if r.Metadata["kind"] != "rare" {
						t.Fatalf("filter leaked %+v", r)
					}
				}
				total += recall(bruteForce(Cosine, vecs, q, 10, func(i int) bool { return i%7 != 0 }), res)
			}
			if avg := total / float64(len(queries)); avg < 0.95 {
				t.Fatalf("filtered recall %.3f", avg)
			}
		})
	}
}

func TestFilterMatchingNothing(t *testing.T) {
	c := mustCollection(t, Options{Dimension: 4})
	fill(t, c, randVecs(1, 200, 4), nil)
	res, _ := c.Search([]float32{1, 2, 3, 4}, 5, WithFilter(func(string, Metadata) bool { return false }))
	if len(res) != 0 {
		t.Fatalf("got %d", len(res))
	}
}

func TestRange(t *testing.T) {
	c := mustCollection(t, Options{Dimension: 2})
	fill(t, c, randVecs(1, 10, 2), nil)
	c.Delete(id(3))
	var seen []string
	c.Range(func(r Record) bool { seen = append(seen, r.ID); return true })
	if len(seen) != 9 || slices.Contains(seen, id(3)) {
		t.Fatalf("Range saw %v", seen)
	}
	n := 0
	c.Range(func(Record) bool { n++; return n < 4 })
	if n != 4 {
		t.Fatalf("early stop: %d", n)
	}
}

func TestCompact(t *testing.T) {
	for _, it := range []IndexType{HNSW, Flat} {
		t.Run(it.String(), func(t *testing.T) {
			vecs := randVecs(5, 1000, 16)
			c := mustCollection(t, Options{Dimension: 16, Index: it})
			fill(t, c, vecs, func(i int) Metadata { return Metadata{"i": id(i)} })
			for i := 0; i < len(vecs); i += 2 {
				c.Delete(id(i))
			}
			c.Compact()
			if s := c.Stats(); s.Live != 500 || s.Tombstones != 0 {
				t.Fatalf("Stats=%+v", s)
			}
			q := randVecs(6, 1, 16)[0]
			res, _ := c.Search(q, 10)
			want := bruteForce(Cosine, vecs, q, 10, func(i int) bool { return i%2 == 0 })
			if r := recall(want, res); r < 0.9 {
				t.Fatalf("recall after compact %.2f", r)
			}
			for _, r := range res {
				if r.Metadata["i"] != r.ID {
					t.Fatalf("metadata mismatch after compact: %+v", r)
				}
			}
			if r, ok := c.Get(id(1)); !ok || !slices.Equal(r.Vector, vecs[1]) {
				t.Fatal("Get after compact")
			}
		})
	}
}

// Run with -race (requires cgo for the race runtime) for full value; without
// it this still checks for deadlocks and panics under concurrent use.
func TestConcurrentAccess(t *testing.T) {
	c := mustCollection(t, Options{Dimension: 16})
	fill(t, c, randVecs(1, 500, 16), nil)
	queries := randVecs(2, 50, 16)
	extra := randVecs(3, 500, 16)
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i, q := range queries {
				if _, err := c.Search(q, 5, WithFilter(Where("x", "y"))); err != nil {
					t.Error(err)
				}
				if _, err := c.Search(q, 5); err != nil {
					t.Error(err)
				}
				c.Get(id(i))
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i, v := range extra {
			c.Upsert(id(1000+i), v, "", Metadata{"x": "y"})
			if i%10 == 0 {
				c.Delete(id(i))
			}
			if i == 250 {
				c.Compact()
			}
		}
	}()
	wg.Wait()
	if c.Len() != 500+500-50 {
		t.Fatalf("Len=%d", c.Len())
	}
}

func TestContent(t *testing.T) {
	c := mustCollection(t, Options{Dimension: 2})
	c.Upsert("a", []float32{1, 0}, "hello world", nil)
	if r, _ := c.Get("a"); r.Content != "hello world" {
		t.Fatalf("Get content %q", r.Content)
	}
	res, _ := c.Search([]float32{1, 0}, 1)
	if res[0].Content != "hello world" {
		t.Fatalf("Search content %q", res[0].Content)
	}
	if s := res[0].Similarity(); s < 0.9999 {
		t.Fatalf("Similarity=%v", s)
	}
	// Same vector, new content: in-place update, no tombstone.
	c.Upsert("a", []float32{1, 0}, "bye", nil)
	if r, _ := c.Get("a"); r.Content != "bye" || c.Stats().Tombstones != 0 {
		t.Fatalf("content update: %q %+v", r.Content, c.Stats())
	}
}

func TestFilterCombinators(t *testing.T) {
	m := Metadata{"lang": "go", "tier": "gold"}
	cases := []struct {
		f    Filter
		want bool
	}{
		{Where("lang", "go"), true},
		{Where("lang", "rust"), false},
		{Where("missing", ""), false},
		{WhereIn("tier", "silver", "gold"), true},
		{WhereIn("tier", "bronze"), false},
		{WhereIn("missing"), false},
		{And(Where("lang", "go"), WhereIn("tier", "gold")), true},
		{And(Where("lang", "go"), Where("tier", "silver")), false},
		{And(), true},
	}
	for i, tc := range cases {
		if got := tc.f("id", m); got != tc.want {
			t.Errorf("case %d: got %v", i, got)
		}
	}
}

func TestDeleteWhere(t *testing.T) {
	c := mustCollection(t, Options{Dimension: 4})
	fill(t, c, randVecs(1, 100, 4), func(i int) Metadata {
		return Metadata{"doc": fmt.Sprint(i % 4)}
	})
	if n := c.DeleteWhere(Where("doc", "2")); n != 25 {
		t.Fatalf("deleted %d", n)
	}
	if c.Len() != 75 {
		t.Fatalf("Len=%d", c.Len())
	}
	res, _ := c.Search([]float32{1, 1, 1, 1}, 100)
	for _, r := range res {
		if r.Metadata["doc"] == "2" {
			t.Fatal("deleted record returned")
		}
	}
}

// Both strategies of the filtered-search planner must give exact or
// near-exact answers: narrow filters (scan) and broad filters (graph).
func TestFilterPlanner(t *testing.T) {
	vecs := clusteredVecs(91, 20000, 16, 30)
	c := mustCollection(t, Options{Dimension: 16})
	fill(t, c, vecs, func(i int) Metadata { return Metadata{"m": fmt.Sprint(i % 1000)} })
	for _, tc := range []struct {
		name     string
		filter   Filter
		keep     func(int) bool
		wantScan bool
	}{
		{"narrow", Where("m", "7"), func(i int) bool { return i%1000 == 7 }, true},
		{"broad", func(_ string, m Metadata) bool { return m["m"] < "5" }, func(i int) bool { return fmt.Sprint(i%1000) < "5" }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := tc.filter
			accept := func(id uint32) bool { return f(c.ids[id], c.meta[id]) }
			if got := c.preferScan(accept, 10, 0); got != tc.wantScan {
				t.Fatalf("preferScan=%v, want %v", got, tc.wantScan)
			}
			var total float64
			queries := clusteredVecs(92, 20, 16, 30)
			for _, q := range queries {
				// ef=200: the synthetic clusters are tightly separated, which
				// is hard for graph search; the narrow case is exact anyway.
				res, _ := c.Search(q, 10, WithFilter(f), WithEf(200))
				total += recall(bruteForce(Cosine, vecs, q, 10, func(i int) bool { return !tc.keep(i) }), res)
			}
			if avg := total / float64(len(queries)); avg < 0.95 {
				t.Fatalf("recall %.3f", avg)
			}
		})
	}
}

func TestWithExact(t *testing.T) {
	vecs := randVecs(5, 3000, 64)
	c := mustCollection(t, Options{Dimension: 64, Metric: Euclidean})
	fill(t, c, vecs, nil)
	for _, q := range randVecs(6, 10, 64) {
		res, _ := c.Search(q, 10, WithExact(), WithEf(1))
		if r := recall(bruteForce(Euclidean, vecs, q, 10, nil), res); r != 1 {
			t.Fatalf("exact recall %v", r)
		}
	}
}
