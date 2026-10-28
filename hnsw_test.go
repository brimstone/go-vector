package vector

import (
	"slices"
	"testing"
)

func hnswRecall(t *testing.T, m Metric, vecs, queries [][]float32, k int, opts ...SearchOption) float64 {
	t.Helper()
	c := mustCollection(t, Options{Dimension: len(vecs[0]), Metric: m})
	fill(t, c, vecs, nil)
	var total float64
	for _, q := range queries {
		res, err := c.Search(q, k, opts...)
		if err != nil {
			t.Fatal(err)
		}
		total += recall(bruteForce(m, vecs, q, k, nil), res)
	}
	return total / float64(len(queries))
}

func TestHNSWRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	vecs := clusteredVecs(21, 5000, 32, 50)
	queries := clusteredVecs(22, 100, 32, 50)
	for _, m := range allMetrics {
		t.Run(m.String(), func(t *testing.T) {
			var opts []SearchOption
			if m == DotProduct {
				// MIPS is intrinsically harder for graph indexes; at the
				// default ef=64 recall is ~0.90, at 128 it is ~0.98.
				opts = append(opts, WithEf(128))
			}
			r := hnswRecall(t, m, vecs, queries, 10, opts...)
			t.Logf("recall@10 = %.4f", r)
			if r < 0.95 {
				t.Fatalf("recall@10 = %.4f, want >= 0.95", r)
			}
		})
	}
}

func TestHNSWRecallUniform(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	// Uniform gaussian data is the hard case for graph indexes.
	vecs := randVecs(31, 5000, 32)
	queries := randVecs(32, 100, 32)
	r := hnswRecall(t, Euclidean, vecs, queries, 10, WithEf(128))
	t.Logf("recall@10 = %.4f", r)
	if r < 0.95 {
		t.Fatalf("recall@10 = %.4f, want >= 0.95", r)
	}
}

func TestHNSWEfImprovesRecall(t *testing.T) {
	vecs := randVecs(41, 3000, 48)
	queries := randVecs(42, 50, 48)
	lo := hnswRecall(t, Euclidean, vecs, queries, 10, WithEf(10))
	hi := hnswRecall(t, Euclidean, vecs, queries, 10, WithEf(300))
	t.Logf("ef=10: %.3f  ef=300: %.3f", lo, hi)
	if hi < lo || hi < 0.98 {
		t.Fatalf("ef=10: %.3f  ef=300: %.3f", lo, hi)
	}
}

func TestHNSWDeletesNeverReturned(t *testing.T) {
	vecs := clusteredVecs(51, 3000, 16, 20)
	c := mustCollection(t, Options{Dimension: 16})
	fill(t, c, vecs, nil)
	deleted := func(i int) bool { return i%3 != 0 } // delete two thirds
	for i := range vecs {
		if deleted(i) {
			c.Delete(id(i))
		}
	}
	var total float64
	queries := clusteredVecs(52, 50, 16, 20)
	for _, q := range queries {
		res, _ := c.Search(q, 10)
		if len(res) != 10 {
			t.Fatalf("got %d results", len(res))
		}
		for _, r := range res {
			var i int
			fmtSscan(r.ID, &i)
			if deleted(i) {
				t.Fatalf("deleted id %s returned", r.ID)
			}
		}
		total += recall(bruteForce(Cosine, vecs, q, 10, deleted), res)
	}
	if avg := total / float64(len(queries)); avg < 0.95 {
		t.Fatalf("recall with tombstones %.3f", avg)
	}
}

func TestHNSWDeterministic(t *testing.T) {
	vecs := randVecs(61, 500, 8)
	build := func() *hnswIndex {
		c := mustCollection(t, Options{Dimension: 8, HNSW: HNSWConfig{Seed: 99}})
		for i, v := range vecs { // one at a time: batches build in parallel
			c.Upsert(id(i), v, "", nil)
		}
		return c.idx.(*hnswIndex)
	}
	a, b := build(), build()
	if a.entry != b.entry || a.maxLevel != b.maxLevel {
		t.Fatal("entry/maxLevel differ")
	}
	for i := range a.links {
		for l := range a.links[i] {
			if !slices.Equal(a.links[i][l], b.links[i][l]) {
				t.Fatalf("node %d level %d differs", i, l)
			}
		}
	}
}

func TestHNSWGraphInvariants(t *testing.T) {
	c := mustCollection(t, Options{Dimension: 8, HNSW: HNSWConfig{M: 4}})
	fill(t, c, randVecs(71, 2000, 8), nil)
	h := c.idx.(*hnswIndex)
	if err := h.validate(); err != nil {
		t.Fatal(err)
	}
	for i, node := range h.links {
		for l, nb := range node {
			if len(nb) > h.maxConn(l) {
				t.Fatalf("node %d level %d has %d links (max %d)", i, l, len(nb), h.maxConn(l))
			}
			for _, x := range nb {
				if x == uint32(i) {
					t.Fatalf("self link at node %d", i)
				}
			}
		}
	}
}

// Batches are inserted by parallel workers; the graph must be just as good
// as a sequential build, and structurally valid.
func TestHNSWParallelBuild(t *testing.T) {
	vecs := clusteredVecs(81, 8000, 32, 40)
	queries := clusteredVecs(82, 100, 32, 40)
	par := mustCollection(t, Options{Dimension: 32})
	fill(t, par, vecs, nil) // one batch -> parallel
	seq := mustCollection(t, Options{Dimension: 32})
	for i := 0; i < len(vecs); i += 50 { // batches too small to parallelise
		recs := make([]Record, 0, 50)
		for j := i; j < i+50; j++ {
			recs = append(recs, Record{ID: id(j), Vector: vecs[j]})
		}
		seq.UpsertBatch(recs)
	}
	if err := par.idx.(*hnswIndex).validate(); err != nil {
		t.Fatal(err)
	}
	var rp, rs float64
	for _, q := range queries {
		want := bruteForce(Cosine, vecs, q, 10, nil)
		a, _ := par.Search(q, 10, WithEf(128))
		b, _ := seq.Search(q, 10, WithEf(128))
		rp += recall(want, a)
		rs += recall(want, b)
	}
	rp, rs = rp/float64(len(queries)), rs/float64(len(queries))
	t.Logf("recall@10 parallel=%.4f sequential=%.4f", rp, rs)
	if rp < 0.97 || rp < rs-0.02 {
		t.Fatalf("parallel build recall %.4f (sequential %.4f)", rp, rs)
	}
}
