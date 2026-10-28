package vector

import (
	"bytes"
	"flag"
	"fmt"
	"sync"
	"testing"
)

// Benchmark dataset size and dimension can be changed on the command line:
//
//	go test -run '^$' -bench . -benchn 100000 -benchdim 1536
//
// The default dimension, 768, is typical of text-embedding models.
var (
	benchN   = flag.Int("benchn", 10000, "number of vectors in search benchmarks")
	benchDim = flag.Int("benchdim", 768, "vector dimension in search benchmarks")
)

type fixture struct {
	vecs, queries [][]float32
	truth         [][]string // exact top-10 per query
	c             *Collection
}

var (
	fixMu    sync.Mutex
	fixtures = map[string]*fixture{}
)

// getFixture builds (once per process) a collection plus exact ground truth.
// The data is embeddingLikeVecs, so recall figures resemble what real text
// embeddings achieve; the unit tests cover harder synthetic distributions.
func getFixture(b *testing.B, it IndexType, m Metric) *fixture {
	b.Helper()
	key := fmt.Sprintf("%v/%v/%d/%d", it, m, *benchN, *benchDim)
	fixMu.Lock()
	defer fixMu.Unlock()
	if f, ok := fixtures[key]; ok {
		return f
	}
	f := &fixture{
		vecs:    embeddingLikeVecs(100, *benchN, *benchDim),
		queries: embeddingLikeVecs(101, 200, *benchDim),
	}
	f.c = mustCollection(b, Options{Dimension: *benchDim, Metric: m, Index: it})
	fill(b, f.c, f.vecs, func(i int) Metadata {
		return Metadata{"bucket": fmt.Sprint(i % 10)}
	})
	for _, q := range f.queries {
		f.truth = append(f.truth, bruteForce(m, f.vecs, q, 10, nil))
	}
	fixtures[key] = f
	return f
}

func BenchmarkDistance(b *testing.B) {
	for _, dim := range []int{128, 384, 768, 1536} {
		v := randVecs(1, 2, dim)
		x, y := v[0], v[1]
		b.Run(fmt.Sprintf("dot/dim=%d", dim), func(b *testing.B) {
			b.SetBytes(int64(dim * 8))
			var s float32
			for i := 0; i < b.N; i++ {
				s += dot(x, y)
			}
			_ = s
		})
		b.Run(fmt.Sprintf("l2/dim=%d", dim), func(b *testing.B) {
			b.SetBytes(int64(dim * 8))
			var s float32
			for i := 0; i < b.N; i++ {
				s += sqL2(x, y)
			}
			_ = s
		})
	}
}

// BenchmarkUpsert measures single-record upserts into a growing collection
// (the slow path; prefer UpsertBatch).
func BenchmarkUpsert(b *testing.B) {
	const pool = 20000
	for _, it := range []IndexType{Flat, HNSW} {
		b.Run(fmt.Sprintf("%v/dim=%d", it, *benchDim), func(b *testing.B) {
			vecs := randVecs(2, pool, *benchDim)
			c := mustCollection(b, Options{Dimension: *benchDim, Index: it})
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if i > 0 && i%pool == 0 { // keep graph size bounded
					b.StopTimer()
					c = mustCollection(b, Options{Dimension: *benchDim, Index: it})
					b.StartTimer()
				}
				c.Upsert(id(i), vecs[i%pool], "", nil)
			}
		})
	}
}

// BenchmarkBuild measures bulk indexing with UpsertBatch, which builds the
// HNSW graph in parallel, at common text-embedding dimensions.
func BenchmarkBuild(b *testing.B) {
	const n = 5000
	for _, dim := range []int{384, 768, 1536} {
		vecs := embeddingLikeVecs(3, n, dim)
		recs := make([]Record, n)
		for i, v := range vecs {
			recs[i] = Record{ID: id(i), Vector: v}
		}
		b.Run(fmt.Sprintf("hnsw/n=%d/dim=%d", n, dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				c := mustCollection(b, Options{Dimension: dim})
				if err := c.UpsertBatch(recs); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(n*b.N)/b.Elapsed().Seconds(), "vectors/s")
		})
	}
}

func BenchmarkSearch(b *testing.B) {
	for _, it := range []IndexType{Flat, HNSW} {
		for _, m := range []Metric{Cosine, Euclidean} {
			name := fmt.Sprintf("%v/%v/n=%d/dim=%d", it, m, *benchN, *benchDim)
			b.Run(name, func(b *testing.B) {
				f := getFixture(b, it, m)
				b.ResetTimer()
				var rec float64
				for i := 0; i < b.N; i++ {
					qi := i % len(f.queries)
					res, _ := f.c.Search(f.queries[qi], 10)
					rec += recall(f.truth[qi], res)
				}
				b.ReportMetric(rec/float64(b.N), "recall@10")
			})
		}
	}
}

// BenchmarkSearchEf shows the HNSW recall/latency trade-off.
func BenchmarkSearchEf(b *testing.B) {
	f := getFixture(b, HNSW, Cosine)
	for _, ef := range []int{16, 32, 64, 128, 256} {
		b.Run(fmt.Sprintf("ef=%d", ef), func(b *testing.B) {
			var rec float64
			for i := 0; i < b.N; i++ {
				qi := i % len(f.queries)
				res, _ := f.c.Search(f.queries[qi], 10, WithEf(ef))
				rec += recall(f.truth[qi], res)
			}
			b.ReportMetric(rec/float64(b.N), "recall@10")
		})
	}
}

func BenchmarkSearchFiltered(b *testing.B) {
	for _, it := range []IndexType{Flat, HNSW} {
		b.Run(it.String()+"/10pct", func(b *testing.B) {
			f := getFixture(b, it, Cosine)
			flt := WithFilter(Where("bucket", "3"))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				f.c.Search(f.queries[i%len(f.queries)], 10, flt)
			}
		})
	}
}

func BenchmarkSearchParallel(b *testing.B) {
	for _, it := range []IndexType{Flat, HNSW} {
		b.Run(it.String(), func(b *testing.B) {
			f := getFixture(b, it, Cosine)
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					f.c.Search(f.queries[i%len(f.queries)], 10)
					i++
				}
			})
		})
	}
}

func BenchmarkSnapshot(b *testing.B) {
	f := getFixture(b, HNSW, Cosine)
	var buf bytes.Buffer
	f.c.WriteTo(&buf)
	data := buf.Bytes()
	b.Run("write", func(b *testing.B) {
		b.SetBytes(int64(len(data)))
		for i := 0; i < b.N; i++ {
			buf.Reset()
			f.c.WriteTo(&buf)
		}
	})
	b.Run("read", func(b *testing.B) {
		b.SetBytes(int64(len(data)))
		for i := 0; i < b.N; i++ {
			if _, err := ReadCollection(bytes.NewReader(data)); err != nil {
				b.Fatal(err)
			}
		}
	})
}
