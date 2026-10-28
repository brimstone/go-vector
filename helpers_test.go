package vector

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

func randVecs(seed uint64, n, dim int) [][]float32 {
	r := rand.New(rand.NewPCG(seed, seed))
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		out[i] = v
	}
	return out
}

// clusteredVecs mimics real embeddings: points scattered around a few centres.
func clusteredVecs(seed uint64, n, dim, clusters int) [][]float32 {
	r := rand.New(rand.NewPCG(seed, seed+1))
	centres := randVecs(seed+2, clusters, dim)
	out := make([][]float32, n)
	for i := range out {
		c := centres[r.IntN(clusters)]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + 0.3*float32(r.NormFloat64())
		}
		out[i] = v
	}
	return out
}

// embeddingLikeVecs mimics the geometry of real text embeddings: points on a
// low-dimensional (latent) manifold, grouped into topics, linearly embedded
// in a high-dimensional space, plus a little isotropic noise. Real embedding
// sets have intrinsic dimension in the tens even when the vectors have
// hundreds or thousands of components; isotropic gaussian data in 768
// dimensions is far harder than anything a real model produces.
func embeddingLikeVecs(seed uint64, n, dim int) [][]float32 {
	const latent, topics = 32, 200
	r := rand.New(rand.NewPCG(seed, 77)) // per-call stream for the points
	pr := rand.New(rand.NewPCG(7, 7))    // fixed projection and topics, shared by all calls
	proj := make([][]float32, latent)
	for i := range proj {
		proj[i] = make([]float32, dim)
		for j := range proj[i] {
			proj[i][j] = float32(pr.NormFloat64())
		}
	}
	centres := make([][]float64, topics)
	for i := range centres {
		centres[i] = make([]float64, latent)
		for j := range centres[i] {
			centres[i][j] = pr.NormFloat64()
		}
	}
	out := make([][]float32, n)
	z := make([]float64, latent)
	for i := range out {
		c := centres[r.IntN(topics)]
		for j := range z {
			z[j] = c[j] + 0.6*r.NormFloat64()
		}
		v := make([]float32, dim)
		for j, zj := range z {
			for k, p := range proj[j] {
				v[k] += float32(zj) * p
			}
		}
		for k := range v {
			v[k] += float32(0.3 * r.NormFloat64())
		}
		out[i] = Normalize(v)
	}
	return out
}

func id(i int) string { return fmt.Sprintf("v%d", i) }

// naiveDist is an obviously-correct float64 reference implementation.
func naiveDist(m Metric, a, b []float32) float64 {
	var d, na, nb, l2 float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		d += x * y
		na += x * x
		nb += y * y
		l2 += (x - y) * (x - y)
	}
	switch m {
	case Cosine:
		if na == 0 || nb == 0 {
			return 1
		}
		return 1 - d/math.Sqrt(na*nb)
	case Euclidean:
		return math.Sqrt(l2)
	default:
		return -d
	}
}

// bruteForce returns the ids of the k nearest vectors to q, excluding any
// index for which skip returns true.
func bruteForce(m Metric, vecs [][]float32, q []float32, k int, skip func(int) bool) []string {
	type pair struct {
		i int
		d float64
	}
	var ps []pair
	for i, v := range vecs {
		if skip != nil && skip(i) {
			continue
		}
		ps = append(ps, pair{i, naiveDist(m, q, v)})
	}
	sort.Slice(ps, func(a, b int) bool { return ps[a].d < ps[b].d })
	var out []string
	for i := 0; i < k && i < len(ps); i++ {
		out = append(out, id(ps[i].i))
	}
	return out
}

func recall(want []string, got []Result) float64 {
	if len(want) == 0 {
		return 1
	}
	set := make(map[string]bool, len(want))
	for _, w := range want {
		set[w] = true
	}
	hit := 0
	for _, g := range got {
		if set[g.ID] {
			hit++
		}
	}
	return float64(hit) / float64(len(want))
}

func mustCollection(t testing.TB, opts Options) *Collection {
	t.Helper()
	c, err := NewCollection(opts)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func fill(t testing.TB, c *Collection, vecs [][]float32, meta func(int) Metadata) {
	t.Helper()
	recs := make([]Record, len(vecs))
	for i, v := range vecs {
		recs[i] = Record{ID: id(i), Vector: v, Content: "text " + id(i)}
		if meta != nil {
			recs[i].Metadata = meta(i)
		}
	}
	if err := c.UpsertBatch(recs); err != nil {
		t.Fatal(err)
	}
}

var allMetrics = []Metric{Cosine, Euclidean, DotProduct}
