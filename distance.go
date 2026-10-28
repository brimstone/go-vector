package vector

import "math"

// Metric is the distance function used by a collection. It is fixed when the
// collection is created.
//
// For text embeddings use Cosine (the default) unless your model's
// documentation says otherwise.
type Metric uint8

const (
	// Cosine ranks by the angle between vectors, ignoring their length.
	// Reported distances are 1 - cosine similarity, in [0, 2]. This is the
	// right choice for almost all text-embedding models.
	Cosine Metric = iota
	// Euclidean ranks by straight-line (L2) distance.
	Euclidean
	// DotProduct ranks by inner product, largest first. Reported distances
	// are the negated inner product. Use it for models trained for maximum
	// inner-product search, or with vectors normalised by Normalize, where
	// it ranks exactly like Cosine. HNSW needs a larger EfSearch (about 2×)
	// for the same recall as with Cosine when norms vary widely.
	DotProduct
)

// String returns "cosine", "euclidean" or "dot".
func (m Metric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case Euclidean:
		return "euclidean"
	case DotProduct:
		return "dot"
	}
	return "unknown"
}

func (m Metric) valid() bool { return m <= DotProduct }

// Dot returns the inner product of a and b. It panics if the lengths differ.
func Dot(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vector: length mismatch")
	}
	return dot(a, b)
}

// SquaredL2 returns the squared Euclidean distance between a and b. It
// panics if the lengths differ.
func SquaredL2(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vector: length mismatch")
	}
	return sqL2(a, b)
}

// CosineSimilarity returns the cosine of the angle between a and b, or 0 if
// either vector has zero length. It panics if the lengths differ.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vector: length mismatch")
	}
	na, nb := dot(a, a), dot(b, b)
	if na == 0 || nb == 0 {
		return 0
	}
	return dot(a, b) / float32(math.Sqrt(float64(na)*float64(nb)))
}

// dot is the hot inner-product kernel. The loop is unrolled 8-wide with four
// independent accumulators; the fixed-size re-slices let the compiler drop
// bounds checks inside the loop.
func dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	for len(a) >= 8 {
		aa, bb := a[:8:8], b[:8:8]
		s0 += aa[0]*bb[0] + aa[4]*bb[4]
		s1 += aa[1]*bb[1] + aa[5]*bb[5]
		s2 += aa[2]*bb[2] + aa[6]*bb[6]
		s3 += aa[3]*bb[3] + aa[7]*bb[7]
		a, b = a[8:], b[8:]
	}
	b = b[:len(a)]
	for i := range a {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// sqL2 is the squared Euclidean distance kernel.
func sqL2(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	for len(a) >= 8 {
		aa, bb := a[:8:8], b[:8:8]
		d0, d1, d2, d3 := aa[0]-bb[0], aa[1]-bb[1], aa[2]-bb[2], aa[3]-bb[3]
		d4, d5, d6, d7 := aa[4]-bb[4], aa[5]-bb[5], aa[6]-bb[6], aa[7]-bb[7]
		s0 += d0*d0 + d4*d4
		s1 += d1*d1 + d5*d5
		s2 += d2*d2 + d6*d6
		s3 += d3*d3 + d7*d7
		a, b = a[8:], b[8:]
	}
	b = b[:len(a)]
	for i := range a {
		d := a[i] - b[i]
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3)
}

// invNorm returns 1/||v||, or 0 for the zero vector.
func invNorm(v []float32) float32 {
	n := dot(v, v)
	if n == 0 {
		return 0
	}
	return float32(1 / math.Sqrt(float64(n)))
}

// finite reports whether every component of v is a finite number.
func finite(v []float32) bool {
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}
