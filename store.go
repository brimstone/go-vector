package vector

// store holds every vector slot of a collection in one contiguous slab,
// which keeps scans cache friendly and avoids per-vector allocations.
// Slot numbers are dense uint32s shared with the index.
type store struct {
	dim    int
	metric Metric
	data   []float32
	inv    []float32 // 1/||v|| per slot (used by Cosine)
	maxSq  float32   // largest squared norm seen (used by DotProduct)
}

func newStore(dim int, m Metric) *store { return &store{dim: dim, metric: m} }

func (s *store) len() int { return len(s.inv) }

func (s *store) vec(id uint32) []float32 {
	o := int(id) * s.dim
	return s.data[o : o+s.dim : o+s.dim]
}

func (s *store) add(v []float32) uint32 {
	id := uint32(len(s.inv))
	s.data = append(s.data, v...)
	s.inv = append(s.inv, invNorm(v))
	s.maxSq = max(s.maxSq, dot(v, v))
	return id
}

// dist returns the internal distance (smaller is closer) between the query q
// (with inverse norm qinv) and slot id. For Euclidean this is the squared
// distance; callers convert with toExternal.
func (s *store) dist(q []float32, qinv float32, id uint32) float32 {
	v := s.vec(id)
	switch s.metric {
	case Cosine:
		return 1 - dot(q, v)*qinv*s.inv[id]
	case Euclidean:
		return sqL2(q, v)
	default:
		return -dot(q, v)
	}
}

// buildDist is the distance used to construct the HNSW graph. It equals dist
// except for DotProduct, where the inner product is not a metric (a vector
// need not be its own nearest neighbour) and building on it directly lets a
// few large-norm vectors become hubs. Instead we use the MIPS-to-L2
// reduction (Bachrach et al. 2014): each vector is implicitly extended with
// one extra coordinate sqrt(Φ² - |x|²), Φ being the largest norm. Euclidean
// order in that space equals inner-product order for (zero-extended)
// queries, so the graph neighbourhoods match what queries rank by.
func (s *store) buildDist(q []float32, qinv float32, id uint32) float32 {
	if s.metric != DotProduct {
		return s.dist(q, qinv, id)
	}
	d := s.aug(qinv) - s.aug(s.inv[id])
	return sqL2(q, s.vec(id)) + d*d
}

// aug returns the augmented coordinate sqrt(Φ² - |x|²) given 1/|x|.
func (s *store) aug(inv float32) float32 {
	var sq float32
	if inv != 0 {
		sq = 1 / (inv * inv)
	}
	return sqrt32(max(s.maxSq-sq, 0))
}

// toExternal converts an internal distance into the value reported to users.
func (s *store) toExternal(d float32) float32 {
	switch s.metric {
	case Cosine:
		if d < 0 { // rounding
			return 0
		}
		return d
	case Euclidean:
		return sqrt32(d)
	default:
		return d
	}
}
