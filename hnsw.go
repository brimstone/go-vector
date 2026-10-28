package vector

import (
	"cmp"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
)

// HNSWConfig tunes the HNSW index. The zero value selects the defaults,
// which suit typical text-embedding workloads (384-3072 dimensions, up to a
// few million vectors).
//
// Rules of thumb:
//   - Raise EfSearch (or pass WithEf per query) when recall is too low; it
//     is the cheapest knob and needs no rebuild.
//   - Raise M (to 24-48) for very high-dimensional or hard-to-separate
//     data; memory grows by roughly 8*M bytes per vector.
//   - Raise EfConstruction for a better graph at the cost of slower
//     inserts. It must be at least M to be meaningful.
type HNSWConfig struct {
	// M is the maximum number of links each node keeps on the upper layers;
	// layer 0 keeps 2*M. Default 16.
	M int

	// EfConstruction is the size of the candidate list used when inserting.
	// Default 200.
	EfConstruction int

	// EfSearch is the default size of the candidate list used when searching.
	// Larger values improve recall and cost latency. It is always raised to
	// at least k. Default 64.
	EfSearch int

	// Seed seeds the random level assignment. Graphs built one vector at a
	// time are fully deterministic for a given seed. Large batches are built
	// in parallel, which makes the graph (but not its quality)
	// non-deterministic. Default 1.
	Seed uint64
}

func (c HNSWConfig) withDefaults() HNSWConfig {
	if c.M <= 0 {
		c.M = 16
	}
	if c.EfConstruction <= 0 {
		c.EfConstruction = 200
	}
	if c.EfSearch <= 0 {
		c.EfSearch = 64
	}
	if c.Seed == 0 {
		c.Seed = 1
	}
	return c
}

const (
	maxHNSWLevel = 16
	lockStripes  = 1024 // per-node link locks used during parallel builds
	minPerWorker = 64   // smallest share of a batch worth a build goroutine
)

// hnswIndex is a Hierarchical Navigable Small World graph (Malkov &
// Yashunin, 2016) over the slots of a store. Deleted slots stay in the graph
// as waypoints; the collection filters them from results.
//
// Concurrency: the owning Collection serialises writers against readers, so
// searches never race with inserts. Within one addBatch call, inserts run in
// parallel and coordinate through stripes (node links) and global (entry
// point).
type hnswIndex struct {
	s        *store
	cfg      HNSWConfig
	m, m0    int
	ml       float64
	links    [][][]uint32 // links[node][layer] -> neighbour slots
	entry    uint32
	maxLevel int // -1 when empty
	rng      *rand.Rand
	pool     sync.Pool // *searchCtx for queries
	seq      builder   // scratch space for sequential inserts

	global  sync.Mutex
	stripes [lockStripes]sync.Mutex
}

func newHNSW(s *store, cfg HNSWConfig) *hnswIndex {
	cfg = cfg.withDefaults()
	return &hnswIndex{
		s:        s,
		cfg:      cfg,
		m:        cfg.M,
		m0:       2 * cfg.M,
		ml:       1 / math.Log(float64(max(cfg.M, 2))),
		maxLevel: -1,
		rng:      rand.New(rand.NewPCG(cfg.Seed, 0x9e3779b97f4a7c15)),
	}
}

func (h *hnswIndex) maxConn(level int) int {
	if level == 0 {
		return h.m0
	}
	return h.m
}

func (h *hnswIndex) randomLevel() int {
	l := int(-math.Log(1-h.rng.Float64()) * h.ml)
	return min(l, maxHNSWLevel)
}

// phase selects the distance function and locking used by a graph walk.
type phase uint8

const (
	querying         phase = iota // query distance, no locks
	building                      // build distance, no locks
	buildingParallel              // build distance, lock neighbour lists
)

// visitedSet is an O(1)-reset visited marker using generation counters.
type visitedSet struct {
	marks []uint32
	gen   uint32
}

func (v *visitedSet) reset(n int) {
	if len(v.marks) < n {
		v.marks = make([]uint32, n+n/2+64)
		v.gen = 0
	}
	v.gen++
	if v.gen == 0 {
		clear(v.marks)
		v.gen = 1
	}
}

// visit marks id and reports whether it had already been visited.
func (v *visitedSet) visit(id uint32) bool {
	if v.marks[id] == v.gen {
		return true
	}
	v.marks[id] = v.gen
	return false
}

// searchCtx is per-walk scratch space.
type searchCtx struct {
	vis  visitedSet
	cand minHeap
	res  maxHeap
	nb   []uint32 // neighbour-list copy (parallel builds only)
}

// builder is per-goroutine scratch space for inserts.
type builder struct {
	ctx                      searchCtx
	eps, w, sel, prune, sel2 []candidate
}

func (h *hnswIndex) d(p phase, q []float32, qinv float32, id uint32) float32 {
	if p == querying {
		return h.s.dist(q, qinv, id)
	}
	return h.s.buildDist(q, qinv, id)
}

// neighbors returns the links of id on layer l. During parallel builds it
// returns a copy taken under the node's lock.
func (h *hnswIndex) neighbors(ctx *searchCtx, id uint32, l int, p phase) []uint32 {
	if p != buildingParallel {
		return h.links[id][l]
	}
	mu := &h.stripes[id%lockStripes]
	mu.Lock()
	ctx.nb = append(ctx.nb[:0], h.links[id][l]...)
	mu.Unlock()
	return ctx.nb
}

func (h *hnswIndex) lock(id uint32, p phase) {
	if p == buildingParallel {
		h.stripes[id%lockStripes].Lock()
	}
}

func (h *hnswIndex) unlock(id uint32, p phase) {
	if p == buildingParallel {
		h.stripes[id%lockStripes].Unlock()
	}
}

func (h *hnswIndex) newNode() [][]uint32 {
	nl := make([][]uint32, h.randomLevel()+1)
	for l := range nl {
		nl[l] = make([]uint32, 0, h.maxConn(l)+1)
	}
	return nl
}

func (h *hnswIndex) add(id uint32) { h.addBatch(id, 1) }

// addBatch indexes the n slots starting at first, which must be the next
// unindexed slots. Levels are drawn sequentially (so the RNG stream is
// deterministic); large batches are then inserted by parallel workers.
func (h *hnswIndex) addBatch(first uint32, n int) {
	if int(first) != len(h.links) {
		panic("vector: hnsw slots must be added in order")
	}
	for i := 0; i < n; i++ {
		h.links = append(h.links, h.newNode())
	}
	start := 0
	if h.maxLevel < 0 && n > 0 {
		h.insert(&h.seq, first, building)
		start = 1
	}
	workers := min(runtime.GOMAXPROCS(0), (n-start)/minPerWorker)
	if workers <= 1 {
		for i := start; i < n; i++ {
			h.insert(&h.seq, first+uint32(i), building)
		}
		return
	}
	var next atomic.Int64
	next.Store(int64(start))
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			b := &builder{}
			for {
				i := next.Add(1) - 1
				if i >= int64(n) {
					return
				}
				h.insert(b, first+uint32(i), buildingParallel)
			}
		}()
	}
	wg.Wait()
}

// insert links node id (whose level lists already exist) into the graph
// (algorithm 1 of the paper).
func (h *hnswIndex) insert(b *builder, id uint32, p phase) {
	level := len(h.links[id]) - 1

	// Snapshot the entry point. A node that will become the new top of the
	// graph holds the global lock for its whole insertion, as in hnswlib.
	parallel := p == buildingParallel
	if parallel {
		h.global.Lock()
	}
	entry, maxLevel := h.entry, h.maxLevel
	if maxLevel < 0 {
		h.entry, h.maxLevel = id, level
		if parallel {
			h.global.Unlock()
		}
		return
	}
	hold := parallel && level > maxLevel
	if parallel && !hold {
		h.global.Unlock()
	}

	q, qinv := h.s.vec(id), h.s.inv[id]
	ep := candidate{entry, h.s.buildDist(q, qinv, entry)}
	for l := maxLevel; l > level; l-- {
		ep = h.greedy(&b.ctx, q, qinv, ep, l, p)
	}
	// In parallel builds other workers may already have linked to id, so the
	// walk can reach id itself; never select it as its own neighbour.
	var notSelf func(uint32) bool
	if parallel {
		notSelf = func(x uint32) bool { return x != id }
	}
	b.eps = append(b.eps[:0], ep)
	for l := min(level, maxLevel); l >= 0; l-- {
		b.ctx.vis.reset(len(h.links))
		b.w = h.searchLayer(&b.ctx, q, qinv, b.eps, h.cfg.EfConstruction, l, notSelf, p, b.w)
		b.sel = h.selectNeighbors(b.w, h.m, b.sel[:0])
		// Link both ways through connect: in parallel builds id's own list
		// may already hold back-links from other workers, so it too must be
		// de-duplicated and pruned.
		for _, c := range b.sel {
			h.connect(b, id, c.id, c.dist, l, p)
			h.connect(b, c.id, id, c.dist, l, p)
		}
		b.eps = append(b.eps[:0], b.w...)
	}
	if level > maxLevel {
		h.entry, h.maxLevel = id, level
	}
	if hold {
		h.global.Unlock()
	}
}

// connect adds the link n -> id on layer l (d is their distance), re-pruning
// n's neighbour list with the selection heuristic if it overflows.
func (h *hnswIndex) connect(b *builder, n, id uint32, d float32, l int, p phase) {
	h.lock(n, p)
	defer h.unlock(n, p)
	nb := h.links[n][l]
	if slices.Contains(nb, id) {
		return
	}
	mmax := h.maxConn(l)
	if len(nb) < mmax {
		h.links[n][l] = append(nb, id)
		return
	}
	nv, ninv := h.s.vec(n), h.s.inv[n]
	b.prune = append(b.prune[:0], candidate{id, d})
	for _, x := range nb {
		b.prune = append(b.prune, candidate{x, h.s.buildDist(nv, ninv, x)})
	}
	slices.SortFunc(b.prune, func(a, b candidate) int { return cmp.Compare(a.dist, b.dist) })
	b.sel2 = h.selectNeighbors(b.prune, mmax, b.sel2[:0])
	nb = nb[:0]
	for _, c := range b.sel2 {
		nb = append(nb, c.id)
	}
	h.links[n][l] = nb
}

// selectNeighbors implements the neighbour-selection heuristic (algorithm 4):
// a candidate is kept only if it is closer to the base node than to every
// already-selected neighbour, which spreads links across clusters and keeps
// the graph navigable. cands must be sorted ascending.
func (h *hnswIndex) selectNeighbors(cands []candidate, m int, out []candidate) []candidate {
	if len(cands) <= m {
		return append(out, cands...)
	}
	for _, c := range cands {
		if len(out) >= m {
			break
		}
		cv, cinv := h.s.vec(c.id), h.s.inv[c.id]
		keep := true
		for _, r := range out {
			if h.s.buildDist(cv, cinv, r.id) < c.dist {
				keep = false
				break
			}
		}
		if keep {
			out = append(out, c)
		}
	}
	return out
}

// greedy walks layer l towards q, returning the local minimum.
func (h *hnswIndex) greedy(ctx *searchCtx, q []float32, qinv float32, ep candidate, l int, p phase) candidate {
	for changed := true; changed; {
		changed = false
		for _, nb := range h.neighbors(ctx, ep.id, l, p) {
			if d := h.d(p, q, qinv, nb); d < ep.dist {
				ep, changed = candidate{nb, d}, true
			}
		}
	}
	return ep
}

// searchLayer is the beam search of algorithm 2. Nodes rejected by accept
// are still traversed, so filtering never disconnects the graph, but they
// are not returned. Results are written to out in ascending order. The
// caller must reset ctx.vis.
func (h *hnswIndex) searchLayer(ctx *searchCtx, q []float32, qinv float32, eps []candidate, ef, level int, accept func(uint32) bool, p phase, out []candidate) []candidate {
	cand, res := ctx.cand[:0], ctx.res[:0]
	for _, e := range eps {
		if ctx.vis.visit(e.id) {
			continue
		}
		cand.push(e)
		if accept == nil || accept(e.id) {
			res.push(e)
			if len(res) > ef {
				res.pop()
			}
		}
	}
	for len(cand) > 0 {
		c := cand.pop()
		if len(res) >= ef && c.dist > res[0].dist {
			break
		}
		for _, nb := range h.neighbors(ctx, c.id, level, p) {
			if ctx.vis.visit(nb) {
				continue
			}
			d := h.d(p, q, qinv, nb)
			if len(res) < ef || d < res[0].dist {
				cand.push(candidate{nb, d})
				if accept == nil || accept(nb) {
					res.push(candidate{nb, d})
					if len(res) > ef {
						res.pop()
					}
				}
			}
		}
	}
	out = res.drainSorted(out)
	ctx.cand, ctx.res = cand, res
	return out
}

func (h *hnswIndex) search(q []float32, qinv float32, k, ef int, accept func(uint32) bool) []candidate {
	if h.maxLevel < 0 {
		return nil
	}
	if ef <= 0 {
		ef = h.cfg.EfSearch
	}
	ef = max(ef, k)
	ctx, _ := h.pool.Get().(*searchCtx)
	if ctx == nil {
		ctx = &searchCtx{}
	}
	defer h.pool.Put(ctx)

	ep := candidate{h.entry, h.s.dist(q, qinv, h.entry)}
	for l := h.maxLevel; l > 0; l-- {
		ep = h.greedy(ctx, q, qinv, ep, l, querying)
	}
	ctx.vis.reset(len(h.links))
	out := h.searchLayer(ctx, q, qinv, []candidate{ep}, ef, 0, accept, querying, make([]candidate, 0, ef))
	if len(out) > k {
		out = out[:k]
	}
	return out
}
