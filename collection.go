package vector

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"sync"
)

// IndexType selects the search index of a collection.
type IndexType uint8

const (
	// HNSW is an approximate nearest-neighbour graph index and the default.
	// Search cost grows roughly logarithmically with collection size, and
	// recall is typically 0.95-0.99 at the default settings. Choose it for
	// anything beyond a few tens of thousands of vectors.
	HNSW IndexType = iota

	// Flat performs an exact brute-force scan on every query. It needs no
	// build time and no extra memory, and always returns the true nearest
	// neighbours. Large scans are split across all CPUs. Choose it for small
	// collections (up to about 50k vectors) or when exact results matter more
	// than latency.
	Flat
)

// String returns "hnsw" or "flat".
func (t IndexType) String() string {
	switch t {
	case HNSW:
		return "hnsw"
	case Flat:
		return "flat"
	}
	return "unknown"
}

// Errors returned by Collection methods. Returned errors may wrap these with
// extra context; test for them with errors.Is.
var (
	// ErrDimensionMismatch is returned when a vector's length differs from
	// the collection's Dimension. With text embeddings this usually means
	// the query was embedded with a different model than the documents.
	ErrDimensionMismatch = errors.New("vector: dimension mismatch")

	// ErrInvalidVector is returned for vectors containing NaN or ±Inf.
	ErrInvalidVector = errors.New("vector: vector contains NaN or Inf")

	// ErrInvalidOptions is returned for invalid collection options.
	ErrInvalidOptions = errors.New("vector: invalid options")

	// ErrEmptyID is returned when a record has an empty ID.
	ErrEmptyID = errors.New("vector: empty id")

	// ErrFull is returned when a collection would exceed 2³²-1 slots. Slots
	// include tombstones; Compact frees them.
	ErrFull = errors.New("vector: collection is full")
)

// Options configures a new collection.
type Options struct {
	// Dimension is the length of every vector, which is fixed by your
	// embedding model (for example 384 for all-MiniLM-L6-v2, 768 for
	// nomic-embed-text, 1536 for OpenAI text-embedding-3-small). Required;
	// must be in [1, 65536].
	Dimension int

	// Metric is the distance function. The zero value is Cosine, which is
	// correct for virtually all text-embedding models.
	Metric Metric

	// Index is the index type. The zero value is HNSW.
	Index IndexType

	// HNSW tunes the HNSW index. Ignored for Flat. Zero fields take their
	// defaults.
	HNSW HNSWConfig
}

func (o Options) validate() error {
	if o.Dimension <= 0 || o.Dimension > 1<<16 {
		return fmt.Errorf("%w: dimension %d out of range [1, 65536]", ErrInvalidOptions, o.Dimension)
	}
	if !o.Metric.valid() {
		return fmt.Errorf("%w: unknown metric %d", ErrInvalidOptions, o.Metric)
	}
	if o.Index > Flat {
		return fmt.Errorf("%w: unknown index type %d", ErrInvalidOptions, o.Index)
	}
	return nil
}

// Metadata is arbitrary string key/value data stored alongside a vector and
// returned with search results. Use it for filtering (see Where) and for
// whatever your application needs to act on a hit: source URL, title, chunk
// number, tenant ID and so on.
type Metadata map[string]string

// Record is a stored vector together with its payload.
type Record struct {
	// ID uniquely identifies the record within its collection. Upserting an
	// existing ID replaces the record. Must not be empty.
	ID string

	// Vector is the embedding. Its length must equal the collection's
	// Dimension.
	Vector []float32

	// Content is optional free text stored with the vector, typically the
	// chunk of text that was embedded, so that search results can be used
	// directly (for example as RAG context) without a second lookup.
	Content string

	// Metadata is optional key/value data.
	Metadata Metadata
}

// Result is a single search hit. Content and Metadata are copies and may be
// modified freely.
type Result struct {
	// ID of the matching record.
	ID string

	// Distance from the query; smaller is closer. Its meaning depends on the
	// collection's Metric:
	//   - Cosine: 1 - cosine similarity, in [0, 2]. Similarity is
	//     1 - Distance; see Result.Similarity.
	//   - Euclidean: L2 distance, ≥ 0.
	//   - DotProduct: the negated inner product.
	Distance float32

	// Content of the matching record.
	Content string

	// Metadata of the matching record.
	Metadata Metadata
}

// Similarity returns 1 - Distance. For Cosine collections this is the cosine
// similarity in [-1, 1]; higher is more similar. It is a convenience for
// relevance thresholds such as "only use hits with similarity ≥ 0.75".
func (r Result) Similarity() float32 { return 1 - r.Distance }

// Filter restricts search results to records for which it returns true. It
// is given the record's ID and metadata.
//
// A Filter is called while the collection is read-locked, possibly from
// several goroutines at once, and possibly for many records per query, so it
// must be fast, safe for concurrent use, must not modify meta, and must not
// call back into the collection.
type Filter func(id string, meta Metadata) bool

// Where returns a Filter that accepts records whose metadata contains key
// with exactly the given value.
func Where(key, value string) Filter {
	return func(_ string, m Metadata) bool {
		v, ok := m[key]
		return ok && v == value
	}
}

// WhereIn returns a Filter that accepts records whose metadata value for key
// is one of values.
func WhereIn(key string, values ...string) Filter {
	set := make(map[string]struct{}, len(values))
	for _, v := range values {
		set[v] = struct{}{}
	}
	return func(_ string, m Metadata) bool {
		v, ok := m[key]
		if !ok {
			return false
		}
		_, ok = set[v]
		return ok
	}
}

// And returns a Filter that accepts records accepted by every filter.
func And(filters ...Filter) Filter {
	return func(id string, m Metadata) bool {
		for _, f := range filters {
			if !f(id, m) {
				return false
			}
		}
		return true
	}
}

type searchConfig struct {
	filter Filter
	ef     int
	exact  bool
}

// SearchOption customises a single search.
type SearchOption func(*searchConfig)

// WithFilter restricts results to records accepted by f.
//
// On HNSW collections the collection estimates how many records f accepts
// by sampling. Very selective filters (for example a single tenant's
// documents in a large shared collection) are answered with an exact scan of
// the matching records; broad ones are applied while walking the graph.
// Either way only matching records are returned, and up to k of them are
// returned whenever at least k exist.
func WithFilter(f Filter) SearchOption { return func(c *searchConfig) { c.filter = f } }

// WithEf overrides HNSWConfig.EfSearch for one query. Higher values give
// better recall and cost more time. Ignored by Flat collections.
func WithEf(ef int) SearchOption { return func(c *searchConfig) { c.ef = ef } }

// WithExact forces an exact brute-force search even on an HNSW collection.
// It is useful for measuring the recall of the approximate index on your own
// data, and for small, high-stakes queries.
func WithExact() SearchOption { return func(c *searchConfig) { c.exact = true } }

type index interface {
	// addBatch indexes the n store slots starting at first.
	addBatch(first uint32, n int)
	search(q []float32, qinv float32, k, ef int, accept func(uint32) bool) []candidate
}

// Collection is a set of vectors of one dimension, with their IDs, content
// and metadata, and a search index.
//
// A Collection is safe for concurrent use by multiple goroutines. Searches
// and reads run in parallel with each other; writes (Upsert, Delete,
// Compact) are serialised and wait for in-flight searches to finish.
//
// A Collection lives in memory. Persist it with WriteTo or SaveFile, or let
// a DB manage it.
type Collection struct {
	mu       sync.RWMutex
	opts     Options
	s        *store
	idx      index
	ids      []string   // slot -> external id
	content  []string   // slot -> content
	meta     []Metadata // slot -> metadata
	deleted  []bool     // slot -> tombstone
	nDeleted int
	byID     map[string]uint32
	gen      uint64 // bumped on every mutation; used by DB.Save
}

// NewCollection creates an empty in-memory collection.
func NewCollection(opts Options) (*Collection, error) {
	if err := opts.validate(); err != nil {
		return nil, err
	}
	opts.HNSW = opts.HNSW.withDefaults()
	c := &Collection{opts: opts, byID: make(map[string]uint32)}
	c.resetIndex()
	return c, nil
}

func (c *Collection) resetIndex() {
	c.s = newStore(c.opts.Dimension, c.opts.Metric)
	if c.opts.Index == Flat {
		c.idx = &flatIndex{s: c.s}
	} else {
		c.idx = newHNSW(c.s, c.opts.HNSW)
	}
	c.ids, c.content, c.meta, c.deleted, c.nDeleted = nil, nil, nil, nil, 0
}

// Options returns the collection's options, with defaults filled in.
func (c *Collection) Options() Options { return c.opts }

// Dimension returns the vector dimension.
func (c *Collection) Dimension() int { return c.opts.Dimension }

// Len returns the number of live (non-deleted) records.
func (c *Collection) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.byID)
}

func (c *Collection) check(v []float32) error {
	if len(v) != c.opts.Dimension {
		return fmt.Errorf("%w: got %d, want %d", ErrDimensionMismatch, len(v), c.opts.Dimension)
	}
	if !finite(v) {
		return ErrInvalidVector
	}
	return nil
}

func (c *Collection) checkRecord(r Record) error {
	if r.ID == "" {
		return ErrEmptyID
	}
	if err := c.check(r.Vector); err != nil {
		return fmt.Errorf("record %q: %w", r.ID, err)
	}
	return nil
}

// Upsert inserts a record, or replaces the record with the same id. The
// vector, content and metadata are copied, so the caller may reuse them.
//
// To insert many records, UpsertBatch is much faster: it takes the lock once
// and builds the HNSW graph in parallel.
func (c *Collection) Upsert(id string, vec []float32, content string, meta Metadata) error {
	return c.UpsertBatch([]Record{{ID: id, Vector: vec, Content: content, Metadata: meta}})
}

// UpsertBatch inserts or replaces many records at once. Every record is
// validated before anything is written, so on error the collection is
// unchanged. If the batch contains the same ID more than once, the last
// occurrence wins.
//
// Large batches (a few hundred records or more) are inserted into the HNSW
// graph by one goroutine per CPU. Searches wait while a batch is inserted, so
// split very large imports into batches of, say, 10k records if the
// collection must stay responsive.
//
// Replacing a record whose vector is unchanged only updates its content and
// metadata, which is cheap. Replacing the vector leaves a tombstone; see
// Compact.
func (c *Collection) UpsertBatch(recs []Record) error {
	for _, r := range recs {
		if err := c.checkRecord(r); err != nil {
			return err
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if uint64(c.s.len())+uint64(len(recs)) > math.MaxUint32 {
		return ErrFull
	}
	first := c.s.len()
	for _, r := range recs {
		c.stage(r.ID, r.Vector, r.Content, maps.Clone(r.Metadata))
	}
	if n := c.s.len() - first; n > 0 {
		c.idx.addBatch(uint32(first), n)
	}
	c.gen++
	return nil
}

// stage writes a record to the store without indexing it. The caller must
// index the new slots with idx.addBatch.
func (c *Collection) stage(id string, vec []float32, content string, meta Metadata) {
	if old, ok := c.byID[id]; ok {
		if slices.Equal(c.s.vec(old), vec) {
			c.content[old], c.meta[old] = content, meta
			return
		}
		c.tombstone(old)
	}
	slot := c.s.add(vec)
	c.ids = append(c.ids, id)
	c.content = append(c.content, content)
	c.meta = append(c.meta, meta)
	c.deleted = append(c.deleted, false)
	c.byID[id] = slot
}

func (c *Collection) tombstone(slot uint32) {
	c.deleted[slot] = true
	c.content[slot], c.meta[slot] = "", nil
	c.nDeleted++
}

// Delete removes the record with the given id and reports whether it
// existed. The vector's slot becomes a tombstone until Compact runs.
func (c *Collection) Delete(id string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.deleteLocked(id) {
		return false
	}
	c.gen++
	return true
}

func (c *Collection) deleteLocked(id string) bool {
	slot, ok := c.byID[id]
	if !ok {
		return false
	}
	delete(c.byID, id)
	c.tombstone(slot)
	return true
}

// DeleteWhere removes every record accepted by f and returns how many were
// removed. For example, to remove all chunks of a re-indexed document:
//
//	coll.DeleteWhere(vector.Where("doc_id", "handbook.pdf"))
func (c *Collection) DeleteWhere(f Filter) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for slot, id := range c.ids {
		if !c.deleted[slot] && f(id, c.meta[slot]) {
			c.deleteLocked(id)
			n++
		}
	}
	if n > 0 {
		c.gen++
	}
	return n
}

// Get returns a copy of the record with the given id.
func (c *Collection) Get(id string) (Record, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	slot, ok := c.byID[id]
	if !ok {
		return Record{}, false
	}
	return c.record(slot), true
}

func (c *Collection) record(slot uint32) Record {
	return Record{
		ID:       c.ids[slot],
		Vector:   slices.Clone(c.s.vec(slot)),
		Content:  c.content[slot],
		Metadata: maps.Clone(c.meta[slot]),
	}
}

// Range calls fn for every live record, in insertion order, until fn returns
// false. The collection is read-locked during iteration, so fn must not
// modify it (doing so deadlocks). Records passed to fn are copies.
func (c *Collection) Range(fn func(Record) bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for slot := range c.ids {
		if !c.deleted[slot] && !fn(c.record(uint32(slot))) {
			return
		}
	}
}

// Search returns up to k records nearest to query, closest first.
//
// query must have the collection's dimension and must come from the same
// embedding model as the stored vectors. For Cosine collections neither the
// query nor the stored vectors need to be normalised.
//
// Fewer than k results are returned only when fewer than k live records
// match the filter (if any).
func (c *Collection) Search(query []float32, k int, opts ...SearchOption) ([]Result, error) {
	if err := c.check(query); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, nil
	}
	var cfg searchConfig
	for _, o := range opts {
		o(&cfg)
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	var accept func(uint32) bool
	switch {
	case cfg.filter != nil:
		f := cfg.filter
		accept = func(id uint32) bool { return !c.deleted[id] && f(c.ids[id], c.meta[id]) }
	case c.nDeleted > 0:
		accept = func(id uint32) bool { return !c.deleted[id] }
	}

	idx := c.idx
	if _, isHNSW := idx.(*hnswIndex); isHNSW && (cfg.exact || (cfg.filter != nil && c.preferScan(accept, k, cfg.ef))) {
		idx = &flatIndex{s: c.s}
	}
	cands := idx.search(query, invNorm(query), k, cfg.ef, accept)

	out := make([]Result, len(cands))
	for i, cd := range cands {
		out[i] = Result{
			ID:       c.ids[cd.id],
			Distance: c.s.toExternal(cd.dist),
			Content:  c.content[cd.id],
			Metadata: maps.Clone(c.meta[cd.id]),
		}
	}
	return out, nil
}

// preferScan decides whether a filtered query on an HNSW collection is
// cheaper as an exact scan. It estimates the filter's selectivity s from a
// strided sample, then compares rough costs:
//
//	scan: n filter calls + s*n distances
//	graph: ~visits/s nodes, each a filter call and a distance
//
// where visits ≈ 20*ef is the typical unfiltered visit count. A filter call
// is assumed to cost about a quarter of a distance computation.
func (c *Collection) preferScan(accept func(uint32) bool, k, ef int) bool {
	n := c.s.len()
	const samples = 256
	step := max(1, n/samples)
	tried, hits := 0, 0
	for i := 0; i < n; i += step {
		tried++
		if accept(uint32(i)) {
			hits++
		}
	}
	if hits == 0 {
		return true
	}
	if ef <= 0 {
		ef = c.opts.HNSW.EfSearch
	}
	ef = max(ef, k)
	s := float64(hits) / float64(tried)
	scan := float64(n) * (s + 0.25)
	graph := 1.25 * 20 * float64(ef) / s
	return scan < graph
}

// Stats describes a collection's storage.
type Stats struct {
	// Live is the number of records that searches can return.
	Live int
	// Tombstones is the number of deleted or replaced slots still held in
	// memory. Compact reclaims them.
	Tombstones int
	// Dimension is the vector dimension.
	Dimension int
	// VectorBytes is the memory used by raw vector data, including
	// tombstones (4 bytes per component).
	VectorBytes int
}

// Stats returns storage statistics.
func (c *Collection) Stats() Stats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return Stats{
		Live:        len(c.byID),
		Tombstones:  c.nDeleted,
		Dimension:   c.opts.Dimension,
		VectorBytes: len(c.s.data) * 4,
	}
}

// Compact rebuilds the collection without tombstones, reclaiming the memory
// left behind by Delete and by Upserts that replaced a vector. For HNSW this
// rebuilds the graph (in parallel), which costs about as much as
// re-inserting every live record. Searches wait while it runs.
//
// Tombstones never affect correctness, only memory and a little speed. A
// reasonable policy is to compact when Stats().Tombstones exceeds about a
// quarter of Stats().Live.
func (c *Collection) Compact() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.nDeleted == 0 {
		return
	}
	oldS, oldIDs, oldContent, oldMeta, oldDel := c.s, c.ids, c.content, c.meta, c.deleted
	c.resetIndex()
	c.byID = make(map[string]uint32, len(oldIDs)-c.nDeleted)
	for i, id := range oldIDs {
		if !oldDel[i] {
			c.stage(id, oldS.vec(uint32(i)), oldContent[i], oldMeta[i])
		}
	}
	c.idx.addBatch(0, c.s.len())
	c.gen++
}
