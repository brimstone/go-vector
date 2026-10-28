package vector

import (
	"runtime"
	"sync"
)

// flatIndex performs exact brute-force search over the store. Adding is free:
// every slot in the store is implicitly indexed.
type flatIndex struct{ s *store }

// parallelScanFloats is the scan size (slots × dimension) above which flat
// scans are split across GOMAXPROCS goroutines: about 4 MB of vector data,
// e.g. 1.4k vectors of 768 dimensions.
const parallelScanFloats = 1 << 20

func (f *flatIndex) addBatch(uint32, int) {}

func (f *flatIndex) search(q []float32, qinv float32, k, _ int, accept func(uint32) bool) []candidate {
	n := f.s.len()
	workers := runtime.GOMAXPROCS(0)
	if n*f.s.dim < parallelScanFloats || workers == 1 {
		h := f.scan(q, qinv, k, accept, 0, n)
		return h.drainSorted(nil)
	}
	chunk := (n + workers - 1) / workers
	parts := make([]maxHeap, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*chunk, min((w+1)*chunk, n)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			parts[w] = f.scan(q, qinv, k, accept, lo, hi)
		}()
	}
	wg.Wait()
	h := make(maxHeap, 0, k)
	for _, p := range parts {
		for _, c := range p {
			pushBounded(&h, c, k)
		}
	}
	return h.drainSorted(nil)
}

func (f *flatIndex) scan(q []float32, qinv float32, k int, accept func(uint32) bool, lo, hi int) maxHeap {
	h := make(maxHeap, 0, k)
	for i := lo; i < hi; i++ {
		id := uint32(i)
		if accept != nil && !accept(id) {
			continue
		}
		pushBounded(&h, candidate{id, f.s.dist(q, qinv, id)}, k)
	}
	return h
}

// pushBounded adds c to h, keeping at most k of the closest candidates.
func pushBounded(h *maxHeap, c candidate, k int) {
	if len(*h) < k {
		h.push(c)
	} else if c.dist < (*h)[0].dist {
		h.replaceTop(c)
	}
}
