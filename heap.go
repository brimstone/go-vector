package vector

import "math"

type candidate struct {
	id   uint32
	dist float32
}

// maxHeap keeps the furthest candidate at index 0.
type maxHeap []candidate

func (h *maxHeap) push(c candidate) {
	a := append(*h, c)
	i := len(a) - 1
	for i > 0 {
		p := (i - 1) / 2
		if a[p].dist >= a[i].dist {
			break
		}
		a[p], a[i] = a[i], a[p]
		i = p
	}
	*h = a
}

func (h *maxHeap) pop() candidate {
	a := *h
	n := len(a) - 1
	top := a[0]
	a[0] = a[n]
	a = a[:n]
	a.down(0)
	*h = a
	return top
}

// replaceTop swaps the furthest element for c and restores heap order.
func (h maxHeap) replaceTop(c candidate) {
	h[0] = c
	h.down(0)
}

func (h maxHeap) down(i int) {
	n := len(h)
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		j := l
		if r := l + 1; r < n && h[r].dist > h[l].dist {
			j = r
		}
		if h[i].dist >= h[j].dist {
			return
		}
		h[i], h[j] = h[j], h[i]
		i = j
	}
}

// drainSorted empties the heap into out in ascending distance order.
func (h *maxHeap) drainSorted(out []candidate) []candidate {
	n := len(*h)
	if cap(out) < n {
		out = make([]candidate, n)
	}
	out = out[:n]
	for i := n - 1; i >= 0; i-- {
		out[i] = h.pop()
	}
	return out
}

// minHeap keeps the closest candidate at index 0.
type minHeap []candidate

func (h *minHeap) push(c candidate) {
	a := append(*h, c)
	i := len(a) - 1
	for i > 0 {
		p := (i - 1) / 2
		if a[p].dist <= a[i].dist {
			break
		}
		a[p], a[i] = a[i], a[p]
		i = p
	}
	*h = a
}

func (h *minHeap) pop() candidate {
	a := *h
	n := len(a) - 1
	top := a[0]
	a[0] = a[n]
	a = a[:n]
	i := 0
	for {
		l := 2*i + 1
		if l >= n {
			break
		}
		j := l
		if r := l + 1; r < n && a[r].dist < a[l].dist {
			j = r
		}
		if a[i].dist <= a[j].dist {
			break
		}
		a[i], a[j] = a[j], a[i]
		i = j
	}
	*h = a
	return top
}

func sqrt32(x float32) float32 { return float32(math.Sqrt(float64(x))) }
