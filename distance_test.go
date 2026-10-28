package vector

import (
	"math"
	"testing"
)

func TestDistanceKernels(t *testing.T) {
	for n := 0; n <= 70; n++ {
		vs := randVecs(uint64(n)+1, 2, n)
		a, b := vs[0], vs[1]
		var wantDot, wantL2 float64
		for i := range a {
			wantDot += float64(a[i]) * float64(b[i])
			d := float64(a[i]) - float64(b[i])
			wantL2 += d * d
		}
		tol := 1e-4 * (1 + float64(n))
		if got := float64(Dot(a, b)); math.Abs(got-wantDot) > tol {
			t.Errorf("n=%d Dot=%v want %v", n, got, wantDot)
		}
		if got := float64(SquaredL2(a, b)); math.Abs(got-wantL2) > tol {
			t.Errorf("n=%d SquaredL2=%v want %v", n, got, wantL2)
		}
	}
}

func TestCosineSimilarity(t *testing.T) {
	cases := []struct {
		a, b []float32
		want float32
	}{
		{[]float32{1, 0}, []float32{1, 0}, 1},
		{[]float32{1, 0}, []float32{0, 1}, 0},
		{[]float32{1, 0}, []float32{-2, 0}, -1},
		{[]float32{0, 0}, []float32{1, 1}, 0},
		{[]float32{3, 4}, []float32{6, 8}, 1},
	}
	for _, c := range cases {
		if got := CosineSimilarity(c.a, c.b); math.Abs(float64(got-c.want)) > 1e-6 {
			t.Errorf("CosineSimilarity(%v,%v)=%v want %v", c.a, c.b, got, c.want)
		}
	}
}

func TestDistanceLengthMismatchPanics(t *testing.T) {
	for name, f := range map[string]func(){
		"Dot":    func() { Dot([]float32{1}, []float32{1, 2}) },
		"L2":     func() { SquaredL2([]float32{1}, []float32{1, 2}) },
		"Cosine": func() { CosineSimilarity([]float32{1}, []float32{1, 2}) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: expected panic", name)
				}
			}()
			f()
		}()
	}
}

func TestHeaps(t *testing.T) {
	vals := []float32{5, 3, 8, 1, 9, 2, 7, 4, 6, 0}
	var mn minHeap
	var mx maxHeap
	for i, v := range vals {
		mn.push(candidate{uint32(i), v})
		mx.push(candidate{uint32(i), v})
	}
	for want := float32(0); want < 10; want++ {
		if got := mn.pop().dist; got != want {
			t.Fatalf("minHeap pop=%v want %v", got, want)
		}
	}
	sorted := mx.drainSorted(nil)
	for i, c := range sorted {
		if c.dist != float32(i) {
			t.Fatalf("drainSorted[%d]=%v", i, c.dist)
		}
	}
}
