package vector

import (
	"bytes"
	"errors"
	"path/filepath"
	"slices"
	"testing"
)

func roundTrip(t *testing.T, c *Collection) *Collection {
	t.Helper()
	var buf bytes.Buffer
	n, err := c.WriteTo(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if n != int64(buf.Len()) {
		t.Fatalf("WriteTo returned %d, wrote %d", n, buf.Len())
	}
	c2, err := ReadCollection(&buf)
	if err != nil {
		t.Fatal(err)
	}
	return c2
}

func TestPersistRoundTrip(t *testing.T) {
	for _, it := range []IndexType{HNSW, Flat} {
		for _, m := range allMetrics {
			t.Run(it.String()+"/"+m.String(), func(t *testing.T) {
				vecs := randVecs(1, 800, 12)
				opts := Options{Dimension: 12, Metric: m, Index: it, HNSW: HNSWConfig{M: 8, EfSearch: 40}}
				c := mustCollection(t, opts)
				fill(t, c, vecs, func(i int) Metadata { return Metadata{"i": id(i), "even": boolStr(i%2 == 0)} })
				for i := 0; i < 100; i++ {
					c.Delete(id(i * 3))
				}
				c2 := roundTrip(t, c)

				if c2.Options() != c.Options() {
					t.Fatalf("options %+v != %+v", c2.Options(), c.Options())
				}
				if c2.Stats() != c.Stats() {
					t.Fatalf("stats %+v != %+v", c2.Stats(), c.Stats())
				}
				// Identical graph + data => identical search results.
				for _, q := range randVecs(2, 20, 12) {
					a, _ := c.Search(q, 10, WithFilter(Where("even", "true")))
					b, _ := c2.Search(q, 10, WithFilter(Where("even", "true")))
					if !slices.EqualFunc(a, b, func(x, y Result) bool {
						return x.ID == y.ID && x.Distance == y.Distance && x.Content == y.Content && x.Metadata["i"] == y.Metadata["i"]
					}) {
						t.Fatalf("results differ:\n%v\n%v", a, b)
					}
				}
				// The loaded collection keeps working for writes.
				if err := c2.Upsert("new", vecs[0], "", nil); err != nil {
					t.Fatal(err)
				}
				if res, _ := c2.Search(vecs[0], 1); res[0].Distance > 1e-5 && m != DotProduct {
					t.Fatalf("new vector not found: %+v", res)
				}
				if it == HNSW {
					if err := c2.idx.(*hnswIndex).validate(); err != nil {
						t.Fatal(err)
					}
				}
			})
		}
	}
}

func TestPersistEmpty(t *testing.T) {
	c2 := roundTrip(t, mustCollection(t, Options{Dimension: 3}))
	if c2.Len() != 0 {
		t.Fatal("not empty")
	}
	if err := c2.Upsert("a", []float32{1, 2, 3}, "", nil); err != nil {
		t.Fatal(err)
	}
}

func TestPersistCorruption(t *testing.T) {
	c := mustCollection(t, Options{Dimension: 4})
	fill(t, c, randVecs(1, 50, 4), nil)
	var buf bytes.Buffer
	c.WriteTo(&buf)
	good := buf.Bytes()

	// Truncation at every length must fail cleanly.
	for n := 0; n < len(good); n += 7 {
		if _, err := ReadCollection(bytes.NewReader(good[:n])); !errors.Is(err, ErrCorrupt) {
			t.Fatalf("truncated at %d: err=%v", n, err)
		}
	}
	// Any single flipped byte must be detected.
	for i := 0; i < len(good); i += 13 {
		bad := slices.Clone(good)
		bad[i] ^= 0x5a
		if _, err := ReadCollection(bytes.NewReader(bad)); err == nil {
			t.Fatalf("flipped byte %d not detected", i)
		}
	}
}

func TestSaveLoadFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "c.vdb")
	c := mustCollection(t, Options{Dimension: 4})
	fill(t, c, randVecs(1, 100, 4), nil)
	if err := c.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	// Overwrite in place.
	c.Delete(id(0))
	if err := c.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	c2, err := LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if c2.Len() != 99 {
		t.Fatalf("Len=%d", c2.Len())
	}
	if _, err := LoadFile(filepath.Join(t.TempDir(), "missing.vdb")); err == nil {
		t.Fatal("expected error")
	}
}

func FuzzReadCollection(f *testing.F) {
	for _, it := range []IndexType{HNSW, Flat} {
		c, _ := NewCollection(Options{Dimension: 3, Index: it, HNSW: HNSWConfig{M: 2}})
		for i, v := range randVecs(1, 20, 3) {
			c.Upsert(id(i), v, "", Metadata{"k": "v"})
		}
		c.Delete(id(4))
		var buf bytes.Buffer
		c.WriteTo(&buf)
		f.Add(buf.Bytes())
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		c, err := ReadCollection(bytes.NewReader(data))
		if err != nil {
			return
		}
		// Anything that loads must be usable without panicking.
		q := make([]float32, c.Dimension())
		c.Search(q, 5)
		c.Upsert("fuzz", q, "", nil)
		c.Compact()
	})
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}
