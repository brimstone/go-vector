package vector

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func dbPath(t *testing.T) string { return filepath.Join(t.TempDir(), "test.vdb") }

func mustOpen(t *testing.T, path string) *DB {
	t.Helper()
	db, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return db
}

func TestDBLifecycle(t *testing.T) {
	path := dbPath(t)
	db := mustOpen(t, path)
	docs, err := db.CreateCollection("docs", Options{Dimension: 4})
	if err != nil {
		t.Fatal(err)
	}
	imgs, _ := db.CreateCollection("images", Options{Dimension: 2, Metric: Euclidean, Index: Flat})
	fill(t, docs, randVecs(1, 30, 4), nil)
	imgs.Upsert("x", []float32{1, 1}, "an image", Metadata{"a": "b"})

	if _, err := db.CreateCollection("docs", Options{Dimension: 4}); !errors.Is(err, ErrExists) {
		t.Fatalf("duplicate create: %v", err)
	}
	for _, bad := range []string{"", "../x", "a/b", ".hidden", string(make([]byte, 200))} {
		if _, err := db.CreateCollection(bad, Options{Dimension: 4}); !errors.Is(err, ErrInvalidName) {
			t.Fatalf("name %q: %v", bad, err)
		}
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := db.CreateCollection("late", Options{Dimension: 1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("create after close: %v", err)
	}

	// Everything lives in exactly one file.
	entries, _ := os.ReadDir(filepath.Dir(path))
	if len(entries) != 1 || entries[0].Name() != "test.vdb" {
		t.Fatalf("directory contents: %v", entries)
	}

	db2 := mustOpen(t, path)
	if got := db2.Collections(); !slices.Equal(got, []string{"docs", "images"}) {
		t.Fatalf("Collections=%v", got)
	}
	d2, _ := db2.Collection("docs")
	if d2.Len() != 30 {
		t.Fatalf("docs Len=%d", d2.Len())
	}
	i2, _ := db2.Collection("images")
	if r, ok := i2.Get("x"); !ok || r.Metadata["a"] != "b" || r.Content != "an image" || i2.Options().Index != Flat {
		t.Fatalf("images: %+v", r)
	}

	if err := db2.DropCollection("images"); err != nil {
		t.Fatal(err)
	}
	if err := db2.DropCollection("images"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("double drop: %v", err)
	}
	db2.Close()
	db3 := mustOpen(t, path)
	defer db3.Close()
	if got := db3.Collections(); !slices.Equal(got, []string{"docs"}) {
		t.Fatalf("drop not persisted: %v", got)
	}
}

func metaCounter(t *testing.T, path string) uint64 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	ms := readMetas(f)
	if len(ms) == 0 {
		return 0
	}
	return ms[0].counter
}

func TestDBSaveOnlyWhenModified(t *testing.T) {
	path := dbPath(t)
	db := mustOpen(t, path)
	defer db.Close()
	c, _ := db.CreateCollection("c", Options{Dimension: 2})
	c.Upsert("a", []float32{1, 2}, "", nil)
	if err := db.Save(); err != nil {
		t.Fatal(err)
	}
	n := metaCounter(t, path)
	db.Save()
	if metaCounter(t, path) != n {
		t.Fatal("unmodified DB rewritten")
	}
	c.Upsert("b", []float32{3, 4}, "", nil)
	db.Save()
	if metaCounter(t, path) != n+1 {
		t.Fatal("modified DB not saved")
	}
}

// The file must stay bounded (at most about three snapshots, see the layout
// comment in db.go) across many saves of growing and shrinking data.
func TestDBFileSizeBounded(t *testing.T) {
	path := dbPath(t)
	db := mustOpen(t, path)
	defer db.Close()
	c, _ := db.CreateCollection("c", Options{Dimension: 16, Index: Flat})
	vecs := randVecs(1, 2000, 16)
	var maxSnap int64
	for round := 0; round < 30; round++ {
		n := 200 + (round*397)%1800
		c.DeleteWhere(func(string, Metadata) bool { return true })
		c.Compact()
		fill(t, c, vecs[:n], nil)
		if err := db.Save(); err != nil {
			t.Fatal(err)
		}
		var buf countingWriter
		buf.w = discard{}
		c.WriteTo(&buf)
		maxSnap = max(maxSnap, buf.n)
		st, _ := os.Stat(path)
		if limit := dataStart + 3*(maxSnap+dataAlign+1024); st.Size() > limit {
			t.Fatalf("round %d: file %d bytes > limit %d", round, st.Size(), limit)
		}
	}
	db2 := mustOpen(t, path)
	defer db2.Close()
	c2, _ := db2.Collection("c")
	if c2.Len() != c.Len() {
		t.Fatalf("reload Len %d != %d", c2.Len(), c.Len())
	}
}

type discard struct{}

func (discard) Write(p []byte) (int, error) { return len(p), nil }

// saveTwice creates a DB whose file holds two snapshots: an older one with
// record "old" only, and the current one with "old" and "new".
func saveTwice(t *testing.T) (path string) {
	path = dbPath(t)
	db := mustOpen(t, path)
	c, _ := db.CreateCollection("c", Options{Dimension: 2})
	c.Upsert("old", []float32{1, 0}, "", nil)
	if err := db.Save(); err != nil {
		t.Fatal(err)
	}
	c.Upsert("new", []float32{0, 1}, "", nil)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func newestMeta(t *testing.T, path string) dbMeta {
	f, _ := os.Open(path)
	defer f.Close()
	return readMetas(f)[0]
}

func corruptAt(t *testing.T, path string, off int64) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	b := make([]byte, 1)
	f.ReadAt(b, off)
	b[0] ^= 0xff
	f.WriteAt(b, off)
}

func hasRecords(t *testing.T, path string, want ...string) {
	t.Helper()
	db := mustOpen(t, path)
	defer db.Close()
	c, ok := db.Collection("c")
	if !ok {
		t.Fatal("collection missing")
	}
	var got []string
	c.Range(func(r Record) bool { got = append(got, r.ID); return true })
	if !slices.Equal(got, want) {
		t.Fatalf("records %v, want %v", got, want)
	}
}

func TestDBRecovery(t *testing.T) {
	t.Run("intact", func(t *testing.T) {
		hasRecords(t, saveTwice(t), "old", "new")
	})
	t.Run("torn meta falls back to previous snapshot", func(t *testing.T) {
		path := saveTwice(t)
		m := newestMeta(t, path)
		corruptAt(t, path, int64(m.slot*metaSlotSize+20))
		hasRecords(t, path, "old")
	})
	t.Run("damaged data falls back to previous snapshot", func(t *testing.T) {
		path := saveTwice(t)
		m := newestMeta(t, path)
		corruptAt(t, path, int64(m.off+m.length/2))
		hasRecords(t, path, "old")
	})
	t.Run("crash before commit leaves last snapshot", func(t *testing.T) {
		// Simulate a crash after writing a new region but before its meta
		// slot: garbage where the next region would go, no meta update.
		path := saveTwice(t)
		m := newestMeta(t, path)
		f, _ := os.OpenFile(path, os.O_RDWR, 0)
		garbage := make([]byte, 8192)
		for i := range garbage {
			garbage[i] = byte(i * 7)
		}
		f.WriteAt(garbage, int64(m.off+m.length+dataAlign))
		f.Close()
		hasRecords(t, path, "old", "new")
	})
	t.Run("saving after fallback replaces damaged snapshot", func(t *testing.T) {
		path := saveTwice(t)
		m := newestMeta(t, path)
		corruptAt(t, path, int64(m.off+m.length/2))
		db := mustOpen(t, path)
		c, _ := db.Collection("c")
		c.Upsert("again", []float32{1, 1}, "", nil)
		if err := db.Close(); err != nil {
			t.Fatalf("save after fallback: %v", err)
		}
		hasRecords(t, path, "old", "again")
	})
}

func TestDBOpenCorruptFile(t *testing.T) {
	path := dbPath(t)
	os.WriteFile(path, []byte("garbage that is not a database"), 0o644)
	if _, err := Open(path); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("err=%v", err)
	}
}

func TestDBConflict(t *testing.T) {
	path := dbPath(t)
	db := mustOpen(t, path)
	db.CreateCollection("c", Options{Dimension: 2})
	db.Close()

	a, b := mustOpen(t, path), mustOpen(t, path)
	ca, _ := a.Collection("c")
	cb, _ := b.Collection("c")
	ca.Upsert("from-a", []float32{1, 0}, "", nil)
	cb.Upsert("from-b", []float32{0, 1}, "", nil)
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}
	if err := b.Save(); !errors.Is(err, ErrConflict) {
		t.Fatalf("second writer: %v, want ErrConflict", err)
	}
	if err := b.Close(); !errors.Is(err, ErrConflict) {
		t.Fatalf("Close: %v, want ErrConflict", err)
	}
	// a can keep saving: its view is current.
	ca.Upsert("more-a", []float32{1, 1}, "", nil)
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	hasRecords(t, path, "from-a", "more-a")
}

func TestDBMemoryOnly(t *testing.T) {
	db := mustOpen(t, "")
	c, err := db.GetOrCreateCollection("x", Options{Dimension: 3})
	if err != nil {
		t.Fatal(err)
	}
	c2, _ := db.GetOrCreateCollection("x", Options{Dimension: 3})
	if c != c2 {
		t.Fatal("expected same collection")
	}
	if _, err := db.GetOrCreateCollection("x", Options{Dimension: 4}); !errors.Is(err, ErrInvalidOptions) {
		t.Fatalf("mismatch: %v", err)
	}
	c.Upsert("a", []float32{1, 2, 3}, "", nil)
	if err := db.Save(); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestFileLockExcludes(t *testing.T) {
	if !fileLocking {
		t.Skip("no file locking on this platform")
	}
	path := dbPath(t)
	os.WriteFile(path, nil, 0o644)
	f1, _ := os.OpenFile(path, os.O_RDWR, 0)
	f2, _ := os.OpenFile(path, os.O_RDWR, 0)
	defer f1.Close()
	defer f2.Close()

	if err := lockFile(f1, true, false); err != nil {
		t.Fatal(err)
	}
	if err := lockFile(f2, true, false); !errors.Is(err, errLockHeld) {
		t.Fatalf("exclusive vs exclusive: %v", err)
	}
	if err := lockFile(f2, false, false); !errors.Is(err, errLockHeld) {
		t.Fatalf("shared vs exclusive: %v", err)
	}
	unlockFile(f1)
	if err := lockFile(f1, false, false); err != nil {
		t.Fatal(err)
	}
	if err := lockFile(f2, false, false); err != nil {
		t.Fatalf("shared vs shared: %v", err)
	}
	if err := lockFile(f1, true, false); !errors.Is(err, errLockHeld) {
		t.Fatalf("exclusive vs shared: %v", err)
	}
}

// TestLockHelperProcess is not a real test: TestDBCrossProcessLock re-runs
// the test binary with GOVECTOR_LOCK_HELPER set, making this function hold
// an exclusive lock on the given file from a separate OS process.
func TestLockHelperProcess(t *testing.T) {
	path := os.Getenv("GOVECTOR_LOCK_HELPER")
	if path == "" {
		t.Skip("helper process only")
	}
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		fmt.Println("error:", err)
		os.Exit(1)
	}
	if err := lockFile(f, true, true); err != nil {
		fmt.Println("error:", err)
		os.Exit(1)
	}
	fmt.Println("locked")
	time.Sleep(700 * time.Millisecond)
	unlockFile(f)
	os.Exit(0)
}

func TestDBCrossProcessLock(t *testing.T) {
	if !fileLocking {
		t.Skip("no file locking on this platform")
	}
	if testing.Short() {
		t.Skip("spawns a process")
	}
	path := dbPath(t)
	db := mustOpen(t, path)
	defer db.Close()
	c, _ := db.CreateCollection("c", Options{Dimension: 2})
	db.Save()

	cmd := exec.Command(os.Args[0], "-test.run=^TestLockHelperProcess$", "-test.count=1")
	cmd.Env = append(os.Environ(), "GOVECTOR_LOCK_HELPER="+path)
	out, _ := cmd.StdoutPipe()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer cmd.Wait()
	line, _ := bufio.NewReader(out).ReadString('\n')
	if line != "locked\n" {
		t.Fatalf("helper: %q", line)
	}

	// The other process holds the lock: Save must wait for it, and Open
	// (shared lock) must too.
	c.Upsert("a", []float32{1, 2}, "", nil)
	start := time.Now()
	if err := db.Save(); err != nil {
		t.Fatal(err)
	}
	if waited := time.Since(start); waited < 300*time.Millisecond {
		t.Fatalf("Save did not wait for the other process's lock (%v)", waited)
	}
	hasRecords(t, path, "a")
}

func FuzzOpenDB(f *testing.F) {
	dir := f.TempDir()
	seed := filepath.Join(dir, "seed.vdb")
	db, _ := Open(seed)
	c, _ := db.CreateCollection("c", Options{Dimension: 2, HNSW: HNSWConfig{M: 2}})
	c.Upsert("a", []float32{1, 2}, "x", Metadata{"k": "v"})
	db.Save()
	c.Upsert("b", []float32{3, 4}, "", nil)
	db.Close()
	data, _ := os.ReadFile(seed)
	f.Add(data)
	f.Fuzz(func(t *testing.T, data []byte) {
		path := filepath.Join(t.TempDir(), "f.vdb")
		os.WriteFile(path, data, 0o644)
		db, err := Open(path)
		if err != nil {
			return
		}
		for _, n := range db.Collections() {
			c, _ := db.Collection(n)
			c.Search(make([]float32, c.Dimension()), 3)
			c.Upsert("fuzz", make([]float32, c.Dimension()), "", nil)
		}
		if err := db.Close(); err != nil {
			t.Fatalf("Close after successful Open: %v", err)
		}
	})
}
