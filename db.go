package vector

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sync"
)

var (
	// ErrExists is returned when creating a collection that already exists.
	ErrExists = errors.New("vector: collection already exists")
	// ErrNotFound is returned for operations on a missing collection.
	ErrNotFound = errors.New("vector: collection not found")
	// ErrInvalidName is returned for collection names that are not 1-128
	// characters of [A-Za-z0-9_.-] starting with a letter, digit or '_'.
	ErrInvalidName = errors.New("vector: invalid collection name")
	// ErrClosed is returned after the DB has been closed.
	ErrClosed = errors.New("vector: db closed")
	// ErrConflict is returned by Save when another process (or another DB
	// handle) saved the database file after this DB loaded it. Saving would
	// silently discard that process's changes, so nothing is written. Reopen
	// the file to pick up the other changes, then re-apply your own.
	ErrConflict = errors.New("vector: database file was modified by another writer")

	errLockHeld = errors.New("vector: file is locked")
)

var nameRE = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$`)

// Single-file layout
//
//	[0, 4096)     header: two meta slots, at offsets 0 and 512
//	[4096, EOF)   data: one or two snapshot regions
//
// A meta slot is: magic[8] version:u32 counter:u64 offset:u64 length:u64
// dataCRC:u32 metaCRC:u32 (CRC32 of the preceding 40 bytes). The valid slot
// with the highest counter names the current snapshot region, whose contents
// are: ncollections:u32, then per collection name:str len:u64 followed by
// len bytes of Collection.WriteTo output.
//
// Save writes the new region where it cannot overlap the current one, fsyncs
// it, then overwrites the older meta slot and fsyncs again, in the style of
// LMDB's dual meta pages. A crash at any point leaves the previous meta slot
// and region intact. The file is truncated after each commit. With stable
// sizes it holds at most two snapshots; while snapshots keep growing, a dead
// gap of up to one older snapshot can precede the live one, so the file is
// bounded by about three times the largest snapshot.
const (
	dbMagic       = "GOVECDB\x01"
	dbVersion     = 1
	metaSlotSize  = 512
	metaLen       = 44
	dataStart     = 4096
	dataAlign     = 4096
	maxCollection = 1 << 16
)

type dbMeta struct {
	slot    int
	counter uint64
	off     uint64
	length  uint64
	crc     uint32
}

func (m dbMeta) encode() []byte {
	b := make([]byte, 0, metaLen)
	b = append(b, dbMagic...)
	b = binary.LittleEndian.AppendUint32(b, dbVersion)
	b = binary.LittleEndian.AppendUint64(b, m.counter)
	b = binary.LittleEndian.AppendUint64(b, m.off)
	b = binary.LittleEndian.AppendUint64(b, m.length)
	b = binary.LittleEndian.AppendUint32(b, m.crc)
	return binary.LittleEndian.AppendUint32(b, crc32.ChecksumIEEE(b))
}

func decodeMeta(b []byte, slot int) (dbMeta, bool) {
	if len(b) < metaLen || string(b[:8]) != dbMagic ||
		binary.LittleEndian.Uint32(b[8:]) != dbVersion ||
		binary.LittleEndian.Uint32(b[40:]) != crc32.ChecksumIEEE(b[:40]) {
		return dbMeta{}, false
	}
	return dbMeta{
		slot:    slot,
		counter: binary.LittleEndian.Uint64(b[12:]),
		off:     binary.LittleEndian.Uint64(b[20:]),
		length:  binary.LittleEndian.Uint64(b[28:]),
		crc:     binary.LittleEndian.Uint32(b[36:]),
	}, true
}

// readMetas returns the valid meta slots, newest first.
func readMetas(f *os.File) []dbMeta {
	var ms []dbMeta
	buf := make([]byte, metaLen)
	for slot := 0; slot < 2; slot++ {
		if _, err := f.ReadAt(buf, int64(slot*metaSlotSize)); err != nil {
			continue
		}
		if m, ok := decodeMeta(buf, slot); ok {
			ms = append(ms, m)
		}
	}
	slices.SortFunc(ms, func(a, b dbMeta) int { return -cmpU64(a.counter, b.counter) })
	return ms
}

func cmpU64(a, b uint64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// DB is a set of named collections stored together in a single file.
//
// Collections live in memory; the file is a snapshot of all of them. Changes
// reach the file only when Save or Close is called. Saves are crash-safe: a
// crash or power loss during Save leaves the previous snapshot intact (see
// the package documentation on durability).
//
// # Multiple processes
//
// The file is locked at the OS level (flock on Unix, LockFileEx on
// Windows). Save holds an exclusive lock while writing, so two processes
// can never write the file at the same time, and Open holds a shared lock
// while reading, so it never observes a save in progress. Locks are released
// between operations; the file is not locked while a DB is merely open.
//
// Because every process works on its own in-memory copy, a lock alone would
// let the last writer silently erase another process's changes. Save
// prevents this: if the file was saved by anyone else since this DB loaded
// or last saved it, Save returns ErrConflict and writes nothing. The
// intended pattern is one writer process with any number of readers, which
// reopen the file to see new data.
//
// Locks are advisory and may not work on network filesystems (NFS, SMB).
// On platforms without a supported lock primitive (wasm, plan9, solaris,
// aix) only the conflict check applies.
//
// A DB is safe for concurrent use by multiple goroutines.
type DB struct {
	mu      sync.RWMutex
	path    string
	f       *os.File
	cols    map[string]*Collection
	saved   map[string]uint64 // collection generation last written
	dirty   bool              // collections created or dropped since last save
	counter uint64            // meta counter of the snapshot this DB is based on
	closed  bool
}

// Open opens the database file at path, creating it if it does not exist,
// and loads every collection in it into memory. Opening takes a shared lock
// on the file while it is read, so it waits for any save in progress in
// another process.
//
// It returns an error wrapping ErrCorrupt if the file exists but is not a
// valid database. If the newest snapshot is damaged, Open falls back to the
// previous one when that is intact.
//
// If path is "", the DB is memory-only and Save is a no-op.
func Open(path string) (*DB, error) {
	db := &DB{path: path, cols: map[string]*Collection{}, saved: map[string]uint64{}}
	if path == "" {
		return db, nil
	}
	_, statErr := os.Stat(path)
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	if errors.Is(statErr, os.ErrNotExist) {
		syncDir(filepath.Dir(path))
	}
	if err := lockFile(f, false, true); err != nil {
		f.Close()
		return nil, err
	}
	cols, counter, err := readDBFile(f)
	unlockFile(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	db.f, db.cols, db.counter = f, cols, counter
	for name, c := range cols {
		db.saved[name] = c.gen
	}
	return db, nil
}

// readDBFile loads the newest intact snapshot. The returned counter is that
// of the newest valid meta slot, even when its region was damaged and an
// older one was loaded, so the next Save replaces the damaged snapshot
// instead of reporting a conflict.
func readDBFile(f *os.File) (map[string]*Collection, uint64, error) {
	metas := readMetas(f)
	if len(metas) == 0 {
		st, err := f.Stat()
		if err != nil {
			return nil, 0, err
		}
		if st.Size() == 0 {
			return map[string]*Collection{}, 0, nil
		}
		return nil, 0, fmt.Errorf("%w: not a database file (no valid header)", ErrCorrupt)
	}
	var firstErr error
	for _, m := range metas {
		cols, err := readRegion(f, m)
		if err == nil {
			return cols, metas[0].counter, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	return nil, 0, firstErr
}

func readRegion(f *os.File, m dbMeta) (map[string]*Collection, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if m.off < dataStart || m.length < 4 || m.off+m.length > uint64(st.Size()) || m.off+m.length < m.off {
		return nil, fmt.Errorf("%w: snapshot region out of bounds", ErrCorrupt)
	}
	crc := crc32.NewIEEE()
	r := bufio.NewReaderSize(io.TeeReader(io.NewSectionReader(f, int64(m.off), int64(m.length)), crc), 1<<16)
	d := &decoder{r: r}
	n := d.u32()
	if d.err == nil && n > maxCollection {
		d.fail("too many collections")
	}
	cols := map[string]*Collection{}
	remaining := m.length - 4
	for i := uint32(0); i < n && d.err == nil; i++ {
		name := d.str()
		clen := d.u64()
		if d.err != nil {
			break
		}
		hdr := uint64(4 + len(name) + 8)
		if !nameRE.MatchString(name) || cols[name] != nil || clen > remaining-min(remaining, hdr) {
			d.fail("bad collection entry")
			break
		}
		remaining -= hdr + clen
		lr := io.LimitReader(r, int64(clen))
		c, err := readCollection(lr, int64(clen))
		if err != nil {
			return nil, fmt.Errorf("collection %q: %w", name, err)
		}
		io.Copy(io.Discard, lr)
		cols[name] = c
	}
	if d.err != nil {
		if errors.Is(d.err, io.EOF) || errors.Is(d.err, io.ErrUnexpectedEOF) {
			return nil, fmt.Errorf("%w: truncated snapshot", ErrCorrupt)
		}
		return nil, d.err
	}
	io.Copy(io.Discard, r) // include any trailing bytes in the checksum
	if crc.Sum32() != m.crc {
		return nil, fmt.Errorf("%w: snapshot checksum mismatch", ErrCorrupt)
	}
	return cols, nil
}

// Path returns the database file path ("" for a memory-only DB).
func (db *DB) Path() string { return db.path }

// CreateCollection creates a new, empty collection. name must be 1-128
// characters from [A-Za-z0-9_.-], starting with a letter, digit or '_'. It
// returns an error wrapping ErrExists if the name is taken.
//
// The collection is written to the file on the next Save.
func (db *DB) CreateCollection(name string, opts Options) (*Collection, error) {
	if !nameRE.MatchString(name) {
		return nil, fmt.Errorf("%w: %q", ErrInvalidName, name)
	}
	c, err := NewCollection(opts)
	if err != nil {
		return nil, err
	}
	db.mu.Lock()
	defer db.mu.Unlock()
	if db.closed {
		return nil, ErrClosed
	}
	if _, ok := db.cols[name]; ok {
		return nil, fmt.Errorf("%w: %q", ErrExists, name)
	}
	db.cols[name] = c
	db.dirty = true
	return c, nil
}

// GetOrCreateCollection returns the named collection, creating it with opts
// if it does not exist. This is the usual way for a program to obtain its
// collections at startup.
//
// An existing collection keeps its stored options. If its Dimension or
// Metric differ from opts, an error wrapping ErrInvalidOptions is returned;
// this typically means the embedding model was changed and the collection
// must be rebuilt under a new name.
func (db *DB) GetOrCreateCollection(name string, opts Options) (*Collection, error) {
	if c, ok := db.Collection(name); ok {
		if o := c.Options(); o.Dimension != opts.Dimension || o.Metric != opts.Metric {
			return nil, fmt.Errorf("%w: collection %q has dimension %d / %s",
				ErrInvalidOptions, name, o.Dimension, o.Metric)
		}
		return c, nil
	}
	c, err := db.CreateCollection(name, opts)
	if errors.Is(err, ErrExists) { // lost a race; retry the lookup
		return db.GetOrCreateCollection(name, opts)
	}
	return c, err
}

// Collection returns the named collection and whether it exists.
func (db *DB) Collection(name string) (*Collection, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	c, ok := db.cols[name]
	return c, ok
}

// Collections returns the names of all collections, sorted.
func (db *DB) Collections() []string {
	db.mu.RLock()
	defer db.mu.RUnlock()
	names := make([]string, 0, len(db.cols))
	for n := range db.cols {
		names = append(names, n)
	}
	slices.Sort(names)
	return names
}

// DropCollection removes a collection from the DB. It disappears from the
// file on the next Save. The *Collection value remains usable in memory but
// is no longer saved. It returns an error wrapping ErrNotFound if there is no
// such collection.
func (db *DB) DropCollection(name string) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if _, ok := db.cols[name]; !ok {
		return fmt.Errorf("%w: %q", ErrNotFound, name)
	}
	delete(db.cols, name)
	delete(db.saved, name)
	db.dirty = true
	return nil
}

// Save writes all collections to the database file if anything changed
// since the last save, and is a cheap no-op otherwise. Because the database
// is a single file, a save rewrites every collection, not only the modified
// ones.
//
// Save takes an exclusive OS-level lock on the file for its duration,
// waiting for other processes' saves or opens to finish first. It returns
// ErrConflict, without writing, if another writer saved the file since this
// DB loaded or last saved it.
//
// Searches continue during Save; writes to collections wait until it
// returns.
func (db *DB) Save() error {
	db.mu.Lock()
	defer db.mu.Unlock()
	return db.saveLocked()
}

func (db *DB) modifiedLocked() bool {
	if db.dirty {
		return true
	}
	for name, c := range db.cols {
		c.mu.RLock()
		gen := c.gen
		c.mu.RUnlock()
		if db.saved[name] != gen {
			return true
		}
	}
	return false
}

func (db *DB) saveLocked() error {
	if db.closed {
		return ErrClosed
	}
	if db.path == "" || !db.modifiedLocked() {
		return nil
	}
	if err := lockFile(db.f, true, true); err != nil {
		return err
	}
	defer unlockFile(db.f)

	metas := readMetas(db.f)
	var live dbMeta
	slot := 0
	if len(metas) > 0 {
		live = metas[0]
		slot = 1 - live.slot // overwrite the older (or unused) slot
	}
	if live.counter != db.counter {
		return ErrConflict
	}

	// Hold every collection's read lock so that the size pass and the write
	// pass see identical data. Searches can still run.
	names := make([]string, 0, len(db.cols))
	for n := range db.cols {
		names = append(names, n)
	}
	slices.Sort(names)
	gens := make([]uint64, len(names))
	sizes := make([]int64, len(names))
	total := uint64(4)
	for i, n := range names {
		c := db.cols[n]
		c.mu.RLock()
		defer c.mu.RUnlock()
		gens[i] = c.gen
		size, err := c.writeLocked(io.Discard)
		if err != nil {
			return err
		}
		sizes[i] = size
		total += uint64(4+len(n)+8) + uint64(size)
	}

	// Place the new region where it cannot overlap the live one.
	off := uint64(dataStart)
	if live.length > 0 && off+total > live.off {
		off = (live.off + live.length + dataAlign - 1) / dataAlign * dataAlign
	}

	crc := crc32.NewIEEE()
	bw := bufio.NewWriterSize(io.MultiWriter(io.NewOffsetWriter(db.f, int64(off)), crc), 1<<16)
	e := &encoder{w: bw}
	e.u32(uint32(len(names)))
	for i, n := range names {
		e.str(n)
		e.u64(uint64(sizes[i]))
		if written, err := db.cols[n].writeLocked(bw); err != nil {
			return err
		} else if written != sizes[i] {
			return fmt.Errorf("vector: internal error: collection %q size changed during save", n)
		}
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	if err := db.f.Sync(); err != nil { // data durable before it is referenced
		return err
	}
	m := dbMeta{counter: live.counter + 1, off: off, length: total, crc: crc.Sum32()}
	if _, err := db.f.WriteAt(m.encode(), int64(slot*metaSlotSize)); err != nil {
		return err
	}
	if err := db.f.Sync(); err != nil { // commit point
		return err
	}
	// The previous snapshot is no longer referenced; drop whatever lies
	// beyond the new one. Failure here is harmless.
	if err := db.f.Truncate(int64(off + total)); err == nil {
		db.f.Sync()
	}

	db.counter = m.counter
	db.dirty = false
	clear(db.saved)
	for i, n := range names {
		db.saved[n] = gens[i]
	}
	return nil
}

// Close saves any changes (see Save) and closes the database file. After
// Close, CreateCollection and Save return ErrClosed. If the final save fails
// (for example with ErrConflict) the file is still closed and the error is
// returned. Closing twice is a no-op.
func (db *DB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if db.closed {
		return nil
	}
	err := db.saveLocked()
	db.closed = true
	if db.f != nil {
		if cerr := db.f.Close(); err == nil {
			err = cerr
		}
	}
	return err
}

func syncDir(dir string) {
	if d, err := os.Open(dir); err == nil {
		d.Sync()
		d.Close()
	}
}
