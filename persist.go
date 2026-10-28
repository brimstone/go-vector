package vector

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
)

// ErrCorrupt is returned (wrapped) when a snapshot fails validation.
var ErrCorrupt = errors.New("vector: corrupt snapshot")

const (
	fileMagic   = "GOVECDB\x00"
	fileVersion = 1
	maxStrLen   = 1 << 24
	maxMetaLen  = 1 << 20
)

// File layout (little endian):
//
//	magic[8] version:u32
//	metric:u8 index:u8 dim:u32 M:u32 efC:u32 efS:u32 seed:u64
//	slots:u32 then per slot:
//	    deleted:u8 id:str content:str nmeta:u32 (key:str val:str)* vec:f32*dim
//	HNSW only: entry:u32 maxLevel:i32 then per slot:
//	    levels:u8 then per level: n:u32 ids:u32*n
//	crc32(IEEE) of everything above:u32
//
// str is u32 length + bytes.

type countingWriter struct {
	w io.Writer
	n int64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	return n, err
}

type encoder struct {
	w       *bufio.Writer
	buf     [8]byte
	scratch []byte
}

func (e *encoder) u8(v uint8)   { e.w.WriteByte(v) }
func (e *encoder) u32(v uint32) { binary.LittleEndian.PutUint32(e.buf[:4], v); e.w.Write(e.buf[:4]) }
func (e *encoder) u64(v uint64) { binary.LittleEndian.PutUint64(e.buf[:8], v); e.w.Write(e.buf[:8]) }
func (e *encoder) str(s string) { e.u32(uint32(len(s))); e.w.WriteString(s) }
func (e *encoder) f32s(v []float32) {
	e.scratch = e.scratch[:0]
	for _, x := range v {
		e.scratch = binary.LittleEndian.AppendUint32(e.scratch, math.Float32bits(x))
	}
	e.w.Write(e.scratch)
}

func (e *encoder) u32s(v []uint32) {
	e.scratch = e.scratch[:0]
	for _, x := range v {
		e.scratch = binary.LittleEndian.AppendUint32(e.scratch, x)
	}
	e.w.Write(e.scratch)
}

// WriteTo writes a self-contained binary snapshot of the collection (options,
// vectors, content, metadata, tombstones and, for HNSW, the graph) to w and
// returns the number of bytes written. It implements io.WriterTo.
//
// Because the graph is saved, ReadCollection restores an HNSW collection
// without rebuilding it. The snapshot ends with a CRC32 checksum that
// ReadCollection verifies.
//
// Searches may run while WriteTo runs; writes wait until it returns. The
// snapshot is consistent: it reflects the collection at a single point in
// time.
func (c *Collection) WriteTo(w io.Writer) (int64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.writeLocked(w)
}

// writeLocked is WriteTo for callers that already hold c.mu (read or write).
func (c *Collection) writeLocked(w io.Writer) (int64, error) {
	cw := &countingWriter{w: w}
	crc := crc32.NewIEEE()
	e := &encoder{w: bufio.NewWriterSize(io.MultiWriter(cw, crc), 1<<16)}

	e.w.WriteString(fileMagic)
	e.u32(fileVersion)
	e.u8(uint8(c.opts.Metric))
	e.u8(uint8(c.opts.Index))
	e.u32(uint32(c.opts.Dimension))
	e.u32(uint32(c.opts.HNSW.M))
	e.u32(uint32(c.opts.HNSW.EfConstruction))
	e.u32(uint32(c.opts.HNSW.EfSearch))
	e.u64(c.opts.HNSW.Seed)

	n := c.s.len()
	e.u32(uint32(n))
	for i := 0; i < n; i++ {
		if c.deleted[i] {
			e.u8(1)
		} else {
			e.u8(0)
		}
		e.str(c.ids[i])
		e.str(c.content[i])
		e.u32(uint32(len(c.meta[i])))
		for k, v := range c.meta[i] {
			e.str(k)
			e.str(v)
		}
		e.f32s(c.s.vec(uint32(i)))
	}

	if h, ok := c.idx.(*hnswIndex); ok {
		e.u32(h.entry)
		e.u32(uint32(int32(h.maxLevel)))
		for _, node := range h.links {
			e.u8(uint8(len(node)))
			for _, nb := range node {
				e.u32(uint32(len(nb)))
				e.u32s(nb)
			}
		}
	}
	if err := e.w.Flush(); err != nil {
		return cw.n, err
	}
	binary.LittleEndian.PutUint32(e.buf[:4], crc.Sum32())
	_, err := cw.Write(e.buf[:4])
	return cw.n, err
}

type decoder struct {
	r       io.Reader
	buf     [8]byte
	scratch []byte
	err     error
}

// block reads n*4 bytes into the scratch buffer.
func (d *decoder) block(n int) []byte {
	if cap(d.scratch) < n*4 {
		d.scratch = make([]byte, n*4)
	}
	b := d.scratch[:n*4]
	if d.err == nil {
		if _, err := io.ReadFull(d.r, b); err != nil {
			d.err = err
		}
	}
	return b
}

func (d *decoder) f32s(dst []float32) {
	b := d.block(len(dst))
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
}

func (d *decoder) u32s(dst []uint32) {
	b := d.block(len(dst))
	for i := range dst {
		dst[i] = binary.LittleEndian.Uint32(b[i*4:])
	}
}

func (d *decoder) read(n int) []byte {
	if d.err != nil {
		return d.buf[:n]
	}
	if _, err := io.ReadFull(d.r, d.buf[:n]); err != nil {
		d.err = err
	}
	return d.buf[:n]
}

func (d *decoder) u8() uint8   { return d.read(1)[0] }
func (d *decoder) u32() uint32 { return binary.LittleEndian.Uint32(d.read(4)) }
func (d *decoder) u64() uint64 { return binary.LittleEndian.Uint64(d.read(8)) }

func (d *decoder) str() string {
	n := d.u32()
	if d.err != nil {
		return ""
	}
	if n > maxStrLen {
		d.fail("string too long")
		return ""
	}
	// Grow as data arrives so a corrupt length can't force a huge allocation.
	b, err := io.ReadAll(io.LimitReader(d.r, int64(n)))
	if err == nil && len(b) != int(n) {
		err = io.ErrUnexpectedEOF
	}
	if err != nil {
		d.err = err
	}
	return string(b)
}

func (d *decoder) fail(msg string) {
	if d.err == nil {
		d.err = fmt.Errorf("%w: %s", ErrCorrupt, msg)
	}
}

// ReadCollection reads a snapshot written by Collection.WriteTo. It returns
// an error wrapping ErrCorrupt if the data is truncated, fails its checksum
// or is otherwise invalid; it never panics on bad input. The data is fully
// validated before the collection is returned.
func ReadCollection(r io.Reader) (*Collection, error) {
	var hint int64
	if l, ok := r.(interface{ Len() int }); ok { // bytes.Reader, bytes.Buffer, ...
		hint = int64(l.Len())
	}
	return readCollection(r, hint)
}

// readCollection is ReadCollection with an optional upper bound on the
// snapshot's size in bytes (0 = unknown), used to preallocate safely.
func readCollection(r io.Reader, sizeHint int64) (*Collection, error) {
	br := bufio.NewReaderSize(r, 1<<16)
	crc := crc32.NewIEEE()
	d := &decoder{r: io.TeeReader(br, crc)}
	c, err := decodeCollection(d, sizeHint)
	if err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			err = fmt.Errorf("%w: truncated", ErrCorrupt)
		}
		return nil, err
	}
	var sum [4]byte
	if _, err := io.ReadFull(br, sum[:]); err != nil {
		return nil, fmt.Errorf("%w: missing checksum", ErrCorrupt)
	}
	if binary.LittleEndian.Uint32(sum[:]) != crc.Sum32() {
		return nil, fmt.Errorf("%w: checksum mismatch", ErrCorrupt)
	}
	return c, nil
}

func decodeCollection(d *decoder, sizeHint int64) (*Collection, error) {
	if string(d.read(8)) != fileMagic {
		if d.err != nil {
			return nil, d.err
		}
		return nil, fmt.Errorf("%w: bad magic", ErrCorrupt)
	}
	if v := d.u32(); d.err == nil && v != fileVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrCorrupt, v)
	}
	var opts Options
	opts.Metric = Metric(d.u8())
	opts.Index = IndexType(d.u8())
	opts.Dimension = int(d.u32())
	opts.HNSW.M = int(d.u32())
	opts.HNSW.EfConstruction = int(d.u32())
	opts.HNSW.EfSearch = int(d.u32())
	opts.HNSW.Seed = d.u64()
	if d.err != nil {
		return nil, d.err
	}
	if err := opts.validate(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorrupt, err)
	}
	if opts.HNSW.M <= 0 || opts.HNSW.M > 1<<12 || opts.HNSW.EfConstruction <= 0 || opts.HNSW.EfSearch <= 0 {
		return nil, fmt.Errorf("%w: bad hnsw config", ErrCorrupt)
	}

	c := &Collection{opts: opts, byID: make(map[string]uint32)}
	c.s = newStore(opts.Dimension, opts.Metric)
	n := int(d.u32())
	// Preallocate only as many slots as the input can actually hold, so a
	// corrupt count cannot trigger a huge allocation.
	if sizeHint > 0 {
		c.reserve(min(int64(n), sizeHint/int64(4*opts.Dimension+6)))
	}
	vec := make([]float32, opts.Dimension)
	for i := 0; i < n && d.err == nil; i++ {
		del := d.u8()
		id := d.str()
		content := d.str()
		nm := d.u32()
		if nm > maxMetaLen {
			d.fail("too much metadata")
			break
		}
		var meta Metadata
		if nm > 0 {
			meta = make(Metadata, min(nm, 64))
			for j := uint32(0); j < nm && d.err == nil; j++ {
				k := d.str()
				meta[k] = d.str()
			}
		}
		d.f32s(vec)
		if d.err != nil {
			break
		}
		if del > 1 || id == "" || !finite(vec) {
			d.fail("bad slot")
			break
		}
		slot := c.s.add(vec)
		c.ids = append(c.ids, id)
		c.deleted = append(c.deleted, del == 1)
		if del == 1 {
			c.meta = append(c.meta, nil)
			c.content = append(c.content, "")
			c.nDeleted++
			continue
		}
		if _, dup := c.byID[id]; dup {
			d.fail("duplicate id")
			break
		}
		c.meta = append(c.meta, meta)
		c.content = append(c.content, content)
		c.byID[id] = slot
	}
	if d.err != nil {
		return nil, d.err
	}

	if opts.Index == Flat {
		c.idx = &flatIndex{s: c.s}
		return c, nil
	}
	h := newHNSW(c.s, opts.HNSW)
	// Re-seed so that inserts after a load don't replay the same levels.
	h.rng = rand.New(rand.NewPCG(opts.HNSW.Seed^uint64(n), 0x9e3779b97f4a7c15))
	h.entry = d.u32()
	h.maxLevel = int(int32(d.u32()))
	h.links = make([][][]uint32, n)
	for i := 0; i < n && d.err == nil; i++ {
		levels := int(d.u8())
		if levels == 0 || levels > maxHNSWLevel+1 {
			d.fail("bad level count")
			break
		}
		node := make([][]uint32, levels)
		for l := range node {
			cnt := int(d.u32())
			if d.err != nil {
				break
			}
			mmax := h.maxConn(l)
			if cnt > mmax {
				d.fail("too many links")
				break
			}
			nb := make([]uint32, cnt, mmax+1)
			d.u32s(nb)
			node[l] = nb
		}
		h.links[i] = node
	}
	if d.err != nil {
		return nil, d.err
	}
	if err := h.validate(); err != nil {
		return nil, err
	}
	c.idx = h
	return c, nil
}

// validate checks graph invariants so a corrupt file can't cause a panic.
func (h *hnswIndex) validate() error {
	n := len(h.links)
	if n == 0 {
		if h.maxLevel != -1 {
			return fmt.Errorf("%w: bad entry point", ErrCorrupt)
		}
		return nil
	}
	if int(h.entry) >= n || h.maxLevel < 0 || len(h.links[h.entry]) != h.maxLevel+1 {
		return fmt.Errorf("%w: bad entry point", ErrCorrupt)
	}
	for _, node := range h.links {
		if len(node) > h.maxLevel+1 {
			return fmt.Errorf("%w: node above max level", ErrCorrupt)
		}
		for l, nb := range node {
			for _, x := range nb {
				if int(x) >= n || len(h.links[x]) <= l {
					return fmt.Errorf("%w: bad link", ErrCorrupt)
				}
			}
		}
	}
	return nil
}

// SaveFile atomically writes a snapshot of the collection to path. It writes
// to a temporary file in the same directory, fsyncs it and renames it over
// path, so a crash leaves either the old snapshot or the new one, never a
// partial file.
func (c *Collection) SaveFile(path string) error {
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp) // no-op after a successful rename
	if _, err := c.WriteTo(f); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	syncDir(dir) // best effort
	return nil
}

// LoadFile reads a collection snapshot written by SaveFile (or WriteTo) from
// path.
func LoadFile(path string) (*Collection, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var size int64
	if st, err := f.Stat(); err == nil {
		size = st.Size()
	}
	c, err := readCollection(f, size)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return c, nil
}

// reserve preallocates room for n slots.
func (c *Collection) reserve(n int64) {
	if n <= 0 {
		return
	}
	c.s.data = make([]float32, 0, n*int64(c.opts.Dimension))
	c.s.inv = make([]float32, 0, n)
	c.ids = make([]string, 0, n)
	c.content = make([]string, 0, n)
	c.meta = make([]Metadata, 0, n)
	c.deleted = make([]bool, 0, n)
}
