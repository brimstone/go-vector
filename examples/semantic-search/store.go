package main

import (
	"github.com/brimstone/go-embedding"
	"github.com/brimstone/go-vector"
)

// collectionStore adapts a vector.Collection to embedding.Store, so an
// embedding.Pipeline can write chunk records straight into the database.
type collectionStore struct{ c *vector.Collection }

var _ embedding.Store = collectionStore{}

// UpsertBatch implements embedding.Store. The chunk text moves from the
// embedding.MetaText metadata key into the record's Content, so it is stored
// once and returned with every search result.
func (s collectionStore) UpsertBatch(recs []embedding.Record) error {
	out := make([]vector.Record, len(recs))
	for i, r := range recs {
		meta := make(vector.Metadata, len(r.Metadata))
		for k, v := range r.Metadata {
			if k != embedding.MetaText {
				meta[k] = v
			}
		}
		out[i] = vector.Record{ID: r.ID, Vector: r.Vector, Content: r.Metadata[embedding.MetaText], Metadata: meta}
	}
	return s.c.UpsertBatch(out)
}
