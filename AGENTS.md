# vector: agent guidance

Pure-Go (no cgo) embeddable vector database. Module `github.com/brimstone/go-vector`.

## Required before finishing any change

Every agent (including subagents) must run these and leave them clean:

```sh
gofmt -l .        # must print nothing; fix with: gofmt -w .
go vet ./...      # must print nothing
go test ./...
```

Or simply `make check`, which runs all three plus the race detector.

## Rules

- No cgo and no third-party dependencies in the library package; stdlib
  only. `CGO_ENABLED=0 go build ./...` must succeed. (The race detector
  needs cgo for its own runtime; that is fine for testing.)
- Every exported identifier needs a doc comment. Primary users embed this
  package in other programs to store and search text embeddings, so docs
  should say how to use things for that.
- Recall-sensitive changes to `hnsw.go` must keep `TestHNSW*` passing and
  should be checked with `go test -run '^$' -bench 'Search' .`, which
  reports recall@10.
- The collection snapshot format is in `persist.go` (bump `fileVersion`);
  the single-file DB layout (header meta slots, regions, locking) is in
  `db.go` (bump `dbVersion`). Keep `FuzzReadCollection` panic-free, and
  keep `TestDBRecovery` / `TestDBCrossProcessLock` passing.
- Cross-check other platforms after touching `lock_*.go`:
  `GOOS=windows go vet ./... && GOOS=darwin go vet ./...`
