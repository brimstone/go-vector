.PHONY: check fmt vet test race bench fuzz

# examples/semantic-search is a separate module (it depends on
# github.com/brimstone/go-embedding, which the library must not), so the
# targets below run in it explicitly.
EXAMPLE := examples/semantic-search

check: fmt vet test race

fmt:
	@out=$$(gofmt -l .); if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

vet:
	go vet ./...
	cd $(EXAMPLE) && go vet ./...

test:
	go test ./...
	cd $(EXAMPLE) && go test ./...

race:
	CGO_ENABLED=1 go test -race ./...
	cd $(EXAMPLE) && CGO_ENABLED=1 go test -race ./...

bench:
	go test -run '^$$' -bench . -benchtime 2s .

fuzz:
	go test -run '^$$' -fuzz FuzzReadCollection -fuzztime 60s .
