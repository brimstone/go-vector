module github.com/brimstone/go-vector/examples/semantic-search

go 1.22

require (
	github.com/brimstone/go-embedding v0.0.0
	github.com/brimstone/go-vector v0.0.0
)

// The example is its own module so that the vector library itself keeps no
// dependencies. These point at local checkouts of both repositories, side by
// side; drop them to use published versions instead.
replace (
	github.com/brimstone/go-embedding => ../../../embedding
	github.com/brimstone/go-vector => ../..
)
