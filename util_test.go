package vector

import (
	"cmp"
	"fmt"
)

func fmtSscan(s string, i *int) { fmt.Sscanf(s, "v%d", i) }

func cmpF(a, b float32) int { return cmp.Compare(a, b) }
