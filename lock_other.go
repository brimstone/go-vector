//go:build !(linux || darwin || freebsd || netbsd || openbsd || dragonfly || windows)

package vector

import "os"

// fileLocking is false on platforms without a supported locking primitive
// (for example wasm, plan9, solaris, aix). There, Open and Save do not lock
// the file; the conflict check in Save still detects writes by other
// processes, but cannot prevent two saves from racing.
const fileLocking = false

func lockFile(*os.File, bool, bool) error { return nil }

func unlockFile(*os.File) error { return nil }
