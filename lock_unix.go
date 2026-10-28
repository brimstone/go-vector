//go:build linux || darwin || freebsd || netbsd || openbsd || dragonfly

package vector

import (
	"errors"
	"os"
	"syscall"
)

const fileLocking = true

// lockFile takes an advisory flock(2) lock on the whole file. flock locks
// belong to the open file description, so they also exclude other DB
// handles in the same process.
func lockFile(f *os.File, exclusive, block bool) error {
	how := syscall.LOCK_SH
	if exclusive {
		how = syscall.LOCK_EX
	}
	if !block {
		how |= syscall.LOCK_NB
	}
	for {
		err := syscall.Flock(int(f.Fd()), how)
		switch {
		case err == nil:
			return nil
		case errors.Is(err, syscall.EINTR):
			continue
		case errors.Is(err, syscall.EWOULDBLOCK):
			return errLockHeld
		default:
			return &os.PathError{Op: "flock", Path: f.Name(), Err: err}
		}
	}
}

func unlockFile(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}
