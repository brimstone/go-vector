//go:build windows

package vector

import (
	"errors"
	"os"
	"syscall"
	"unsafe"
)

const fileLocking = true

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLockFileEx   = kernel32.NewProc("LockFileEx")
	procUnlockFileEx = kernel32.NewProc("UnlockFileEx")
)

const (
	lockfileFailImmediately = 0x1
	lockfileExclusiveLock   = 0x2
	errorLockViolation      = syscall.Errno(33)
)

// Windows byte-range locks are mandatory: a locked range cannot be read or
// written through other handles. Locking a single byte far beyond any real
// file size gives flock-like advisory semantics without blocking I/O.
func lockRange() *syscall.Overlapped {
	return &syscall.Overlapped{Offset: 0, OffsetHigh: 0x7fffffff}
}

func lockFile(f *os.File, exclusive, block bool) error {
	var flags uintptr
	if exclusive {
		flags |= lockfileExclusiveLock
	}
	if !block {
		flags |= lockfileFailImmediately
	}
	r, _, err := procLockFileEx.Call(f.Fd(), flags, 0, 1, 0, uintptr(unsafe.Pointer(lockRange())))
	if r != 0 {
		return nil
	}
	if errors.Is(err, errorLockViolation) {
		return errLockHeld
	}
	return &os.PathError{Op: "LockFileEx", Path: f.Name(), Err: err}
}

func unlockFile(f *os.File) error {
	r, _, err := procUnlockFileEx.Call(f.Fd(), 0, 1, 0, uintptr(unsafe.Pointer(lockRange())))
	if r == 0 {
		return &os.PathError{Op: "UnlockFileEx", Path: f.Name(), Err: err}
	}
	return nil
}
