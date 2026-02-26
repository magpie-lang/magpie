module std.sync
exports { TMutex, TRwLock, TChannel, TCell }
imports { }
digest "0000000000000000"

heap struct TMutex<t: type> {
}

heap struct TRwLock<t: type> {
}

heap struct TChannel<t: type> {
}

heap struct TCell<t: type> {
}
