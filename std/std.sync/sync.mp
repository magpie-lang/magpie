module std.sync
exports { TCell, TChannel, TMutex, TRwLock }
imports { }
digest "5a29da5f9a7763e27806fa9370d331035bd872f96edf6b9a0792e96cfc988e73"

heap struct TMutex<t: type> {
}

heap struct TRwLock<t: type> {
}

heap struct TChannel<t: type> {
}

heap struct TCell<t: type> {
}
