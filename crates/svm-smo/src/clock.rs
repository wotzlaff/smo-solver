//! Wall-clock time that also works on `wasm32-unknown-unknown`.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(all(target_arch = "wasm32", feature = "web-time"))]
pub(crate) use web_time::Instant;

/// Fallback without a clock: elapsed time is always zero (time limits are ignored).
#[cfg(all(target_arch = "wasm32", not(feature = "web-time")))]
#[derive(Clone, Copy)]
pub(crate) struct Instant;

#[cfg(all(target_arch = "wasm32", not(feature = "web-time")))]
impl Instant {
    pub(crate) fn now() -> Self {
        Instant
    }

    pub(crate) fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
