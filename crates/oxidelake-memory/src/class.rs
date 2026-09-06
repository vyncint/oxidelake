//! Host memory classes.

use std::fmt;

/// How a host buffer was allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryClass {
    /// Page-locked memory registered with the device driver: eligible for DMA
    /// without staging copies.
    Pinned,
    /// Ordinary aligned heap memory (always available).
    Pageable,
}

impl fmt::Display for MemoryClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MemoryClass::Pinned => "pinned",
            MemoryClass::Pageable => "pageable",
        })
    }
}
