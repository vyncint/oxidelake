//! Host buffers: pinned when the backend can provide it, pageable otherwise.

use std::fmt;

use oxidelake_core::EngineError;

use crate::{AlignedBuf, MemoryClass};

/// Something that can hand out page-locked host memory (a CUDA context).
pub trait PinnedProvider: Send + Sync + fmt::Debug {
    /// Allocates `len` zeroed bytes of pinned memory.
    fn alloc_pinned(&self, len: usize) -> Result<HostAlloc, EngineError>;
}

enum Inner {
    Pageable(AlignedBuf),
    #[cfg(feature = "cuda")]
    Pinned(crate::pinned::PinnedBuf),
}

/// A host buffer whose memory class is observable.
pub struct HostAlloc {
    class: MemoryClass,
    inner: Inner,
}

impl HostAlloc {
    /// Allocates ordinary cache-line-aligned host memory.
    pub fn pageable(len: usize) -> Result<Self, EngineError> {
        Ok(Self {
            class: MemoryClass::Pageable,
            inner: Inner::Pageable(AlignedBuf::host(len)?),
        })
    }

    /// Wraps a pinned allocation.
    #[cfg(feature = "cuda")]
    pub fn from_pinned(buf: crate::pinned::PinnedBuf) -> Self {
        Self {
            class: MemoryClass::Pinned,
            inner: Inner::Pinned(buf),
        }
    }

    /// Tries the provider for pinned memory and falls back to pageable memory,
    /// logging the reason at debug level. The result's [`Self::class`] tells
    /// callers which path was taken.
    pub fn pinned_or_pageable(
        len: usize,
        provider: Option<&dyn PinnedProvider>,
    ) -> Result<Self, EngineError> {
        if let Some(provider) = provider {
            match provider.alloc_pinned(len) {
                Ok(buf) => return Ok(buf),
                Err(err) => {
                    tracing::debug!(error = %err, len, "pinned allocation unavailable; using pageable memory")
                }
            }
        }
        Self::pageable(len)
    }

    /// The memory class actually used.
    pub const fn class(&self) -> MemoryClass {
        self.class
    }

    /// Logical length in bytes.
    pub fn len(&self) -> usize {
        match &self.inner {
            Inner::Pageable(b) => b.len(),
            #[cfg(feature = "cuda")]
            Inner::Pinned(b) => b.len(),
        }
    }

    /// `true` when the buffer holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The contents. For pinned memory this first waits for outstanding device
    /// work that targets the buffer.
    pub fn as_slice(&self) -> Result<&[u8], EngineError> {
        match &self.inner {
            Inner::Pageable(b) => Ok(b.as_slice()),
            #[cfg(feature = "cuda")]
            Inner::Pinned(b) => b.as_slice(),
        }
    }

    /// Mutable contents (see [`Self::as_slice`]).
    pub fn as_mut_slice(&mut self) -> Result<&mut [u8], EngineError> {
        match &mut self.inner {
            Inner::Pageable(b) => Ok(b.as_mut_slice()),
            #[cfg(feature = "cuda")]
            Inner::Pinned(b) => b.as_mut_slice(),
        }
    }

    /// The pinned allocation, if this buffer is pinned.
    #[cfg(feature = "cuda")]
    pub fn as_pinned(&self) -> Option<&crate::pinned::PinnedBuf> {
        match &self.inner {
            Inner::Pinned(b) => Some(b),
            Inner::Pageable(_) => None,
        }
    }

    /// The pinned allocation, mutably, if this buffer is pinned.
    #[cfg(feature = "cuda")]
    pub fn as_pinned_mut(&mut self) -> Option<&mut crate::pinned::PinnedBuf> {
        match &mut self.inner {
            Inner::Pinned(b) => Some(b),
            Inner::Pageable(_) => None,
        }
    }
}

impl fmt::Debug for HostAlloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostAlloc")
            .field("class", &self.class)
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FailingProvider;

    impl PinnedProvider for FailingProvider {
        fn alloc_pinned(&self, len: usize) -> Result<HostAlloc, EngineError> {
            Err(EngineError::allocation(len, 64, "no device"))
        }
    }

    #[test]
    fn falls_back_to_pageable_when_provider_fails() {
        let buf = HostAlloc::pinned_or_pageable(128, Some(&FailingProvider)).unwrap();
        assert_eq!(buf.class(), MemoryClass::Pageable);
        assert_eq!(buf.len(), 128);
        assert!(buf.as_slice().unwrap().iter().all(|&b| b == 0));
    }

    #[test]
    fn no_provider_means_pageable() {
        let mut buf = HostAlloc::pinned_or_pageable(8, None).unwrap();
        buf.as_mut_slice().unwrap()[0] = 9;
        assert_eq!(buf.as_slice().unwrap()[0], 9);
        assert_eq!(buf.class().to_string(), "pageable");
    }
}
