//! Buffer layout descriptors and the alignment constants every allocator honours.

use crate::EngineError;

/// Alignment of host buffers: one cache line, also the AVX-512 vector width.
pub const HOST_ALIGN: usize = 64;

/// Alignment of buffers bound for a device: one coalesced global-memory transaction.
pub const DEVICE_ALIGN: usize = 128;

/// Length and alignment of a buffer, with helpers for padding and validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferLayout {
    len: usize,
    align: usize,
}

impl BufferLayout {
    /// Builds a layout, validating that `align` is a non-zero power of two.
    pub fn new(len: usize, align: usize) -> Result<Self, EngineError> {
        if align == 0 || !align.is_power_of_two() {
            return Err(EngineError::allocation(
                len,
                align,
                "alignment must be a non-zero power of two",
            ));
        }
        Ok(Self { len, align })
    }

    /// A host-aligned ([`HOST_ALIGN`]) layout.
    pub const fn host(len: usize) -> Self {
        Self {
            len,
            align: HOST_ALIGN,
        }
    }

    /// A device-aligned ([`DEVICE_ALIGN`]) layout.
    pub const fn device(len: usize) -> Self {
        Self {
            len,
            align: DEVICE_ALIGN,
        }
    }

    /// The requested length in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` when the buffer holds no bytes.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The alignment in bytes.
    pub const fn align(&self) -> usize {
        self.align
    }

    /// The length rounded up to a multiple of the alignment (zero stays zero).
    pub const fn padded_len(&self) -> usize {
        let rem = self.len % self.align;
        if rem == 0 {
            self.len
        } else {
            self.len + (self.align - rem)
        }
    }

    /// `true` when `addr` satisfies this layout's alignment.
    pub const fn is_aligned(&self, addr: usize) -> bool {
        addr.is_multiple_of(self.align)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_alignment() {
        assert!(BufferLayout::new(8, 0).is_err());
        assert!(BufferLayout::new(8, 48).is_err());
        assert!(BufferLayout::new(8, 64).is_ok());
    }

    #[test]
    fn pads_to_alignment() {
        assert_eq!(BufferLayout::host(0).padded_len(), 0);
        assert_eq!(BufferLayout::host(1).padded_len(), 64);
        assert_eq!(BufferLayout::host(64).padded_len(), 64);
        assert_eq!(BufferLayout::device(129).padded_len(), 256);
    }

    #[test]
    fn checks_addresses() {
        assert!(BufferLayout::device(1).is_aligned(0x1000));
        assert!(!BufferLayout::device(1).is_aligned(0x1040));
        assert!(BufferLayout::host(1).is_aligned(0x1040));
    }
}
