//! # Raw Slice Types
//!
//! This crate provides two generic raw slice type, [RawSlice] and [RawSliceMut], which allow
//! erasing the lifetime of a borrowed slice.
//!
//! ## Motivation
//!
//! In Rust, lifetimes are a powerful tool for ensuring memory safety at compile time.
//! However, there are cases where lifetimes become too restrictive, such as when working
//! with borrowed data across asynchronous or interrupt-driven contexts.
//!
//! This data structure is particularly useful in embedded systems, where data may be
//! passed to asynchronous peripherals such as serial TX drivers using interrupts or DMA.
//! The data may be static, but it could also reside on the stack. By using a shared [RawBufSlice],
//! you can pass borrowed data to a driver **without** needing to explicitly manage lifetimes.
//!
//! ## Safety Considerations
//!
//! - **No Lifetime Tracking:** Since [RawSlice] erases lifetimes, **the caller must ensure**
//!   that the referenced data remains valid while the [RawSlice] is in use.
//! - **Concurrency Risks:** Accessing the same underlying data from multiple contexts
//!   (e.g., an ISR and a task) requires proper synchronization.
//! - **Immutability:** [RawSlice] provides a **read-only view** of the data. If you need
//!   mutability, [RawSliceMut] can be used.
//!
//! ## Usage Example
//!
//! ```rust
//! use raw_slice::RawBufSlice;
//!
//! static DATA: &[u8] = &[1, 2, 3, 4];
//!
//! let raw_buf = unsafe { RawBufSlice::new(DATA) };
//!
//! // Later, in an ISR or different context
//! unsafe {
//!     if let Some(slice) = raw_buf.get() {
//!         // Process the data, e.g. send it via serial interface
//!         // self.rx.write(slice);
//!     }
//! }
//! ```
//!
//! ## API Design
//!
//! While this crate provides methods to interact with the stored data, most of these operations
//! remain `unsafe` due to the compiler's inability to enforce lifetimes after erasure. Users should
//! carefully ensure the referenced data remains valid for the required duration. In addition
//! to the concept of a slice being empty, a raw slice can also be NULL.
//!
//! ## Embedded DMA Support
//!
//! - The [RawBufSlice] structure implements the [embedded_dma::ReadBuffer] trait
//! - The [RawBufSliceMut] structure implements the [embedded_dma::WriteBuffer] trait
#![no_std]

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RawSlice<T> {
    data: *const T,
    len: usize,
}

/// Safety: This type MUST be used with mutex to ensure concurrent access is valid.
unsafe impl<T: Send> Send for RawSlice<T> {}

impl<T> RawSlice<T> {
    /// Creates a new `RawSlice<T>` from a slice reference.
    ///
    /// # Safety
    ///
    /// - The caller **must** ensure that the slice outlives this `RawSlice<T>`.
    /// - The original slice **must not** be mutated while this `RawSlice<T>` is used.
    #[allow(dead_code)]
    pub const unsafe fn new(data: &[T]) -> Self {
        Self {
            data: data.as_ptr(),
            len: data.len(),
        }
    }

    /// Creates an empty `RawSlice<T>`, equivalent to a null pointer with zero length.
    pub const fn new_nulled() -> Self {
        Self {
            data: core::ptr::null(),
            len: 0,
        }
    }

    /// Updates the raw pointer and length to point to a new slice.
    ///
    /// # Safety
    ///
    /// - The caller **must** ensure that the slice outlives this `RawSlice<T>`.
    /// - The original slice **must not** be mutated while this `RawSlice<T>` is used.
    pub const unsafe fn set(&mut self, data: &[T]) {
        self.data = data.as_ptr();
        self.len = data.len();
    }

    /// Set the internal data pointer to NULL and also clears the data length.
    pub const fn set_null(&mut self) {
        self.data = core::ptr::null();
        self.len = 0;
    }

    /// Check whether the internal data pointer is NULL.
    pub const fn is_null(&self) -> bool {
        self.data.is_null()
    }

    /// Converts the raw pointer into a slice.
    ///
    /// Returns [None] if the pointer is null.
    ///
    /// # Safety
    ///
    /// - The caller **must** ensure that the underlying memory is still valid.
    /// - Using this function after the original slice is dropped results in UB.
    pub const unsafe fn get(&self) -> Option<&[T]> {
        if self.data.is_null() {
            return None;
        }
        Some(unsafe { core::slice::from_raw_parts(self.data, self.len) })
    }

    /// Returns [None] if the pointer is null and whether [Self::len] is 0 otherwise.
    pub const fn is_empty(&self) -> Option<bool> {
        if self.is_null() {
            return None;
        }
        Some(self.len == 0)
    }

    /// Returns [None] if the pointer is null and the length of the raw slice otherwise.
    pub const fn len(&self) -> Option<usize> {
        if self.is_null() {
            return None;
        }
        Some(self.len)
    }
}

impl<T> Default for RawSlice<T> {
    fn default() -> Self {
        Self::new_nulled()
    }
}

pub type RawBufSlice = RawU8Slice;
pub type RawU8Slice = RawSlice<u8>;
pub type RawU16Slice = RawSlice<u16>;
pub type RawU32Slice = RawSlice<u32>;

macro_rules! impl_dma_read_buf {
    ($slice_type:ident, $ty:ident) => {
        /// This allows using [Self] in DMA APIs which expect a [embedded_dma::ReadBuffer].
        ///
        /// However, the user still must ensure that any alignment rules for DMA buffers required by
        /// the hardware are met and than any MPU/MMU configuration necessary is also performed for this
        /// to work properly.
        unsafe impl embedded_dma::ReadBuffer for $slice_type {
            type Word = $ty;

            unsafe fn read_buffer(&self) -> (*const Self::Word, usize) {
                (self.data, self.len)
            }
        }
    };
}

impl_dma_read_buf!(RawBufSlice, u8);
impl_dma_read_buf!(RawU16Slice, u16);
impl_dma_read_buf!(RawU32Slice, u32);

#[derive(Debug, Copy, Clone)]
pub struct RawSliceMut<T> {
    data: *mut T,
    len: usize,
}

/// Safety: This type MUST be used with mutex to ensure concurrent access is valid.
unsafe impl<T: Send> Send for RawSliceMut<T> {}

impl<T> RawSliceMut<T> {
    /// Creates a new `RawSlice<T>` from a slice reference.
    ///
    /// # Safety
    ///
    /// - The caller **must** ensure that the slice outlives this `RawSlice<T>`.
    /// - The original slice **must not** be mutated while this `RawSlice<T>` is used.
    #[allow(dead_code)]
    pub const unsafe fn new(data: &mut [T]) -> Self {
        Self {
            data: data.as_mut_ptr(),
            len: data.len(),
        }
    }

    /// Creates an empty `RawSlice<T>`, equivalent to a null pointer with zero length.
    pub const fn new_nulled() -> Self {
        Self {
            data: core::ptr::null_mut(),
            len: 0,
        }
    }

    /// Updates the raw pointer and length to point to a new slice.
    ///
    /// # Safety
    ///
    /// - The caller **must** ensure that the slice outlives this `RawSlice<T>`.
    /// - The original slice **must not** be mutated while this `RawSlice<T>` is used.
    pub const unsafe fn set(&mut self, data: &mut [T]) {
        self.data = data.as_mut_ptr();
        self.len = data.len();
    }

    /// Converts the raw pointer into a slice.
    ///
    /// Returns [None] if the pointer is null.
    ///
    /// # Safety
    ///
    /// - The caller **must** ensure that the underlying memory is still valid.
    /// - Using this function after the original slice is dropped results in UB.
    pub const unsafe fn get<'slice>(&self) -> Option<&'slice [T]> {
        if self.data.is_null() {
            return None;
        }
        Some(unsafe { core::slice::from_raw_parts(self.data, self.len) })
    }

    /// Converts the raw pointer into a mutable slice.
    ///
    /// Returns [None] if the pointer is null.
    ///
    /// # Safety
    ///
    /// - The caller **must** ensure that the underlying memory is still valid.
    /// - Using this function after the original slice is dropped results in UB.
    pub const unsafe fn get_mut<'slice>(&mut self) -> Option<&'slice mut [T]> {
        if self.data.is_null() {
            return None;
        }
        Some(unsafe { core::slice::from_raw_parts_mut(self.data, self.len) })
    }

    pub const fn set_null(&mut self) {
        self.data = core::ptr::null_mut();
        self.len = 0;
    }

    pub const fn is_null(&self) -> bool {
        self.data.is_null()
    }

    /// Returns [None] if the pointer is null and whether [Self::len] is 0 otherwise.
    pub const fn is_empty(&self) -> Option<bool> {
        if self.is_null() {
            return None;
        }
        Some(self.len == 0)
    }

    /// Returns [None] if the pointer is null and the length of the raw slice otherwise.
    pub const fn len(&self) -> Option<usize> {
        if self.is_null() {
            return None;
        }
        Some(self.len)
    }
}

impl<T> Default for RawSliceMut<T> {
    fn default() -> Self {
        Self::new_nulled()
    }
}

pub type RawBufSliceMut = RawU8SliceMut;
pub type RawU8SliceMut = RawSliceMut<u8>;
pub type RawU16SliceMut = RawSliceMut<u16>;
pub type RawU32SliceMut = RawSliceMut<u32>;

macro_rules! impl_dma_write_buf {
    ($slice_type:ident, $ty:ident) => {
        /// This allows using [Self] in DMA APIs which expect a [embedded_dma::WriteBuffer].
        ///
        /// However, the user still must ensure that any alignment rules for DMA buffers required by
        /// the hardware are met and than any MPU/MMU configuration necessary was also performed.
        unsafe impl embedded_dma::WriteBuffer for $slice_type {
            type Word = $ty;

            unsafe fn write_buffer(&mut self) -> (*mut Self::Word, usize) {
                (self.data, self.len)
            }
        }
    };
}

impl_dma_write_buf!(RawBufSliceMut, u8);
impl_dma_write_buf!(RawU16SliceMut, u16);
impl_dma_write_buf!(RawU32SliceMut, u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_basic() {
        let slice = [1, 2, 3, 4];
        let mut slice_raw = unsafe { RawBufSlice::new(&slice) };
        assert_eq!(slice_raw.len().unwrap(), 4);
        assert!(!slice_raw.is_null());
        assert!(!slice_raw.is_empty().unwrap());
        assert_eq!(slice_raw.len().unwrap(), 4);
        let slice_read_back = unsafe { slice_raw.get().unwrap() };
        assert_eq!(slice_read_back, slice);
        slice_raw.set_null();
        generic_empty_test(&slice_raw);
    }

    #[test]
    pub fn test_empty() {
        let empty = RawBufSlice::new_nulled();
        generic_empty_test(&empty);
    }

    #[test]
    pub fn test_empty_mut() {
        let mut empty = RawBufSliceMut::new_nulled();
        generic_empty_test_mut(&mut empty);
    }

    #[test]
    pub fn test_clonable() {
        let slice = [1, 2, 3, 4];
        let slice_raw = unsafe { RawBufSlice::new(&slice) };
        let slice_copied = slice_raw;
        assert_eq!(slice_copied, slice_raw);
    }

    #[test]
    pub fn test_basic_mut() {
        let mut slice = [1, 2, 3, 4];
        let mut slice_raw = unsafe { RawBufSliceMut::new(&mut slice) };
        assert_eq!(slice_raw.len().unwrap(), 4);
        assert!(!slice_raw.is_null());
        assert!(!slice_raw.is_empty().unwrap());
        assert_eq!(slice_raw.len().unwrap(), 4);
        let slice_read_back = unsafe { slice_raw.get().unwrap() };
        assert_eq!(slice_read_back, slice);
        let mut_slice_read_back = unsafe { slice_raw.get_mut().unwrap() };
        assert_eq!(slice_read_back, mut_slice_read_back);
        mut_slice_read_back[0] = 5;
        assert_eq!(slice[0], 5);
        slice_raw.set_null();
        generic_empty_test_mut(&mut slice_raw);
    }

    fn generic_empty_test(slice: &RawBufSlice) {
        assert!(slice.is_null());
        assert!(slice.is_empty().is_none());
        assert!(slice.len().is_none());
        assert!(unsafe { slice.get() }.is_none());
    }

    fn generic_empty_test_mut(slice: &mut RawBufSliceMut) {
        assert!(slice.is_null());
        assert!(slice.is_empty().is_none());
        assert!(slice.len().is_none());
        assert!(unsafe { slice.get() }.is_none());
        assert!(unsafe { slice.get_mut() }.is_none());
    }
}
