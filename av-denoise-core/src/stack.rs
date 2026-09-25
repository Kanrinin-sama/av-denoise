use std::ffi::OsStr;

/// Stack bytes the kernel codegen thread needs.
pub const CODEGEN_STACK_BYTES: usize = 16 << 20;

/// Raises `RUST_MIN_STACK` to [`CODEGEN_STACK_BYTES`] when it is unset.
///
/// No-op if the variable is already set.
///
/// # Safety
///
/// The same safety rules as [std::env::set_var] applies.
pub unsafe fn raise_codegen_stack_limit() {
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        // SAFETY: forwarded from this function's own precondition.
        unsafe { std::env::set_var("RUST_MIN_STACK", CODEGEN_STACK_BYTES.to_string()) };
    }
}

/// Whether the process's `RUST_MIN_STACK` is large enough for codegen.
pub fn codegen_stack_is_sufficient() -> bool {
    limit_is_sufficient(std::env::var_os("RUST_MIN_STACK").as_deref())
}

/// Parses a raw `RUST_MIN_STACK` value and checks it against [`CODEGEN_STACK_BYTES`].
///
/// Unset is insufficient. A value that fails to parse is also insufficient.
fn limit_is_sufficient(raw: Option<&OsStr>) -> bool {
    raw.and_then(|v| v.to_str())
        .and_then(|v| v.parse::<usize>().ok())
        .is_some_and(|bytes| bytes >= CODEGEN_STACK_BYTES)
}
