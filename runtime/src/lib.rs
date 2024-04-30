#![no_std]

pub type ClockyFunc = extern "C" fn(*const Closure) -> *const ();

#[repr(C)]
pub struct Closure {
    func: ClockyFunc,
    arity: u32,
    // it looks like i can't encode this right now:
    // https://github.com/rust-lang/rust/issues/43467
    //
    // env: [u32],
}

// monomorphic for now!
#[repr(C)]
pub struct Stream {
    head: *const f32,
    tail: *const Closure,
}

#[no_mangle]
pub unsafe extern "C" fn hd_stream(s: *const Stream) -> f32 {
    *(*s).head
}

#[no_mangle]
pub unsafe extern "C" fn adv_stream(s: *const Stream) -> *const Stream {
    let tail = (*s).tail;
    ((*tail).func)(tail) as *const Stream
}

static mut BUF: [f32; 1024] = [0.0; 1024];

#[no_mangle]
pub unsafe extern "C" fn sample(mut s: *const Stream, _n: u32) -> *const f32 {
    for i in 0..1024 {
        BUF[i] = hd_stream(s);
        s = adv_stream(s);
    }
    BUF.as_ptr()
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}

#[no_mangle]
pub extern "C" fn sin(x: f32) -> f32 {
    libm::sinf(x)
}

#[no_mangle]
pub extern "C" fn cos(x: f32) -> f32 {
    libm::cosf(x)
}
