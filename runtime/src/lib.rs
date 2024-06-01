#![no_std]

use core::arch::wasm32;
use core::ptr;
use core::mem;
use core::slice;

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

#[no_mangle]
pub unsafe extern "C" fn sample(mut s: *const Stream, n: u32, out_ptr: *mut f32) -> *const Stream {
    let out = slice::from_raw_parts_mut(out_ptr, n as usize);
    for i in 0..n as usize {
        *out.get_unchecked_mut(i) = hd_stream(s);
        s = adv_stream(s);
    }
    s
}

#[no_mangle]
pub unsafe extern "C" fn apply_clock(clos: *const Closure, clk: *const ()) -> *const () {
    assert_eq!((*clos).arity, 1);
    let func: extern "C" fn(*const Closure, *const ()) -> *const () = mem::transmute((*clos).func);
    func(clos, clk)
}

#[cfg(not(test))]
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

static mut HEAP_PTR: *mut () = ptr::null_mut();
static mut HEAP_END: *mut () = ptr::null_mut();

// ??
const ALLOC_SIZE_PAGES: usize = 4;

#[no_mangle]
pub unsafe extern "C" fn alloc(n: u32) -> *mut () {
    let old_heap_ptr = HEAP_PTR;
    if let Some(new_heap_ptr) = (HEAP_PTR as usize).checked_add(n as usize) {
        if new_heap_ptr < HEAP_END as usize {
            HEAP_PTR = new_heap_ptr as *mut ();
            old_heap_ptr
        } else {
            let old_size = wasm32::memory_grow(0, ALLOC_SIZE_PAGES);
            if old_size < usize::MAX {
                let allocation = (old_size * 65536) as *mut ();
                HEAP_PTR = (old_size * 65536 + n as usize) as *mut ();
                HEAP_END = ((old_size + ALLOC_SIZE_PAGES) * 65536) as *mut ();
                allocation
            } else {
                panic!()
            }
        }
    } else {
        panic!()
    }
}
