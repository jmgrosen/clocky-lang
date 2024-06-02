use std::mem;
use std::slice;

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

#[repr(C)]
pub struct Clock {
    since_last_tick: f32,
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
    // TODO: fix this ordering in the compiler...?
    let func: extern "C" fn(*const (), *const Closure) -> *const () = mem::transmute((*clos).func);
    func(clk, clos)
}

// TODO: warning: be wary with alignment with this... maybe should do repr(packed)?
#[repr(C)]
struct SinceLastTickClosure {
    clos: Closure,
    clock: *const Clock,
}

unsafe extern "C" fn since_last_tick_closure(self_: *const SinceLastTickClosure) -> *const Stream {
    let st: *mut Stream = mem::transmute(alloc(mem::size_of::<Stream>() as u32));
    let val: *mut f32 = mem::transmute(alloc(mem::size_of::<f32>() as u32));
    *val = (*(*self_).clock).since_last_tick;
    (*st).head = val;
    (*st).tail = mem::transmute(self_);
    st
}

#[no_mangle]
pub unsafe extern "C" fn since_last_tick_stream(clock: *const Clock) -> *const Stream {
    let clos: *mut SinceLastTickClosure = mem::transmute(alloc(mem::size_of::<SinceLastTickClosure>() as u32));
    (*clos).clos.func = mem::transmute(since_last_tick_closure as unsafe extern "C" fn(*const SinceLastTickClosure) -> *const Stream);
    (*clos).clos.arity = 0;
    (*clos).clock = clock;
    since_last_tick_closure(clos)
}

#[no_mangle]
pub extern "C" fn sin(x: f32) -> f32 {
    libm::sinf(x)
}

#[no_mangle]
pub extern "C" fn cos(x: f32) -> f32 {
    libm::cosf(x)
}

#[no_mangle]
pub unsafe extern "C" fn alloc(n: u32) -> *mut () {
    // TODO: obviously so much. but right now i'm thinking about the
    // alignment
    std::alloc::alloc(std::alloc::Layout::from_size_align(n as usize, 4).unwrap()) as *mut ()
}
