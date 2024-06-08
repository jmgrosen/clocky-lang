use std::iter;
use std::mem;
use std::ptr;
use std::slice;

use bit_set::BitSet;

pub type ClockyFunc = unsafe extern "C" fn(*const Closure) -> *const ();

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
    // TODO: fix this ordering in the compiler...?
    let func: extern "C" fn(*const (), *const Closure) -> *const () = mem::transmute((*clos).func);
    func(clk, clos)
}

// TODO: warning: be wary with alignment with this... maybe should do repr(packed)?
#[repr(C)]
struct SinceLastTickClosure {
    clos: Closure,
    clock: *const ClockSet,
}

unsafe fn since_last_clock_set_tick(clock_set: &ClockSet) -> f32 {
    // TODO: should find more efficient method
    clock_set.iter().map(|clk_id| SCHEDULER.clocks[clk_id].since_last_tick()).reduce(f32::min).unwrap()
}

unsafe extern "C" fn since_last_tick_closure(self_: *const SinceLastTickClosure) -> *const Stream {
    let st: *mut Stream = mem::transmute(alloc(mem::size_of::<Stream>() as u32));
    let val: *mut f32 = mem::transmute(alloc(mem::size_of::<f32>() as u32));
    let clock_set: &ClockSet = &*(*self_).clock;
    *val = since_last_clock_set_tick(clock_set);
    (*st).head = val;
    (*st).tail = mem::transmute(self_);
    st
}

#[no_mangle]
pub unsafe extern "C" fn since_last_tick_stream(clock: *const ClockSet) -> *const Stream {
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

unsafe extern "C" fn unit_closure(_self_: *const Closure) -> *const () {
    // do we even need to allocate, really?
    alloc(4)
}

#[no_mangle]
pub static UNIT_CLOSURE: Closure = Closure {
    func: unit_closure as ClockyFunc,
    arity: 0,
};

#[no_mangle]
pub unsafe extern "C" fn wait_closure(_clk: *const ClockSet) -> *const Closure {
    // TODO: why doesn't &UNIT_CLOSURE work?
    Box::into_raw(Box::new(Closure {
        func: unit_closure as ClockyFunc,
        arity: 0,
    }))
}

#[repr(C)]
struct ScheduledClosure {
    func: ClockyFunc,
    n_args: u32,
    clos_to_call: *const Closure,
    cell_to_fill: *mut usize,
}

#[no_mangle]
unsafe extern "C" fn scheduled_closure_func(clos: *const Closure) -> *const () {
    let sched_clos: *const ScheduledClosure = mem::transmute(clos);
    let clos_to_call: *const Closure = (*sched_clos).clos_to_call;
    let res = ((*clos_to_call).func)(clos_to_call);
    let cell = slice::from_raw_parts_mut((*sched_clos).cell_to_fill, 2);
    cell[0] = 0;
    cell[1] = res as usize;
    ptr::null()
}

#[repr(C)]
struct DelayedValue {
    sentinel: usize,
    val: *const (),
}

// TODO: only currently works if source_clock ticks *strictly* before
// target_clock, but if they are triggered by the same clock, the
// closure really should run first...
#[no_mangle]
pub unsafe extern "C" fn schedule(source_clock: *const ClockSet, target_clock: *const ClockSet, clos: *const Closure) -> *const Closure {
    let sched_clos = alloc(mem::size_of::<ScheduledClosure>() as u32) as *mut ScheduledClosure;
    (*sched_clos).func = scheduled_closure_func;
    (*sched_clos).n_args = 0;
    (*sched_clos).clos_to_call = clos;
    let target_cell: *mut usize = mem::transmute(alloc((mem::size_of::<usize>() * 2) as u32));
    *target_cell = 1;
    (*sched_clos).cell_to_fill = target_cell;

    SCHEDULER.scheduled_tasks.push(Task {
        triggering_clocks: (*source_clock).clone(),
        cancelling_clocks: (*target_clock).clone(),
        clos: sched_clos as *const Closure,
    });

    let delayed_val = alloc(mem::size_of::<ScheduledClosure>() as u32) as *mut DelayedValue;
    (*delayed_val).sentinel = 0;
    (*delayed_val).val = target_cell as *const ();

    // UGH my codebase is SUCH a MESS. this "Closure" should really be
    // a Closure/DelayedValue union or something like that. i suppose
    // i should make a thunk type.
    delayed_val as *const Closure
}

#[no_mangle]
pub unsafe extern "C" fn get_clock_set(clk_id: ClockId) -> *const ClockSet {
    Box::into_raw(Box::new(iter::once(clk_id).collect()))
}

enum Clock {
    Audio,
    Periodic { period: f32, remaining: f32 },
}

impl Clock {
    /// returns true if it ticks during this time. eventually we
    /// should make it return a timedelta since when it should have
    /// ticked or something.
    fn pass_time(&mut self, dur: f32) -> bool {
        match self {
            &mut Clock::Audio =>
                true,
            &mut Clock::Periodic { period, ref mut remaining } => {
                // TODO: offer different strategies for catching up, etc.
                if dur > *remaining {
                    *remaining = period - (dur - *remaining) % period;
                    true
                } else {
                    *remaining -= dur;
                    false
                }
            },
        }
    }

    // TODO: ??? this doesn't make sense
    fn since_last_tick(&self) -> f32 {
        match self {
            &Clock::Audio => 1. / 48e3,
            &Clock::Periodic { period, .. } => period, // sigh, jessie
        }
    }
}

type ClockId = usize;
type ClockSet = BitSet;

struct Task {
    triggering_clocks: ClockSet,
    cancelling_clocks: ClockSet,
    clos: *const Closure,
}

impl Task {
    unsafe fn run(&self) {
        let clos = self.clos;
        ((*clos).func)(clos);
    }
}

// so, so, so inefficient. but let's start here.
struct Scheduler {
    clocks: Vec<Clock>,
    scheduled_tasks: Vec<Task>,
}

static mut SCHEDULER: Scheduler = Scheduler {
    clocks: Vec::new(),
    scheduled_tasks: Vec::new(),
};

#[no_mangle]
pub unsafe extern "C" fn init_scheduler() {
    SCHEDULER.clocks.push(Clock::Audio);
}

#[no_mangle]
pub unsafe extern "C" fn make_clock(freq: f32) -> *const ClockSet {
    let clock_id = SCHEDULER.clocks.len();
    let period = 1. / freq;
    SCHEDULER.clocks.push(Clock::Periodic { period, remaining: period });
    get_clock_set(clock_id)
}

#[no_mangle]
pub unsafe fn step_scheduler(dur: f32) {
    let mut clocks_ticked = BitSet::new();
    for (i, cl) in SCHEDULER.clocks.iter_mut().enumerate() {
        if cl.pass_time(dur) {
            clocks_ticked.insert(i);
        }
    }
    SCHEDULER.scheduled_tasks.retain_mut(|task| {
        if task.triggering_clocks.is_disjoint(&clocks_ticked) {
            task.cancelling_clocks.is_disjoint(&clocks_ticked)
        } else {
            task.run();
            false
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn sample_scheduler(mut s: *const Stream, n: u32, out_ptr: *mut f32) -> *const Stream {
    // assumes s is an audio-rate stream
    //
    // TODO: this is not just inefficient, but incorrect -- we should
    // respect ordering of clocks ticking, but right now, if they both
    // happen within the same sample, they're ordered
    // arbitrarily. this is perhaps kind of unimportant right now, but
    // will be more significant if we do something slower than audio
    // rate. it also assumes each clock ticks at most once per
    // sample. aggghhh just let this be good enough for now jessie!!!!
    let out = slice::from_raw_parts_mut(out_ptr, n as usize);
    for ox in out.iter_mut() {
        *ox = hd_stream(s);
        s = adv_stream(s);
        step_scheduler(1.0 / 48e3);
    }
    s
}
