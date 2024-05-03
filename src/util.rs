use std::{fmt, mem::MaybeUninit};

use typed_arena::Arena;

pub fn parenthesize(f: &mut fmt::Formatter<'_>, p: bool, inner: impl FnOnce(&mut fmt::Formatter<'_>) -> fmt::Result) -> fmt::Result {
    if p {
        write!(f, "(")?;
        inner(f)?;
        write!(f, ")")
    } else {
        inner(f)
    }
}

pub struct ArenaPlus<'a, T> {
    pub arena: &'a Arena<T>,
    pub ptr_arena: &'a Arena<&'a T>,
}

impl<'a, T> ArenaPlus<'a, T> {
    pub fn alloc(&self, e: T) -> &'a T {
        self.arena.alloc(e)
    }

    pub fn alloc_slice(&self, it: impl IntoIterator<Item=&'a T>) -> &'a [&'a T] {
        self.ptr_arena.alloc_extend(it)
    }

    /// A form of alloc_slice that is reentrant: you can use the arena
    /// in the iterator. It requires the iterator to return a correct
    /// upper bound size hint.
    pub fn alloc_slice_r(&self, it: impl IntoIterator<Item=&'a T>) -> &'a [&'a T] {
        // TODO: is this safe? my gut says, "not technically but
        // probably fine in practice." this will end up with
        // uninitialized references within ptr_arena, which rust
        // sounds like it *really* does not like, but it's
        // prooooooobably fine, right? perhaps this could be fixed by
        // changing ptr_arena to instead allocate MaybeUninit<&'a
        // T>?
        let it = it.into_iter();
        let (_, Some(bound)) = it.size_hint() else {
            panic!("alloc_slice_r called with iterator that does not return an upper bound")
        };

        unsafe {
            let uninit = self.ptr_arena.alloc_uninitialized(bound);
            // don't use .enumerate() bc we need the count afterwards
            let mut i = 0;
            for e in it {
                uninit[i].write(e);
                i += 1;
            }

            let valid_portion = &uninit[..i];
            &*(valid_portion as *const [MaybeUninit<&'a T>] as *const [&'a T])
        }
    }

    #[allow(unused)]
    pub fn alloc_slice_alloc(&self, it: impl IntoIterator<Item=T>) -> &'a [&'a T] {
        self.alloc_slice(it.into_iter().map(|e| self.alloc(e)))
    }

    #[allow(unused)]
    pub fn alloc_slice_maybe(&self, it: impl IntoIterator<Item=Option<&'a T>>) -> Option<&'a [&'a T]> {
        let mut success = true;
        let slice = self.ptr_arena.alloc_extend(it.into_iter().map_while(|e| { success = e.is_some(); e }));
        if success { Some(slice) } else { None }
    }

    /// A form of alloc_slice that is reentrant: you can use the arena
    /// in the iterator. It requires the iterator to return a correct
    /// upper bound size hint.
    #[allow(unused)]
    pub fn alloc_slice_maybe_r(&self, it: impl IntoIterator<Item=Option<&'a T>>) -> Option<&'a [&'a T]> {
        // TODO: is this safe? my gut says, "not technically but
        // probably fine in practice." this will end up with
        // uninitialized references within ptr_arena, which rust
        // sounds like it *really* does not like, but it's
        // prooooooobably fine, right? perhaps this could be fixed by
        // changing ptr_arena to instead allocate MaybeUninit<&'a
        // T>?
        let it = it.into_iter();
        let (_, Some(bound)) = it.size_hint() else {
            panic!("alloc_slice_r called with iterator that does not return an upper bound")
        };

        unsafe {
            let uninit = self.ptr_arena.alloc_uninitialized(bound);
            // don't use .enumerate() bc we need the count afterwards
            let mut i = 0;
            for oe in it {
                if let Some(e) = oe {
                    uninit[i].write(e);
                    i += 1;
                } else {
                    return None;
                }
            }

            let valid_portion = &uninit[..i];
            Some(&*(valid_portion as *const [MaybeUninit<&'a T>] as *const [&'a T]))
        }
    }

    #[allow(unused)]
    fn alloc_slice_maybe_alloc(&self, it: impl IntoIterator<Item=Option<T>>) -> Option<&'a [&'a T]> {
        self.alloc_slice_maybe(it.into_iter().map(|oe| oe.map(|e| self.alloc(e))))
    }
}
