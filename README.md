# Clocky

Clocky is a very-WIP functional-reactive programming language with type-level clocks. It's primarily inspired by [Async RaTT](https://dl.acm.org/doi/10.1145/3607847) and [Rhine](https://dl.acm.org/doi/10.1145/3242744.3242757).

The idea is to eventually allow audio programming that can mix high-level control with low-level synthesis and analysis. Performance is nowhere near good enough for that yet, though.

Clocky is currently implemented with a bespoke compiler to WebAssembly. You can play with the language in its [sketchbook interface](https://jessie.grosen.systems/clocky-lang/).

## Language overview

Let's start by looking at the type of that classic functional building block, `map`:
```
for a : type. for b : type. for k : clock.
  [](a -> b) -> ~^(k) a -> ~^(k) b
```
In a more mathematical notation, we might write the body of that type as
```math
\square (a \to b) \to {\sim}^k\, a \to {\sim}^k\, b
```

${\sim}^k\, a$ is a stream of values of type $a$ that runs at clock $k$. What this means is that we can get access to the current value of the stream right now, but we have to wait until clock $k$ ticks to get the rest of the stream. Clocky has a notion of "time" built-in--we can reason about what we can do *now* versus *later*.

Accessing further parts of a stream is something we can only do later, but there are some things we can only do now, like accessing a function we have lying around: that function might depend on precisely-timed streams! This is why `map` needs a "boxed" function $\square (a \to b)$, which ensures that it is time-invariant and can continue to be called in the future.

## Local installation/development

Install a Rust nightly toolchain for both your host architecture as well as wasm32-unknown-unknown, then run
```sh
cargo +nightly -Z bindeps build --release
```
To build the sketchbook interface, install wasm-pack and npm, then run
```sh
wasm-pack build
cd sketching
npm run build
```
