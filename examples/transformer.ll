let sinsig =
  ((&sinsig. \sigin.
    let (x, siginp) = *sigin in
    sin x :: !sinsig !siginp)
  : ~sample -> ~sample) in
let persamp = (div (mul 440.0 (mul 2.0 pi)) 48000.0) in
let lin = ((&s. \x. x :: !s (add x persamp)) : sample -> ~sample) 0.0 in
sinsig lin
