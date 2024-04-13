let mapsig: (sample -> sample) -> ~^(k) sample -> ~^(k) sample =
  \f. (&^(k) mappedsig. \sigin.
         let (x, siginp) = *sigin in
         f x :: !mappedsig !siginp)
  in
let persamp = (div (mul 440.0 (mul 2.0 pi)) 48000.0) in
let lin = ((&^(k) s. \x. x :: !s (add x persamp)) : sample -> ~^(k) sample) 0.0 in
mapsig sin lin
