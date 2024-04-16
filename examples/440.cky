let persamp = (div (mul 440.0 (mul 2.0 pi)) 48000.0) in
((&s. \x. sin x :: !s (add x persamp)) : sample -> ~sample) 0.0