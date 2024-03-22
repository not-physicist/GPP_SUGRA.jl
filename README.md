# GPP-SUGRA

[![Build Status](https://github.com/not-physicist/GPP-SUGRA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/not-physicist/GPP-SUGRA.jl/actions/workflows/CI.yml?query=branch%3Amain)

## TODO

- ~~(Try to) Use number of efolds as time variable for inflation~~
- ~~Improve memory allocation for init_func in pp.jl: interpolate m and dm for all k~~
- ~~Maybe one can have better performance in pp.jl~~

## Performance improvement

- Before rewriting interpolation for omega's: `6.888595 seconds (81.68 k allocations: 3.817 GiB, 1.36% gc time)`
- After: a bit longer, ~2.4 GiB allocation
- Now: `9.314240 seconds (81.92 k allocations: 2.349 GiB, 0.13% gc time)`
