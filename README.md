# GPP-SUGRA

[![Build Status](https://github.com/not-physicist/GPP-SUGRA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/not-physicist/GPP-SUGRA.jl/actions/workflows/CI.yml?query=branch%3Amain)

## TODO

## Performance improvement using TModes.solve_f_benchmark()

- Before rewriting interpolation for omega's: `6.888595 seconds (81.68 k allocations: 3.817 GiB, 1.36% gc time)`
- After: a bit longer, ~2.4 GiB allocation
- Now: `9.314240 seconds (81.92 k allocations: 2.349 GiB, 0.13% gc time)` or with Interpolation: `8.851697 seconds (102.15 k allocations: 4.439 GiB, 1.19% gc time, 0.03% compilation time)`
- Use fuse vectorization and static arrays: no real performance changes
- Use built-in solution handing (instead of sol.u), no real improvement
- Use `@inbounds` for solve_diff, a bit improvement: 9.224772 seconds (82.03 k allocations: 2.348 GiB, 0.44% gc time)
- Use `@inbounds` for `save_each`'s, a bit improvement, but expect more improvement in real usage: 9.187108 seconds (81.91 k allocations: 2.348 GiB, 0.63% gc time)
- Ensure type stability: 9.387 s (51368 allocations: 2.35 GiB)
- Now seems the best performance: Single result which took 9.279 s (0.68% GC) to evaluate, with a memory estimate of 2.35 GiB, over 79753 allocations.
