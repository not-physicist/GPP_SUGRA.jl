using Pkg

Pkg.activate("./")

using GPP_SUGRA

@time TModes.save_eom(3.6, 0.0035, "data/TMode-0.0035/")
@time TModes.save_f(0.0035, "data/TMode-0.0035/", num_mᵪ=10, num_k=50)
