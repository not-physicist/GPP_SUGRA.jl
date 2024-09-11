using Pkg

Pkg.activate("./")

using GPP_SUGRA

@time TModes.save_eom(3.5, 0.0035, "data/TMode-0.0035/")
