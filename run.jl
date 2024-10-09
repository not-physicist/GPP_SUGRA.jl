using Pkg
Pkg.activate("./")

using GPP_SUGRA

# @time TModes.save_eom_benchmark()
# @time TModes.save_f_benchmark()
# @time TModes.save_f_benchmark2()

# @time TModes.save_eom(3.0, 0.001, "data/TMode-0.001/", 1e4)
# @time TModes.save_f(0.001, num_mᵪ=20, num_k=50)

# @time TModes.save_eom(3.6, 0.0035, "data/TMode-0.0035/", 5e3)
# @time TModes.save_f(0.0035, "data/TMode-0.0035/", num_mᵪ=20, num_k=50)
# @time TModes.save_m_eff(0.0035, "data/TMode-0.0035/")

@time TModes.save_eom(1.5, 0.0001, "data/TMode-0.0001/", 1e3)
@time TModes.save_f(0.0001, "data/TMode-0.0001/", num_mᵪ=10, num_k=20)
