using Pkg

Pkg.activate("./")

using GPP_SUGRA

# want roughly 4 efolds of inflation
@time TModes.save_eom(1.67, 0.001)
@time TModes.save_f(0.001, num_mᵪ=20, num_m32=10, num_k=100)

#  @time TModes.save_eom(0.845, 0.0005)
#  @time TModes.save_f(0.0005, num_mᵪ=20, num_m32=10, num_k=100)
