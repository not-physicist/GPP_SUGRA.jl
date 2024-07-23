using Pkg

Pkg.activate("./")

using GPP_SUGRA

@time TModes.save_eom_benchmark()
@time TModes.save_f_benchmark()

# want roughly 4 efolds of inflation
#  @time TModes.save_eom(1.67, 0.001, "data/TMode-0.001/")
#  @time TModes.save_f(0.001, num_mᵪ=20, num_m32=10, num_k=100)


#  @time TModes.save_eom(1.55, 0.001, "data/TMode-0.001-short/")
 # @time TModes.save_f(0.001, "data/TMode-0.001-short/", num_mᵪ=20, num_m32=5, num_k=100)

 # @time TModes.save_eom(1.55, 0.001, "data/TMode-0.001-new/")
 # @time TModes.save_f(0.001, "data/TMode-0.001-new/", num_mᵪ=5, num_m32=5, num_k=20)

#  @time TModes.save_eom(0.845, 0.0005)
#  @time TModes.save_f(0.0005, num_mᵪ=20, num_m32=10, num_k=100)

# @time TModes.save_eom(0.33, 0.0001)
#  @time TModes.save_f(0.0001, num_mᵪ=20, num_m32=5, num_k=100)
