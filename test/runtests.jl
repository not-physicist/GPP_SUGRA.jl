using GPP_SUGRA
using Test, MultiQuad

#=
@testset "SmallFields.jl" begin
    @test SmallFields.save_ode()
    @test SmallFields.test_save_f()

    # clean up
    rm("data/", recursive=true)
end

@testset "TModes.jl" begin
    dn = "data/TMode-test"
    @test TModes.ModelDatas.test_ϕₑ()
    @test TModes.save_ode()
    @test TModes.test_save_f()

    # clean up
    rm("data/", recursive=true)
end
=# 

@testset "common.jl" begin
    func(y, x) = sin(x) * y^2
    int, err = dblquad(func, 1, 2, x->0, x->x^2, rtol=1e-5)

    X = collect(range(1, 5, 10000))
    Y = collect(range(0, 30, 10000))
    int2 = GPP_SUGRA.Commons.double_trap((i, j) -> func(Y[i], X[j]), 1, 2, x -> 0, x -> x^2, X, Y)

    @test isapprox(int, int2, rtol=1e-2)
    @test isapprox(dblquad(func, 1, 5, x->0, x->x^2, rtol=1e-5)[1], 
                    GPP_SUGRA.Commons.double_trap((i, j) -> func(Y[i], X[j]), 1, 5, x -> 0, x -> x^2, X, Y), rtol=1e-2)
end
