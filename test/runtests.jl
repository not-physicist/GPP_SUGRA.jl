using GPP_SUGRA
using Test

@testset "SmallFields.jl" begin
    @test SmallFields.save_ode()
    @test SmallFields.test_save_f()

    # clean up
    rm("data/", recursive=true)
end

@testset "TModes.jl" begin
    @test TModes.ModelDatas.test_ϕₑ()
    @test TModes.save_ode()
    @test TModes.test_save_f()

    # clean up
    rm("data/", recursive=true)
end
