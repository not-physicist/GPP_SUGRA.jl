using GPP_SUGRA
using Test


@testset "GPP-SUGRA.jl" begin
    @test GPP_SUGRA.SmallFields.save_ode()
    @test GPP_SUGRA.SmallFields.test_save_f()
    # clean up
    rm("data/", recursive=true)
end
