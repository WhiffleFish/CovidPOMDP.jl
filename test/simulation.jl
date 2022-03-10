
@testset "MDP Simulation" begin
    pomdp = CovidPOMDP()

    b0 = initialstate(pomdp)
    s0 = rand(b0)
    a = rand(actions(pomdp))

    hist = simulate(pomdp, s0)
    S = hist.sus
    I = hist.inf
    R = hist.rec
    N = hist.N
    T = hist.T

    # nonnegativity
    @test all(≥(0), S)
    @test all(≥(0), I)
    @test all(≥(0), R)
    @test all(≥(0), hist.pos_test)

    # Population size preservation
    @test all(==(N), S[i] + I[i] + R[i] for i in 1:T)

    # @testinferred CovidPOMDPs.sim_step(pomdp, s0, a)
end

@testset "POMDP Simulation" begin
    pomdp = CovidPOMDP()
    sol = CovidPOMDPs.ProportionalControlSolver()
    planner = solve(sol, pomdp)

    b0 = initialstate(pomdp, Distributions.Uniform(1,10_000))
    s0 = rand(b0)
    a = rand(actions(pomdp))

    hist = simulate(pomdp, s0)
    S = hist.sus
    I = hist.inf
    R = hist.rec
    N = hist.N
    T = hist.T

    # nonnegativity
    @test all(≥(0), S)
    @test all(≥(0), I)
    @test all(≥(0), R)
    @test all(≥(0), hist.pos_test)
    @test all( 0.0 .≤ getfield.(hist.actions,:testing_prop) .≤ 1.0)

    # Population size preservation
    @test all(==(N), S[i] + I[i] + R[i] for i in 1:T)
end
