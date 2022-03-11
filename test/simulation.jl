function validate_history(hist::SimHist)
    S = hist.sus
    I = hist.inf
    R = hist.rec
    N = hist.N
    T = hist.T

    @test all(≥(0), S)
    @test all(≥(0), I)
    @test all(≥(0), R)
    @test all(≥(0), hist.pos_test)

    # Population size preservation
    @test all(==(N), S[i] + I[i] + R[i] for i in 1:T)

    @test all( 0.0 .≤ getfield.(hist.actions,:testing_prop) .≤ 1.0)

    return nothing
end

@testset "Initialization" begin
    dists = CovidPOMDPs.FitInfectionDistributions(
        CovidPOMDPs.INF_DATAFRAME,
        CovidPOMDPs.INFECTION_HORIZON,
        50
    )
    @test @isinferred first(dists)

    test_probs = [
        CovidPOMDPs.prop_above_LOD(CovidPOMDPs.VIRAL_LOADS, day, CovidPOMDPs.LIMIT_OF_DETECTION)
        for day in 1:CovidPOMDPs.INFECTION_HORIZON
    ]

    @test @isinferred first(test_probs)
    @test all(≥(0.), test_probs)
end

@testset "MDP Simulation" begin
    pomdp = CovidPOMDP()

    b0 = initialstate(pomdp)
    s0 = rand(b0)
    a = rand(actions(pomdp))

    hist = simulate(pomdp, s0)
    validate_history(hist)
end

@testset "POMDP Simulation" begin
    pomdp = CovidPOMDP()
    sol = CovidPOMDPs.ProportionalControlSolver()
    planner = solve(sol, pomdp)

    b0 = initialstate(pomdp, Distributions.Uniform(1,10_000))
    s0 = rand(b0)
    a = rand(actions(pomdp))

    hist = simulate(
        pomdp,
        b0,
        planner;
        T = 50,
        upd = BootstrapFilter,
        n_p = 1_000,
        progress=true
    )
    validate_history(hist)
end
