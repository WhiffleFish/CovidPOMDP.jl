@testset "MDP Simulation" begin
    pomdp = SingleCovidPOMDP()

    b0 = initialstate(pomdp)
    s0 = rand(b0)
    a = rand(actions(pomdp))

    hist = simulate(pomdp, s0)
    validate_history(hist)
end

@testset "POMDP Simulation" begin
    testing_props = [0,0.5,1.0]
    pomdp = SingleCovidPOMDP(actions = CovidPOMDPs.CovidActionSpace(testing_props))
    A = actions(pomdp)
    @assert length(A) == length(testing_props)
    @assert all(isa.(A, CovidAction))

    sol = CovidPOMDPs.ProportionalControlSolver()
    planner = solve(sol, pomdp)

    b0 = initialstate(pomdp, Distributions.Uniform(1,10_000))
    s0 = rand(b0)
    sir = CovidPOMDPs.SIR(s0)
    @test all(≥(0), sir)
    @test sum(sir) == pomdp.N

    a = rand(actions(pomdp))

    hist = simulate(
        pomdp,
        b0,
        planner;
        T = 50,
        upd = BootstrapFilter,
        n_p = 1_000,
        progress = true
    )
    validate_history(hist)
end
