@testset "POMCPOW" begin
    pomdp = SingleCovidPOMDP(actions = CovidPOMDPs.CovidActionSpace(7;zero=true))
    b0 = initialstate(pomdp)

    sol = POMCPOWSolver(
        criterion = MaxUCB(2.0),
        max_depth = 30,
        tree_queries = 1_000,
        check_repeat_act = true,
        check_repeat_obs = false,
        enable_action_pw = false,
        k_observation = 3,
        alpha_observation = 0.05,
        estimate_value = CovidPOMDPs.ProportionalControlSolver(10.0)
    )

    planner = solve(sol, pomdp)

    a = action(planner, b0)
    @test a isa CovidAction
    @test 0. ≤ a.testing_prop ≤ 1.

    hist = simulate(
        pomdp,
        b0,
        planner;
        T = 50,
        upd = CovidPOMDPs.NoisyCovidFilter,
        n_p = 1_000,
        progress = false
    )

    validate_history(hist)
end
