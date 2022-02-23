@testset "POMCPOW" begin
    pomdp = CovidPOMDP()
    b0 = initialstate(pomdp)

    sol = POMCPOWSolver(
        max_depth = 50,
        tree_queries = 1_000,
        check_repeat_act = false,
        check_repeat_obs = false,
        enable_action_pw = true,
        k_action = 5,
        alpha_action = 0.1,
        k_observation = 2,
        alpha_observation = 0.1
    )
    planner = solve(sol, pomdp)

    a = action(planner, b0)
    @test a isa CovidAction
    @test 0. ≤ a.testing_prop ≤ 1.
end
