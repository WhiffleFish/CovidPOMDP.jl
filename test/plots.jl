@testset "Plots" begin
    pomdp = CovidPOMDP()
    sol = CovidPOMDPs.ProportionalControlSolver()
    planner = solve(sol, pomdp)

    b0 = initialstate(pomdp, Distributions.Uniform(1,10_000))
    s0 = rand(b0)
    a = rand(actions(pomdp))

    h = simulate(
        pomdp,
        b0,
        planner;
        T = 50,
        upd = BootstrapFilter,
        n_p = 1_000,
        progress = false
    )

    @test plot(h) isa Figure;
    @test plot(h; kind=:line) isa Figure;
    @test_throws DomainError plot(h; kind=:wrong);

    @test CovidPOMDPs.plot_inf_belief(h) isa Figure;
end
