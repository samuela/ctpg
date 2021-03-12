"""A bouncy ball with gravity. Goal is to reach a specific height at t = T."""

include("../ppg_toi.jl")

import DifferentialEquations: Tsit5
import UnicodePlots: lineplot

gravity = -9.8
v_dynamics(v, x, u) = [gravity]
x_dynamics(v, x, u) = v
cost(v, x, u) = 0.0
policy(v, x, t, p) = 0.0
toi_affect(v, x, dt) = (-v, -dt * v - x)
terminal_cost(x) = (x[1] - 5) ^ 2

T = 2.0
loss_pullback = ppg_toi_goodies(
    v_dynamics,
    x_dynamics,
    cost,
    policy,
    TOIStuff([(v, x) -> x[1]], toi_affect, 1e-6),
    T
)

v0 = [0.0]
x0 = [10.0]

cost_per_iter = []
for iter in 1:1000
    sol, pb1 = loss_pullback(v0, x0, zeros(), Tsit5(), Dict())
    xT = sol.solutions[end].u[end].x[2][2:end]
    local cost, pb2 = Zygote.pullback(terminal_cost, xT)

    (g_xT, ) = pb2(1.0)
    local g = ([0; zero(v0)], [0; g_xT])
    g = pb1(g, InterpolatingAdjoint()).g_z0
    g = g[2][2:end]

    global x0 -= 0.01 * g

    @show iter
    @show cost
    @show xT
    ts = 0:0.01:T
    # Don't forget the first is for the cost!
    lineplot(ts, [z[2] for z in sol.(ts)]) |> show
    println()

    push!(cost_per_iter, cost)
end

@assert cost_per_iter[end] <= 1e-6

lineplot((1:length(cost_per_iter)), convert(Array{Float64}, cost_per_iter)) |> show
