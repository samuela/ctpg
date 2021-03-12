"""LQR example to test whether the gradients on parameters work."""

include("../ppg_toi.jl")

import DifferentialEquations: Tsit5
import UnicodePlots: lineplot
import Random: seed!

seed!(123)

d = 3
v_dynamics(v, x, u) = u
x_dynamics(v, x, u) = v
cost(v, x, u) = sum(x .^ 2)
policy(v, x, p_flat, t) = begin
    p = reshape(p_flat, (d, 2 * d))
    p * [v; x]
end

T = 1.0
loss_pullback = ppg_toi_goodies(
    v_dynamics,
    x_dynamics,
    cost,
    policy,
    TOIStuff([], (v, x, dt) -> begin @error "this should never happen" end, 1e-3),
    T
)

v0 = [-1.0, -2.0, 0.3]
x0 = [0.75, 0.25, -0.1]
p = randn(2 * d * d)
sol, _ = loss_pullback(v0, x0, p, nothing, Dict())

# TODO plot all coords, plot during train loop
ts = 0:0.01:T
lineplot(ts, [z.x[2][2] for z in sol.(ts)], title = "x") |> show
lineplot(ts, [z.x[1][2] for z in sol.(ts)], title = "v") |> show

@assert length(sol.solutions) == 1

for _ in 1:1000
    local sol, pb1 = loss_pullback(v0, x0, p, Tsit5(), Dict())
    loss = sol.solutions[end].u[end].x[2][1]

    local g = ([0.0; zero(v0)], [1.0; zero(x0)])
    g = pb1(g, InterpolatingAdjoint()).g_p
    global p -= g

    # @show loss
    # @show sum(g .^ 2)
end

sol, pb1 = loss_pullback(v0, x0, p, Tsit5(), Dict())
g = ([0.0; zero(v0)], [1.0; zero(x0)])
g = pb1(g, InterpolatingAdjoint()).g_p
@assert sum(g .^ 2) < 1e-3
