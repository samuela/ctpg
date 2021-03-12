"""Test that we only register down-crossings. This one is a little tricky
because although the true solution goes above zero, the integrators are smart
enough to actually skip any concrete steps above zero. So we need to do this
testing the interpolated points thing."""

include("../ppg_toi.jl")

import DifferentialEquations
import UnicodePlots: lineplot

g = 9.8
v_dynamics(v, x, u) = [-g]
x_dynamics(v, x, u) = v
cost(v, x, u) = 0
policy(v, x, t, p) = 0
toi_affect(v, x, dt) = (-v, -dt * v - x)

T = 1.0
loss_pullback = ppg_toi_goodies(
    v_dynamics,
    x_dynamics,
    cost,
    policy,
    TOIStuff([(v, x) -> x[1]], toi_affect, 1e-6),
    T
)

x0 = -1
v0 = 5
parabola(t) = t * v0 - g / 2 * t^2 + x0
peak_time = v0 / g
@assert parabola(peak_time) > 0
land_time = (v0 + sqrt(v0^2 + 2 * g * x0)) / g
@assert land_time < T
sol, _ = loss_pullback([v0], [x0], zeros(), nothing, Dict())

ts = 0:0.01:T
# Don't forget the first is for the cost!
lineplot(ts, [z[2] for z in sol.(ts)]) |> show
lineplot(ts, [z[3] for z in sol.(ts)]) |> show

@assert length(sol.solutions) == 2
@show sol(peak_time)
@assert sol(peak_time) ≈ ArrayPartition([0, 0], [0, parabola(peak_time)])
