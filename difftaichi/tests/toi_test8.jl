"""Test backprop-through-the-solver with a specific TOI affect the whole time."""

import UnicodePlots: lineplot
import Zygote

terminal_cost(x) = (x[1] - 0.6) ^ 2

T = 1.0
dt = 0.01

v0 = [-1.0]
x0 = [0.75]

for _ in 1:100
    final_state, pb1 = Zygote.pullback((v0, x0) -> begin
        v = v0
        x = x0
        for _ = 1:(T/dt)
            old_v = v
            old_x = x
            toi = -old_x[1] / min(old_v[1], -eps())
            impact_velocity = -old_v
            new_v = if toi < dt impact_velocity else old_v end
            x = old_x + min(toi, dt) * old_v + max(dt - toi, 0) * new_v
            v = new_v
        end
        (v, x)
    end, v0, x0)
    xT = final_state[2]
    local _, pb2 = Zygote.pullback(terminal_cost, xT)

    (g_xT, ) = pb2(1.0)
    local g = (zero(v0), g_xT)
    g = pb1(g)
    g = g[2]

    global x0 -= 0.1 * g
end
@assert x0 ≈ [0.4]
