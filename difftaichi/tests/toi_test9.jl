"""Backprop through semi-implicit Euler."""

import UnicodePlots: lineplot
import Zygote

v_dynamics(v, x, u) = [0.0]
x_dynamics(v, x, u) = v
toi_affect(v, x, dt) = (-v, -dt * v - x)
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
            new_v = old_v + dt * v_dynamics(old_v, old_x, nothing)
            # Note the use of new_v here instead of old_v.
            new_x = old_x + dt * x_dynamics(new_v, old_x, nothing)
            if new_x[1] < 0 && new_v[1] < 0
                # The difftaichi mass_spring.py actually uses new_v here.
                (v, x) = toi_affect(old_v, old_x, dt)
            else
                v = new_v
                x = new_x
            end
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
