"""Testing a direct translation of https://github.com/yuanming-hu/difftaichi/blob/master/examples/misc/rigid_body_toi_visualize.py."""

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
            new_v = v
            # This is technically semi-implicit Euler if you think about it.
            new_x = old_x + dt * new_v
            toi = 0.0
            if new_x[1] < 0 && new_v[1] < 0
                new_v = -new_v
                toi = -old_x[1] / old_v[1]
            end
            v = new_v
            x = old_x + toi * old_v + (dt - toi) * new_v
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
