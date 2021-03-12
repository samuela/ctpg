import os

import taichi as ti

real = ti.f32
ti.init(default_fp=real)

# See https://github.com/taichi-dev/taichi/issues/1966.
x = ti.Vector.field(2, dtype=real, shape=(), needs_grad=True)
acc = ti.Vector.field(2, dtype=real, shape=(), needs_grad=True)
acc_bar = ti.Vector.field(2, dtype=real, shape=(), needs_grad=True)

n_gravitation = 8
# This is our actuation, or `u` in some other code.
gravitation = ti.field(dtype=real, shape=(n_gravitation, ), needs_grad=True)

loss = ti.field(dtype=real, shape=(), needs_grad=True)

pad = 0.1
gravitation_position = [[pad, pad], [pad, 1 - pad], [1 - pad, 1 - pad],
                        [1 - pad, pad], [0.5, 1 - pad], [0.5, pad], [pad, 0.5],
                        [1 - pad, 0.5]]
assert len(gravitation_position) == n_gravitation

K = 1e-3

@ti.kernel
def forces():
    for _ in range(1):  # parallelize this loop
        # Necessary to zero out acc before each call to avoid add to prev results.
        acc.fill(0.0)
        for i in ti.static(range(n_gravitation)):  # instead of this one
            r = x - ti.Vector(gravitation_position[i])
            len_r = ti.max(r.norm(), 1e-1)
            acc += K * gravitation[i] / (len_r * len_r * len_r) * r

@ti.kernel
def forces_vjp():
    loss[None] = acc[None].dot(acc_bar[None])

def forces_fn(x_np, u_np):
    x.from_numpy(x_np)
    gravitation.from_numpy(u_np)
    forces()
    return acc.to_numpy()

def forces_fn_vjp(x_np, u_np, acc_bar_np):
    x.from_numpy(x_np)
    gravitation.from_numpy(u_np)
    acc_bar.from_numpy(acc_bar_np)

    # loss[None] = 0
    with ti.Tape(loss):
        forces()
        forces_vjp()

    return x.grad.to_numpy(), gravitation.grad.to_numpy()

def animate(xs, goals, gravitations, outputdir):
    assert len(xs) == len(gravitations)
    gui = ti.GUI("Electric", (512, 512), background_color=0x3C733F, show_gui=False)

    for t in range(len(xs)):
        gui.clear()

        for i in range(len(gravitation_position)):
            r = (gravitations[t][i] + 1) * 30
            gui.circle(gravitation_position[i], 0xccaa44, r)

        gui.circle((xs[t][0], xs[t][1]), 0xF20530, 30)
        gui.circle((goals[t][0], goals[t][1]), 0x3344cc, 10)

        gui.show(os.path.join(outputdir, "{:04d}.png".format(t + 1)))
