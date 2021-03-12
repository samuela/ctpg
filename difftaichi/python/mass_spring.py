import os

import taichi as ti

RESULTS_DIR = "mass_spring_output"
real = ti.f32
# See https://github.com/taichi-dev/taichi/issues/1886.
ti.init(default_fp=real, cpu_max_num_threads=1)

# use_toi = False
# steps = 2048 // 3
n_objects = None
n_springs = None
gravity = -4.8
# dt = 0.004

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()

x = vec()
# v = vec()
v_acc = vec()
act = scalar()
# Incoming gradient on v_acc.
v_acc_bar = vec()

spring_anchor_a = ti.var(ti.i32)
spring_anchor_b = ti.var(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()

# n_sin_waves = 10

@ti.layout
def place():
    ti.root.dense(ti.i, n_objects).place(x, v_acc, v_acc_bar)
    ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                         spring_length, spring_stiffness,
                                         spring_actuation)
    ti.root.dense(ti.i, n_springs).place(act)
    ti.root.place(loss)
    ti.root.lazy_grad()

def setup_robot(objects, springs):
    global n_objects, n_springs
    n_objects = len(objects)
    n_springs = len(springs)

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    # This shouldn't really be necessary since this should be set in `forces_fn`
    # and `forces_fn_vjp`.
    for i in range(n_objects):
        x[i] = objects[i]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_length[i] = s[2]
        spring_stiffness[i] = s[3]
        spring_actuation[i] = s[4]

@ti.kernel
def forces():
    # It's necessary to zero out v_acc before each call. The difftaichi version doesn't need this because they index
    # each array by time as well.
    for i in range(n_objects):
        v_acc[i].fill(0.0)

    # Spring forces
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[a]
        pos_b = x[b]
        dist = pos_a - pos_b
        length = dist.norm() + 1e-4

        target_length = spring_length[i] * (1.0 + spring_actuation[i] * act[i])
        impulse = (length - target_length) * spring_stiffness[i] / length * dist

        ti.atomic_add(v_acc[a], -impulse)
        ti.atomic_add(v_acc[b], impulse)

    # Gravity
    for i in range(n_objects):
        ti.atomic_add(v_acc[i][1], gravity)

@ti.kernel
def forces_vjp():
    for i in range(n_objects):
        ti.atomic_add(loss, v_acc[i][0] * v_acc_bar[i][0])
        ti.atomic_add(loss, v_acc[i][1] * v_acc_bar[i][1])

# Can't write this function until we get rid of the x for every step thing.
def forces_fn(x_np, act_np):
    x.from_numpy(x_np)
    act.from_numpy(act_np)
    forces()
    return v_acc.to_numpy()

def forces_fn_vjp(x_np, act_np, v_acc_bar_np):
    x.from_numpy(x_np)
    act.from_numpy(act_np)
    v_acc_bar.from_numpy(v_acc_bar_np)

    # The trick here is that the JVP, v^T (\nabla_x f(x)), is equivalent to a
    # scalar backprop: d/dx [v^T f(x)].
    loss[None] = 0
    with ti.Tape(loss):
        forces()
        forces_vjp()

    return x.grad.to_numpy(), act.grad.to_numpy()

def animate(xs, acts, ground_height: float, outputdir=None, head_id=0):
    """Animate a the policy controlling the robot.

    * `total_steps` controls the number of time steps to animate for. This is
        necessary since the final animation is run for more time steps than in
        training.
    * `outputdir` controls whether or not frames of the animation are screenshotted
        and dumped to a results directory.
    """
    assert len(xs) == len(acts)
    gui = ti.GUI("Mass Spring Robot", (512, 512), background_color=0xFFFFFF, show_gui=False)

    for t in range(len(xs)):
        gui.line(
            begin=(0, ground_height),
            end=(1, ground_height),
            color=0x0,
            radius=3)

        def circle(x, y, color):
            gui.circle((x, y), ti.rgb_to_hex(color), 7)

        for i in range(n_springs):
            a = acts[t][i] * 0.5
            r = 2
            if spring_actuation[i] == 0:
                a = 0
                c = 0x222222
            else:
                r = 4
                c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
            gui.line(begin=tuple(xs[t][spring_anchor_a[i], :]),
                     end=tuple(xs[t][spring_anchor_b[i], :]),
                     radius=r,
                     color=c)

        for i in range(n_objects):
            color = (0.4, 0.6, 0.6)
            if i == head_id:
                color = (0.8, 0.2, 0.3)
            circle(xs[t][i][0], xs[t][i][1], color)

        if outputdir is not None:
            gui.show(os.path.join(outputdir, "{:04d}.png".format(t + 1)))
        else:
            gui.show()
