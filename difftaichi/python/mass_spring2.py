"""
TODO:
* save final weights
* load saved weights, turn off weight updates
* transition forward sim into julia
* test that gradients are the same
"""

from mass_spring_robot_config import robots
import random
import sys
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os
from datetime import datetime
import pickle

random.seed(0)
np.random.seed(0)

RESULTS_DIR = "mass_spring_output"
real = ti.f32
ti.init(default_fp=real)

use_toi = False
max_steps = 4096
steps = 2048 // 3
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()

x = vec()
v = vec()
v_acc = vec()

head_id = 0

n_objects = None
ground_height = 0.1
gravity = -4.8
dt = 0.004

spring_omega = 10
damping = 15

n_springs = None
spring_anchor_a = ti.var(ti.i32)
spring_anchor_b = ti.var(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()

n_sin_waves = 10
weights1 = scalar()
bias1 = scalar()

n_hidden = 32
weights2 = scalar()
bias2 = scalar()
hidden = scalar()

center = vec()

act = scalar()

# This is a function because n_objects is filled out once the robot is set.
def n_input_states():
    return n_sin_waves + 4 * n_objects + 2


@ti.layout
def place():
    ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, v_acc)
    ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                         spring_length, spring_stiffness,
                                         spring_actuation)
    ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(weights1)
    ti.root.dense(ti.i, n_hidden).place(bias1)
    ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
    ti.root.dense(ti.i, n_springs).place(bias2)
    ti.root.dense(ti.ij, (max_steps, n_hidden)).place(hidden)
    ti.root.dense(ti.ij, (max_steps, n_springs)).place(act)
    ti.root.dense(ti.i, max_steps).place(center)
    ti.root.place(loss)
    ti.root.lazy_grad()


@ti.kernel
def compute_center(t: ti.i32):
    for _ in range(1):
        c = ti.Vector([0.0, 0.0])
        for i in ti.static(range(n_objects)):
            c += x[t, i]
        center[t] = (1.0 / n_objects) * c


@ti.kernel
def nn1(t: ti.i32):
    for i in range(n_hidden):
        actuation = 0.0
        for j in ti.static(range(n_sin_waves)):
            actuation += weights1[i, j] * ti.sin(spring_omega * t * dt +
                                                 2 * math.pi / n_sin_waves * j)
        for j in ti.static(range(n_objects)):
            offset = x[t, j] - center[t]
            # use a smaller weight since there are too many of them
            actuation += weights1[i, j * 4 + n_sin_waves] * offset[0] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 1] * offset[1] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 2] * v[t, j][0] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 3] * v[t, j][1] * 0.05
        actuation += weights1[i, n_objects * 4 + n_sin_waves] * center[t][0]
        actuation += weights1[i, n_objects * 4 + n_sin_waves + 1] * center[t][1]
        actuation += bias1[i]
        actuation = ti.tanh(actuation)
        hidden[t, i] = actuation


@ti.kernel
def nn2(t: ti.i32):
    for i in range(n_springs):
        actuation = 0.0
        for j in ti.static(range(n_hidden)):
            actuation += weights2[i, j] * hidden[t, j]
        actuation += bias2[i]
        actuation = ti.tanh(actuation)
        act[t, i] = actuation


@ti.kernel
def forces(t: ti.i32):
    # Spring forces
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[t, a]
        pos_b = x[t, b]
        dist = pos_a - pos_b
        length = dist.norm() + 1e-4

        target_length = spring_length[i] * (1.0 + spring_actuation[i] * act[t, i])
        impulse = (length - target_length) * spring_stiffness[i] / length * dist

        ti.atomic_add(v_acc[t + 1, a], -impulse)
        ti.atomic_add(v_acc[t + 1, b], impulse)

    # Gravity
    for i in range(n_objects):
        ti.atomic_add(v_acc[t + 1, i][1], gravity)


@ti.kernel
def advance_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        old_v = s * v[t - 1, i] + dt * v_acc[t, i]
        old_x = x[t - 1, i]
        new_x = old_x + dt * old_v
        toi = 0.0
        new_v = old_v
        if new_x[1] < ground_height and old_v[1] < -1e-4:
            toi = -(old_x[1] - ground_height) / old_v[1]
            new_v = ti.Vector([0.0, 0.0])
        new_x = old_x + toi * old_v + (dt - toi) * new_v

        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def advance_no_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        old_v = s * v[t - 1, i] + dt * v_acc[t, i]
        old_x = x[t - 1, i]
        new_v = old_v
        depth = old_x[1] - ground_height
        if depth < 0 and new_v[1] < 0:
            # friction projection
            new_v[0] = 0
            new_v[1] = 0
        new_x = old_x + dt * new_v
        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = -x[t, head_id][0]


# gui = ti.core.GUI("Mass Spring Robot", ti.veci(1024, 1024))
# canvas = gui.get_canvas()

def animate(total_steps: int, output=None):
    """Animate a the policy controlling the robot.

    * `total_steps` controls the number of time steps to animate for. This is
        necessary since the final animation is run for more time steps than in
        training.
    * `output` controls whether or not frames of the animation are screenshotted
        and dumped to a results directory.
    """
    if output:
        os.makedirs("{}/{}/".format(RESULTS_DIR, output))

    for t in range(1, total_steps):
        canvas.clear(0xFFFFFF)
        canvas.path(ti.vec(0, ground_height),
                    ti.vec(1, ground_height)).color(0x0).radius(3).finish()

        def circle(x, y, color):
            canvas.circle(ti.vec(x, y)).color(
                ti.rgb_to_hex(color)).radius(7).finish()

        for i in range(n_springs):
            def get_pt(x):
                return ti.vec(x[0], x[1])

            a = act[t - 1, i] * 0.5
            r = 2
            if spring_actuation[i] == 0:
                a = 0
                c = 0x222222
            else:
                r = 4
                c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
            canvas.path(
                get_pt(x[t, spring_anchor_a[i]]),
                get_pt(x[t, spring_anchor_b[i]])).color(c).radius(r).finish()

        for i in range(n_objects):
            color = (0.4, 0.6, 0.6)
            if i == head_id:
                color = (0.8, 0.2, 0.3)
            circle(x[t, i][0], x[t, i][1], color)

        gui.update()
        if output:
            gui.screenshot("{}/{}/{:04d}.png".format(RESULTS_DIR, output, t))

def forward(total_steps: int):
    for t in range(1, total_steps):
        compute_center(t - 1)
        nn1(t - 1)
        nn2(t - 1)
        forces(t - 1)
        if use_toi:
            advance_toi(t)
        else:
            advance_no_toi(t)

    loss[None] = 0
    compute_loss(steps - 1)


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_acc[t, i] = ti.Vector([0.0, 0.0])


def setup_robot(objects, springs):
    global n_objects, n_springs
    n_objects = len(objects)
    n_springs = len(springs)

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    for i in range(n_objects):
        x[0, i] = objects[i]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_length[i] = s[2]
        spring_stiffness[i] = s[3]
        spring_actuation[i] = s[4]


def init_weights():
    for i in range(n_hidden):
        for j in range(n_input_states()):
            weights1[i, j] = np.random.randn() * math.sqrt(2 / (n_hidden + n_input_states())) * 2
        bias1[i] = 0.0

    for i in range(n_springs):
        for j in range(n_hidden):
            # TODO: n_springs should be n_actuators
            weights2[i, j] = np.random.randn() * math.sqrt(2 / (n_hidden + n_springs)) * 3
        bias2[i] = 0.0

def optimize(num_iters: int, toi, visualize):
    global use_toi
    use_toi = toi

    losses = []
    # forward('initial{}'.format(robot_id), visualize=visualize)
    for iter in range(num_iters):
        clear_states()
        # with ti.Tape(loss) automatically clears all gradients
        with ti.Tape(loss):
            forward(steps)
        if visualize:
            animate(steps)

        print('Iter=', iter, 'Loss=', loss[None])

        total_norm_sqr = 0
        for i in range(n_hidden):
            for j in range(n_input_states()):
                total_norm_sqr += weights1.grad[i, j]**2
            total_norm_sqr += bias1.grad[i]**2

        for i in range(n_springs):
            for j in range(n_hidden):
                total_norm_sqr += weights2.grad[i, j]**2
            total_norm_sqr += bias2.grad[i]**2

        # print(total_norm_sqr)

        # learning_rate = 25
        # gradient_clip = 1
        # scale = learning_rate * min(1.0, gradient_clip / total_norm_sqr ** 0.5)
        gradient_clip = 0.2
        scale = gradient_clip / (total_norm_sqr**0.5 + 1e-6)

        for i in range(n_hidden):
            for j in range(n_input_states()):
                weights1[i, j] -= scale * weights1.grad[i, j]
            bias1[i] -= scale * bias1.grad[i]

        for i in range(n_springs):
            for j in range(n_hidden):
                weights2[i, j] -= scale * weights2.grad[i, j]
            bias2[i] -= scale * bias2.grad[i]
        losses.append(loss[None])

    return losses

def main(robot_id, toi=True, visualize=False):
    # Set up the robot configuration.
    setup_robot(*robots[robot_id]())

    # Load existing weights.
    # with open("weights.pkl", "rb") as fh:
    #     w1, b1, w2, b2 = pickle.load(fh)
    #     weights1.from_numpy(w1)
    #     bias1.from_numpy(b1)
    #     weights2.from_numpy(w2)
    #     bias2.from_numpy(b2)

    # Train the policy.
    stuff = []
    for _ in range(10):
        init_weights()
        optimize(1, toi=toi, visualize=visualize)
        stuff.append((weights1.to_numpy(), weights2.to_numpy(), x.to_numpy(), v.to_numpy(), weights1.grad.to_numpy(), bias1.grad.to_numpy(), weights2.grad.to_numpy(), bias2.grad.to_numpy()))

    with open("stuffdump.pkl", "wb") as fh:
        pickle.dump(stuff, fh)

    # Run the final policy.
    # clear_states()
    # animate will show the exact same thing, so no need to visualize.
    # forward(2 * steps)
    # animate(2 * steps)
    # animate(2 * steps, output=f"robot{robot_id}_{datetime.now()}")

    # Save weights.
    # with open("weights.pkl", "wb") as fh:
    #     pickle.dump((weights1.to_numpy(), bias1.to_numpy(), weights2.to_numpy(), bias2.to_numpy()), fh)
