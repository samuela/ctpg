"""A rewrite of the DiffTaichi electric example.

This script will run both PPG and BPTT, create a new folder like "2020-10-12T17:06:12.846-electric-pure-julia" and dump
results and videos into there.

Select whether to use pure julia definition of the dynamics or call out to the DiffTaichi version by changing
`forces_fn`. Only affects wallclock performance since we verify that they are numerically equivalent and their gradients
agree with finite differences.
"""

# We test that the julia and DiffTaichi implementations are equivalent for Float32 inputs, but we don't get floating
# point equivalent results when training because the Julia version ends up not casting to Float32 in `forces_fn`.
# Forcing the julia `forces_fn` to do everything in Float32 instead of Float64 is actually kind of obnoxious and of
# little value so we don't bother. Float64's are higher accuracy anyhow.

# Note: we use ppg_toi.jl instead of ppg.jl simply because ppg_toi.jl supports DynamicalODEProblems. There are no
# contacts in this problem.
include("../ppg_toi.jl")

import DiffEqFlux: FastChain, FastDense, initial_params
import Random: seed!
import Flux
import Flux: Momentum
import ProgressMeter
import DifferentialEquations: Tsit5, BS3
import LinearAlgebra: norm
import DiffEqSensitivity: ZygoteVJP
import PyCall: @pyimport, @py_str, pyimport
import Test: @test
import FiniteDifferences

# See https://github.com/JuliaPy/PyCall.jl/issues/48#issuecomment-515787405.
py"""
import sys
sys.path.insert(0, "./difftaichi/electric")
"""

np = pyimport("numpy")
electric = pyimport("electric")
importlib = pyimport("importlib")
importlib.reload(electric)

# DiffTaichi does everything in float32, so in order to test equivalence with them we need to be able to do the same.
difftaichi_dt = 0.03

# DiffTaichi takes 512 steps each of length dt = 0.03.
T = 15.36

# DiffTaichi uses 512 // 256 = 2.
num_segments = 2

# The exact equivalent from the difftaichi code is that our dampening = (1 - exp(-dt * d)) / d where d is their damping.
# In difftaichi d = 0.2 and dt = 0.03 which comes out to...
damping = 0.029910179730323616
@assert damping >= 0

# We need to copy `gravitation_position` to Julia land in order to please Zygote.
const gravitation_position = deepcopy(electric.gravitation_position)
const K = deepcopy(electric.K)
const n_gravitation = size(gravitation_position, 1)

# Here's the pure Julia version:
julia_forces_fn(x, u) = begin
    sum([begin
        r = x - gravitation_position[i, :]
        len_r = max(norm(r), 1e-1)
        K * u[i] / (len_r ^ 3) * r
    end for i in 1:n_gravitation])
end

# Here's the DiffTaichi version:
function difftaichi_forces_fn(x, u)
    electric.forces_fn(np.array(x, dtype=np.float32), np.array(u, dtype=np.float32))
end
Zygote.@adjoint function difftaichi_forces_fn(x, u)
    # We reuse these for both the forward and backward.
    x_np = np.array(x, dtype=np.float32)
    u_np = np.array(u, dtype=np.float32)
    function pullback(ΔΩ)
        electric.forces_fn_vjp(x_np, u_np, np.array(ΔΩ, dtype=np.float32))
    end
    electric.forces_fn(x_np, u_np), pullback
end

begin
    @info "Testing forces_fns are equivalent"
    seed!(123)
    for i in 1:100
        local x = randn(Float32, (2, ))
        local u = randn(Float32, (n_gravitation, ))
        @test difftaichi_forces_fn(x, u) ≈ julia_forces_fn(x, u)
    end
end

begin
    @info "Testing forces_fn gradients"
    seed!(123)
    for _ in 1:10
        # We don't shrink x too small because we have the length of the spring in the denominator somewhere in the
        # forces calculation, aka don't squish the springs too much to avoid crazy forces.
        local x = randn(Float32, (2, ))
        local u = 0.01 * randn(Float32, (n_gravitation, ))
        local g = 0.01 * randn(Float32, (2, ))
        local g_x_fd, g_u_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), (x, u) -> sum(julia_forces_fn(x, u) .* g), x, u)
        local g_x_dt, g_u_dt = Zygote.gradient((x, u) -> sum(difftaichi_forces_fn(x, u) .* g), x, u)
        local g_x_j, g_u_j = Zygote.gradient((x, u) -> sum(julia_forces_fn(x, u) .* g), x, u)

        # We assume at this point that the julia and difftaichi versions have been produce the same output, and assuming
        # we trust Zygote's AD, this is a safe way to test that our difftaichi gradients are good.
        @test g_x_dt ≈ g_x_j
        @test g_u_dt ≈ g_u_j
        @test g_x_fd ≈ g_x_dt
    end
end

# Pick whether to use `julia_forces_fn` or `difftaichi_forces_fn`. Note that the julia version is more than 10x faster
# than the DiffTaichi version, and we test that they're equivalent so we generally take a shortcut and run the julia
# version.
forces_fn = julia_forces_fn
# forces_fn = difftaichi_forces_fn

function sample_task()
    # Not really sure why DiffTaichi samples goal points this way, but this is what it is.
    goal_points = 0.2 .+ rand(num_segments + 1, 2) * 0.6
    # We nudge the final time step to be just a liiiiiitle bit bigger than T so that we don't overflow goal_points when
    # t = T.
    timesteps = range(0, nextfloat(T), length = num_segments + 1)

    goal_stuff(t) = begin
        ix = searchsortedlast(timesteps, t)
        a = goal_points[ix, :]
        b = goal_points[ix + 1, :]
        tt = (t - timesteps[ix]) / (timesteps[ix + 1] - timesteps[ix])
        goal = (1 - tt) * a + tt * b
        # Note that this isn't technically right since it doesn't account for time/distance. But this is how difftaichi
        # does it: https://github.com/yuanming-hu/difftaichi/blob/0ac795a5a4dafab50c52592448d224e71ee0328d/examples/electric.py#L170-L173.
        goal_v = b - a
        (goal, goal_v)
    end

    v_dynamics(v, x, u) = forces_fn(x, u) - damping * v
    x_dynamics(v, x, u) = v
    cost(v, x, u, t) = begin
        goal, _ = goal_stuff(t)
        sum((x - goal) .^ 2)
    end
    observation(v, x, t) = begin
        goal, goal_v = goal_stuff(t)
        # For some reason, difftaichi takes off 0.5 like so:
        # [x .- 0.5; v; goal .- 0.5; goal_v .- 0.5]
        [x; v; goal; goal_v]
    end

    v0 = zeros(2)
    x0 = goal_points[1, :]
    v0, x0, v_dynamics, x_dynamics, cost, observation, goal_stuff
end

# DiffTaichi uses a single hidden layer with 64 units.
num_hidden = 64
policy = FastChain(
    FastDense(8, num_hidden, tanh),
    FastDense(num_hidden, n_gravitation, tanh),
)
init_policy_params = 0.1 * initial_params(policy)

# DiffTaichi does 200,000 iterations
num_iters = 100000

# Optimizers are stateful, so we shouldn't just reuse them. DiffTaichi does SGD with 2e-2 learning rate.
# make_optimizer = () -> Momentum(2e-2, 0.0)
make_optimizer = () -> Momentum(1e-3)

function run_ppg(rng_seed, outputdir)
    # Seed here so that both interp and euler get the same batches.
    seed!(rng_seed)

    loss_per_iter = fill(NaN, num_iters)
    elapsed_per_iter = fill(NaN, num_iters)
    nf_per_iter = fill(NaN, num_iters)
    n∇ₓf_per_iter = fill(NaN, num_iters)
    n∇ᵤf_per_iter = fill(NaN, num_iters)
    # julia is column-major so this way is more cache-efficient.
    # policy_params_per_iter = fill(NaN, length(init_policy_params), num_iters)
    # g_per_iter = fill(NaN, length(init_policy_params), num_iters)

    policy_params = deepcopy(init_policy_params)
    opt = make_optimizer()
    progress = ProgressMeter.Progress(num_iters)
    for iter in 1:num_iters
        t0 = Base.time_ns()
        # We have to sample a new task each time because we randomize where the goal goes.
        v0, x0, v_dyn, x_dyn, cost, observation, goal_stuff = sample_task()
        # Use the TOI version because it supports DynamicalODEProblems, just give it a no-op toi setup.
        sol, pullback = ppg_toi_goodies(
            v_dyn,
            x_dyn,
            cost,
            (v, x, params, t) -> policy(observation(v, x, t), params),
            TOIStuff([], (v, x, dt) -> begin @error "this should never happen" end, 1e-3),
            T
        )(v0, x0, policy_params, Tsit5(), Dict(:rtol => 1e-3, :atol => 1e-3))
        # )(v0, x0, policy_params, Tsit5(), Dict())
        loss = sol.solutions[end].u[end].x[2][1]

        pb_stuff = pullback(([0.0; zeros(length(v0))], [1.0; zeros(length(x0))]), InterpolatingAdjoint(autojacvec = ZygoteVJP()))
        g = pb_stuff.g_p
        # clamp!(g, -10, 10)
        Flux.Optimise.update!(opt, policy_params, g)
        elapsed = Base.time_ns() - t0

        loss_per_iter[iter] = loss
        elapsed_per_iter[iter] = elapsed
        nf_per_iter[iter] = pb_stuff.nf + sum(s.destats.nf for s in sol.solutions)
        n∇ₓf_per_iter[iter] = pb_stuff.n∇ₓf
        n∇ᵤf_per_iter[iter] = pb_stuff.n∇ᵤf
        # policy_params_per_iter[:, iter] = policy_params
        # g_per_iter[:, iter] = g

        ProgressMeter.next!(
            progress;
            showvalues = [
                (:iter, iter),
                (:loss, loss),
                (:elapsed_ms, elapsed / 1e6),
                (:nf, nf_per_iter[iter]),
                (:n∇ₓf, n∇ₓf_per_iter[iter]),
                (:n∇ᵤf, n∇ᵤf_per_iter[iter]),
            ],
        )

        if iter % 1000 == 0
            ts = 0:difftaichi_dt:T
            zs = sol.(ts)
            vs = [z.x[1][2:end] for z in zs]
            xs = [z.x[2][2:end] for z in zs]
            acts = [policy(observation(v, x, t), policy_params) for (v, x, t) in zip(vs, xs, ts)]
            goals = [goal for (goal, _) in goal_stuff.(ts)]

            Base.Filesystem.mktempdir() do dir
                electric.animate(xs, goals, acts, outputdir = dir)
                # -y option overwrites existing video. See https://stackoverflow.com/questions/39788972/ffmpeg-override-output-file-if-exists
                Base.run(`ffmpeg -y -framerate 100 -i $dir/%04d.png $dir/video.mp4`)
                # For some reason ffmpeg isn't happy just outputting to outputdir.
                mv("$dir/video.mp4", "$outputdir/electric_iter$iter.mp4")
            end
        end
    end

    (
        loss_per_iter = loss_per_iter,
        elapsed_per_iter = elapsed_per_iter,
        nf_per_iter = nf_per_iter,
        n∇ₓf_per_iter = n∇ₓf_per_iter,
        n∇ᵤf_per_iter = n∇ᵤf_per_iter,
        # policy_params_per_iter = policy_params_per_iter,
        # g_per_iter = g_per_iter,
    )
end

function run_bptt(rng_seed, num_timesteps, dt)
    seed!(rng_seed)

    loss_per_iter = fill(NaN, num_iters)
    elapsed_per_iter = fill(NaN, num_iters)
    nf_per_iter = fill(NaN, num_iters)
    n∇ₓf_per_iter = fill(NaN, num_iters)
    n∇ᵤf_per_iter = fill(NaN, num_iters)
    # julia is column-major so this way is more cache-efficient.
    # policy_params_per_iter = fill(NaN, length(init_policy_params), num_iters)
    # g_per_iter = fill(NaN, length(init_policy_params), num_iters)

    policy_params = deepcopy(init_policy_params)
    opt = make_optimizer()
    progress = ProgressMeter.Progress(num_iters)
    for iter in 1:num_iters
        t0 = Base.time_ns()
        # We have to sample a new task each time because we randomize where the goal goes.
        v0, x0, v_dyn, x_dyn, cost, observation, goal_stuff = sample_task()

        loss, pullback = Zygote.pullback(policy_params) do params
            v = v0
            x = x0
            total_cost = 0.0
            # Note: do we have an off-by-one issue here relative to difftaichi?
            for iter in 1:num_timesteps
                # Note: difftaichi may actually use t-1 for u and t for the cost. Double check this.
                t = iter * dt
                u = policy(observation(v, x, t), params)
                v += dt * v_dyn(v, x, u)
                x += dt * x_dyn(v, x, u)
                total_cost += dt * cost(v, x, u, t)
            end
            total_cost
        end
        (g, ) = pullback(1.0)
        # clamp!(g, -10, 10)
        Flux.Optimise.update!(opt, policy_params, g)
        elapsed = Base.time_ns() - t0

        loss_per_iter[iter] = loss
        elapsed_per_iter[iter] = elapsed
        nf_per_iter[iter] = num_timesteps
        n∇ₓf_per_iter[iter] = num_timesteps
        n∇ᵤf_per_iter[iter] = num_timesteps
        # policy_params_per_iter[:, iter] = policy_params
        # g_per_iter[:, iter] = g

        ProgressMeter.next!(
            progress;
            showvalues = [
                (:iter, iter),
                (:loss, loss),
                (:elapsed_ms, elapsed / 1e6),
                (:nf, nf_per_iter[iter]),
                (:n∇ₓf, n∇ₓf_per_iter[iter]),
                (:n∇ᵤf, n∇ᵤf_per_iter[iter]),
            ],
        )

        # Note: no videos for BPTT because I'm lazy.
    end

    (
        loss_per_iter = loss_per_iter,
        elapsed_per_iter = elapsed_per_iter,
        nf_per_iter = nf_per_iter,
        n∇ₓf_per_iter = n∇ₓf_per_iter,
        n∇ᵤf_per_iter = n∇ᵤf_per_iter,
        # policy_params_per_iter = policy_params_per_iter,
        # g_per_iter = g_per_iter,
    )
end

import Dates
import JLSO
import ArgParse

# It's far more convenient (for `run_many.jl`) to pass CLI args than environment variables for better or worse.
args = begin
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! s begin
        "--experiment_dir"
            help = "where to dump results"
            arg_type = String
        "--rng_seed"
            help = "random seed"
            arg_type = Int
            default = 123
    end
    ArgParse.parse_args(s)
end

@info "pure julia version"
rng_seed = args["rng_seed"]
experiment_dir = isnothing(args["experiment_dir"]) ? "results/$(Dates.now())-electric-seed$(rng_seed)" : args["experiment_dir"]
mkdir(experiment_dir)

@info "PPG"
ppg_results = run_ppg(rng_seed, experiment_dir)

@info "BPTT"
# DiffTaichi does 512 steps.
bptt_results = run_bptt(rng_seed, 512, difftaichi_dt)

JLSO.save("$experiment_dir/results.jlso", :ppg_results => ppg_results, :bptt_results => bptt_results)
