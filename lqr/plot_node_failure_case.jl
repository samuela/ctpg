import DifferentialEquations: Tsit5
import Flux
import Flux: ADAM
import DiffEqFlux: FastChain, FastDense, initial_params, ODEProblem, solve
import Random: seed!
import Statistics: mean
import Zygote
import DiffEqSensitivity: InterpolatingAdjoint, BacksolveAdjoint, ODEAdjointProblem
import LinearAlgebra: I, norm
import ControlSystems
import PyPlot
import PyCall: PyDict

include("common.jl")
include("../ppg.jl")

seed!(123)

rcParams = PyDict(PyPlot.matplotlib["rcParams"])
rcParams["font.size"] = 20
rcParams["figure.figsize"] = (8, 4)

floatT = Float32
x_dim = 2
T = 25.0
num_iters = 1000
x0 = ones(x_dim)

num_hidden = 32
policy = FastChain(
    FastDense(x_dim, num_hidden, tanh),
    # FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, x_dim),
)
init_policy_params = initial_params(policy) * 0.1

const A = Matrix{floatT}(0 * I, x_dim, x_dim)
const B = Matrix{floatT}(I, x_dim, x_dim)
const Q = Matrix{floatT}(I, x_dim, x_dim)
const R = Matrix{floatT}(I, x_dim, x_dim)
const K = ControlSystems.lqr(A, B, Q, R)
dynamics, cost, sample_x0 = LinearEnv.linear_env(floatT, x_dim, 0 * I, I, I, I)

lqr_goodies = ppg_goodies(dynamics, cost, (x, _, _) -> -K * x, T)
lqr_sol, _ = lqr_goodies.loss_pullback(x0, nothing, Tsit5(), Dict())
lqr_loss = lqr_sol[end][1]

learned_policy_goodies = ppg_goodies(dynamics, cost, (x, t, θ) -> policy(x, θ), T)

function run_neural_ode()
    policy_params = deepcopy(init_policy_params)
    learned_loss_per_iter = fill(NaN, num_iters)
    reconst_error_per_iter = fill(NaN, num_iters)
    opt = ADAM()
    for iter = 1:num_iters
        @time begin
            fwd_sol, vjp = learned_policy_goodies.loss_pullback(x0, policy_params, Tsit5(), Dict())
            bwd = vjp(vcat(1, zero(x0)), BacksolveAdjoint(checkpointing = false))
            loss, _ = extract_loss_and_xT(fwd_sol)
            Flux.Optimise.update!(opt, policy_params, bwd.g)
            learned_loss_per_iter[iter] = loss
            reconst_error_per_iter[iter] = norm(bwd.x0_reconstructed - vcat(0.0, x0))
            # println("Episode $iter, excess loss = $(loss - lqr_loss)")
        end
    end

    learned_loss_per_iter, reconst_error_per_iter
end

################################################################################
function run_ctpg()
    policy_params = deepcopy(init_policy_params)
    learned_loss_per_iter = fill(NaN, num_iters)
    opt = ADAM()
    for iter = 1:num_iters
        @time begin
            fwd_sol, vjp = learned_policy_goodies.loss_pullback(x0, policy_params, Tsit5(), Dict())
            bwd = vjp(vcat(1, zero(x0)), InterpolatingAdjoint())
            # bwd = vjp(vcat(1, zero(x0)), QuadratureAdjoint())
            loss, _ = extract_loss_and_xT(fwd_sol)
            Flux.Optimise.update!(opt, policy_params, bwd.g)
            learned_loss_per_iter[iter] = loss
            # println("Episode $iter, excess loss = $(loss_ - lqr_loss)")
        end
    end

    learned_loss_per_iter
end

@info "Neural ODE"
node_learned_loss_per_iter, node_reconst_error_per_iter = run_neural_ode()

@info "CTPG"
ctpg_learned_loss_per_iter = run_ctpg()

begin
    _, ax1 = PyPlot.subplots()
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss \$\\mathcal{L}(x_0, \\theta)\$", color = "tab:blue")
    ax1.tick_params(axis = "y", labelcolor = "tab:blue")
    # ax1.set_yscale("log")
    ax1.plot(node_learned_loss_per_iter, color = "tab:blue")
    PyPlot.axhline(lqr_loss, linestyle = "--", color = "grey")
    ax1.set_ylim(-5, 55)

    ax2 = ax1.twinx()
    ax2.set_ylabel("L2 error", color = "tab:red")
    ax2.tick_params(axis = "y", labelcolor = "tab:red")
    ax2.plot([], color = "tab:blue", label = "Neural ODE solution")
    ax2.plot([], linestyle = "--", color = "grey", label = "LQR solution")
    ax2.plot(node_reconst_error_per_iter, color = "tab:red", label = "Backsolve error")
    ax2.set_ylim(-5e5, 8e6)

    PyPlot.legend(loc = "upper left")
    PyPlot.tight_layout()
    PyPlot.savefig("node_comparison.pdf")
end

begin
    _, ax1 = PyPlot.subplots()
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss \$\\mathcal{L}(x_0, \\theta)\$", color = "tab:blue")
    ax1.tick_params(axis = "y", labelcolor = "tab:blue")
    # ax1.set_yscale("log")
    ax1.plot(ctpg_learned_loss_per_iter, color = "tab:blue")
    PyPlot.axhline(lqr_loss, linestyle = "--", color = "grey")
    ax1.set_ylim(-5, 55)

    ax2 = ax1.twinx()
    ax2.set_ylabel("L2 error", color = "tab:red")
    ax2.tick_params(axis = "y", labelcolor = "tab:red")
    ax2.plot([], color = "tab:blue", label = "CTPG (ours)")
    ax2.plot([], linestyle = "--", color = "grey", label = "LQR solution")

    # Interpolated forward pass has no reconstruction error since we have a
    # knot point at x(0)!
    ax2.plot(
        zeros(size(ctpg_learned_loss_per_iter)),
        color = "tab:red",
        label = "Backsolve error",
    )
    ax2.set_ylim(-5e5, 8e6)

    PyPlot.legend(loc = "upper right")
    PyPlot.tight_layout()
    PyPlot.savefig("ours_comparison.pdf")
end
