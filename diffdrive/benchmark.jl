"""A self-contained script to test the performance of the forward and adjoint
DifferentialEquations.jl solves with the quadrotor."""

include("common.jl")
include("../ppg.jl")

import DifferentialEquations: Tsit5, VCABM
import DiffEqFlux: FastChain, FastDense, initial_params, ODEProblem, solve
import Random: seed!
import DiffEqSensitivity:
    InterpolatingAdjoint, BacksolveAdjoint, ODEAdjointProblem, ReverseDiffVJP
using BenchmarkTools

seed!(123)

T = 15.0

dynamics, cost, sample_x0, obs = DiffDriveEnv.env(Float32, 1.0f0, 0.5f0)
x0 = sample_x0()

num_hidden = 64
policy = FastChain(
    (x, _) -> obs(x),
    FastDense(11, num_hidden, tanh),
    FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, 2),
)

policy_params = initial_params(policy)
ppg_stuff = ppg_goodies(dynamics, cost, policy, T)

@info "forward"
fwd_sol, vjp = @btime ppg_stuff.loss_pullback(x0, policy_params, Tsit5())
@show fwd_sol.destats.nf
g = vcat(1, zero(x0))

@info "QuadratureAdjoint"
bwd = @btime vjp(
    g,
    QuadratureAdjoint(autojacvec = ReverseDiffVJP(true), abstol = 1e-3, reltol = 1e-3),
)
@show bwd.nf, bwd.n∇ₓf, bwd.n∇ᵤf
@show bwd.quadgk_err / length(bwd.g)

@info "InterpolatingAdjoint"
bwd = @btime vjp(g, InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
@show bwd.nf, bwd.n∇ₓf, bwd.n∇ᵤf

@info "BacksolveAdjoint"
bwd = @btime vjp(g, BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)))
@show bwd.nf, bwd.n∇ₓf, bwd.n∇ᵤf

nothing
