"""Quick script to test that the math is correct."""

import DifferentialEquations: ODEProblem, Tsit5, solve
import QuadGK: quadgk

policy(x, θ) = x .* θ
f(x, u) = 0.9 * x + u
dyn(x, θ, t) = f(x, policy(x, θ))
w(x, θ, t) = sum(x.^2 + policy(x, θ).^2)
T = 1.0

# We're assuming that J(x(T)) = 0, so we can skip dJ/dx(T) = 0. The `adjoint_sensitivities` function doesn't seem to
# have support for terminal costs.
poop(x0, θ) = begin
    prob = ODEProblem(dyn, x0, (0.0, T))
    sol = solve(prob, Tsit5(), p=θ)
    res, err = quadgk(0.0, T) do t
        x = sol(t)
        w(x, policy(x, θ), t)
    end
    res
end

import FiniteDifferences
dpoop_fd(x0, θ) = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), poop, x0, θ)

import DiffEqSensitivity: adjoint_sensitivities, InterpolatingAdjoint
import Zygote
dwdx(out, x, θ, t) = begin
    (dx,) = Zygote.gradient((x) -> w(x, θ, t), x)
    out .= dx
end
# Note that this is technically ∂w/∂u ∂u/∂θ which is slightly different from dw/dθ since x(t) also depends on θ.
dwdθ(out, x, θ, t) = begin
    (dθ,) = Zygote.gradient((θ) -> w(x, θ, t), θ)
    out .= dθ
end

dpoop_de(x0, θ) = begin
    prob = ODEProblem(dyn, x0, (0.0, T))
    sol = solve(prob, Tsit5(), p=θ)
    adjoint_sensitivities(sol, Tsit5(), w, nothing, (dwdx, dwdθ))
end

dpoop_custom(x0, θ) = begin
    prob = ODEProblem(dyn, x0, (0.0, T))
    x = solve(prob, Tsit5(), p=θ)

    adj = solve(ODEProblem((at, _, t) -> begin
        xt = x(t)

        # Calculate dw/dx(t)
        (dwdx, _, _) = Zygote.gradient(w, xt, θ, t)

        # Calculate a^T ∂f/∂x(t)
        u = policy(xt, θ)
        _, pb = Zygote.pullback(f, xt, u)
        (aT_∂f∂x, ) = pb(at)

        -dwdx - aT_∂f∂x
    end, [0.0], (T, 0.0)))

    g = solve(ODEProblem((gt, _, t) -> begin
        xt = x(t)
        at = adj(t)

        # Calculate a^T ∂f/∂u ∂π/∂θ
        _, pb = Zygote.pullback((θ) -> f(xt, policy(xt, θ)), θ)
        (aT_∂f∂u_∂π∂θ, ) = pb(at)

        # Calculate ∂w/∂u ∂π/∂θ
        (∂w∂u_∂π∂θ, ) = Zygote.gradient((θ) -> w(xt, θ, t), θ)

        -aT_∂f∂u_∂π∂θ - ∂w∂u_∂π∂θ
    end, [0.0], (T, 0.0)))

    adj[end], g[end]
end

import Random: seed!
seed!(123)
for _ in 1:10
    x0 = 0.1 * randn(1)
    θ = 0.1 * randn(1)
    @show dpoop_fd(x0, θ)
    @show dpoop_de(x0, θ)
    @show dpoop_custom(x0, θ)
    println()
end
