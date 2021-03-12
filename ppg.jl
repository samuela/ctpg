import DiffEqBase
import DiffEqSensitivity:
    solve,
    ODEProblem,
    ODEAdjointProblem,
    InterpolatingAdjoint,
    BacksolveAdjoint,
    QuadratureAdjoint,
    AdjointSensitivityIntegrand
import QuadGK: quadgk!
import ThreadPools: qmap, tmap, bmap
import Zygote
import Statistics: mean

function extract_loss_and_xT(fwd_sol)
    fwd_sol[end][1], fwd_sol[end][2:end]
end

"""Returns a differentiable loss function that rolls out a policy in an
environment and calculates its cost."""
function ppg_goodies(dynamics, cost, policy, T)
    # function aug_dynamics!(dz, z, policy_params, t)
    #     x = @view z[2:end]
    #     u = policy(x, t, policy_params)
    #     dz[1] = cost(x, u)
    #     # Note that dynamics!(dz[2:end], x, u) breaks Zygote/ReverseDiff :(
    #     dz[2:end] = dynamics(x, u)
    # end
    function aug_dynamics(z, policy_params, t)
        x = @view z[2:end]
        u = policy(x, t, policy_params)
        [cost(x, u); dynamics(x, u)]
    end

    # using BenchmarkTools
    # @benchmark aug_dynamics!(
    #     rand(floatT, x_dim + 1),
    #     rand(floatT, x_dim + 1),
    #     init_policy_params,
    #     0.0,
    # )

    # See https://discourse.julialang.org/t/why-the-separation-of-odeproblem-and-solve-in-differentialequations-jl/43737
    # for a discussion of the performance of the pullbacks.
    function loss_pullback(x0, policy_params, solvealg, solve_kwargs)
        z0 = vcat(0.0, x0)
        fwd_sol = solve(
            ODEProblem(aug_dynamics, z0, (0, T), policy_params),
            solvealg,
            u0 = z0,
            p = policy_params;
            solve_kwargs...,
        )

        # TODO: this is not compatible with QuadratureAdjoint because nothing is
        # consistent... See https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/quadrature_adjoint.jl#L171.

        function _adjoint_solve(g_zT, sensealg; kwargs...)
            # See https://diffeq.sciml.ai/stable/analysis/sensitivity/#Syntax-1
            # and https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/sensitivity_interface.jl#L9.
            solve(
                ODEAdjointProblem(
                    fwd_sol,
                    sensealg,
                    (out, x, p, t, i) -> (out[:] = g_zT),
                    [T],
                ),
                solvealg;
                kwargs...,
            )
        end

        # This is the pullback using the augmented system and a discrete
        # gradient input at time T. Alternatively one could use the continuous
        # adjoints on the non-augmented system although this seems to be slower
        # and a less stable feature.
        function pullback(g_zT, sensealg::BacksolveAdjoint)
            bwd_sol = _adjoint_solve(
                g_zT,
                sensealg,
                dense = false,
                save_everystep = false,
                save_start = false,
                # reltol = 1e-3,
                # abstol = 1e-3,
            )

            # The first z_dim elements of bwd_sol.u are the gradient wrt z0,
            # next however many are the gradient wrt policy_params. The final
            # z_dim are the reconstructed z(t) trajectory.

            # Logic pilfered from https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/sensitivity_interface.jl#L9.
            # We need more control than that interface gives us. Why they negate
            # the gradients is beyond me...
            p = fwd_sol.prob.p
            l = p === nothing || p === DiffEqBase.NullParameters() ? 0 :
                length(fwd_sol.prob.p)
            g_x0 = bwd_sol[end][1:length(fwd_sol.prob.u0)]

            # When running the backsolve adjoint we have additional f
            # evaluations every step of the backwards pass, since we need -f to
            # reconstruct the x path.
            (
                g = -bwd_sol[end][(1:l).+length(fwd_sol.prob.u0)],
                nf = bwd_sol.destats.nf,
                n∇ₓf = bwd_sol.destats.nf,
                n∇ᵤf = bwd_sol.destats.nf,
                x0_reconstructed = bwd_sol[end][end-length(fwd_sol.prob.u0)+1:end],
            )
        end
        function pullback(g_zT, sensealg::InterpolatingAdjoint)
            bwd_sol = _adjoint_solve(
                g_zT,
                sensealg,
                dense = false,
                save_everystep = false,
                save_start = false,
                # reltol = 1e-3,
                # abstol = 1e-3,
            )

            # The first z_dim elements of bwd_sol.u are the gradient wrt z0,
            # next however many are the gradient wrt policy_params.
            p = fwd_sol.prob.p
            l = p === nothing || p === DiffEqBase.NullParameters() ? 0 :
                length(fwd_sol.prob.p)
            g_x0 = bwd_sol[end][1:length(fwd_sol.prob.u0)]

            # We do exactly as many f calls as there are function calls in the
            # forward pass, and in the backward pass we don't need to call f,
            # but instead we call ∇f.
            (
                g = -bwd_sol[end][(1:l).+length(fwd_sol.prob.u0)],
                nf = 0,
                n∇ₓf = bwd_sol.destats.nf,
                n∇ᵤf = bwd_sol.destats.nf,
            )
        end
        function pullback(g_zT, sensealg::QuadratureAdjoint)
            # See https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/quadrature_adjoint.jl#L173.
            # This is 75% of the time and allocs of the pullback. quadgk is
            # actually lightweight relatively speaking.
            bwd_sol = _adjoint_solve(
                g_zT,
                sensealg,
                save_everystep = true,
                save_start = true,
                # reltol = 1e-3,
                # abstol = 1e-3,
            )

            integrand = AdjointSensitivityIntegrand(fwd_sol, bwd_sol, sensealg, nothing)

            # Do in-place quadgk for a smidge more perf.
            quad_nf = Ref(0)
            g = similar(integrand.p)
            g .= 0
            _, err = quadgk!(
                (out, t) -> (quad_nf[] += 1; integrand(out, t)),
                g,
                0.0,
                T,
                # It's possible to pass abstol and reltol to QuadratureAdjoint.
                atol = sensealg.abstol,
                rtol = sensealg.reltol,
                # order = 3,
            )
            (g = -g, nf = 0, n∇ₓf = bwd_sol.destats.nf, n∇ᵤf = quad_nf[], quadgk_err = err)
        end

        # TODO:
        # * rewrite the error plots to use this version
        # * fix consumers of this api broken by the change in return type

        fwd_sol, pullback
    end

    function ez_loss_and_grad(
        x0,
        policy_params,
        solvealg,
        sensealg;
        fwd_solve_kwargs = Dict(),
    )
        # @info "fwd"
        fwd_sol, vjp = loss_pullback(x0, policy_params, solvealg, fwd_solve_kwargs)
        # @info "bwd"
        bwd = vjp(vcat(1, zero(x0)), sensealg)
        loss, _ = extract_loss_and_xT(fwd_sol)
        # @info "fin"
        loss, bwd.g, (nf = fwd_sol.destats.nf + bwd.nf, n∇ₓf = bwd.n∇ₓf, n∇ᵤf = bwd.n∇ᵤf)
    end

    function euler_with_cost(x0, policy_params, dt, num_steps)
        x = x0
        cost_accum = 0.0
        for _ = 1:num_steps
            # For now we assume that policy isn't doing anything with the `t`
            # input, at least for Euler integration.
            u = policy(x, nothing, policy_params)
            cost_accum += dt * cost(x, u)
            x += dt * dynamics(x, u)
        end
        cost_accum
    end

    function ez_euler_bptt(x0, policy_params, dt)
        num_steps = floor(Int, T / dt)
        loss, pullback =
            Zygote.pullback((θ) -> euler_with_cost(x0, θ, dt, num_steps), policy_params)
        g, = pullback(1.0)
        loss, g, (nf = num_steps, n∇ₓf = num_steps, n∇ᵤf = num_steps)
    end

    function _aggregate_batch_results(res)
        (
            mean(loss for (loss, _, _) in res),
            mean(g for (_, g, _) in res),
            (
                nf = sum(info.nf for (_, _, info) in res),
                n∇ₓf = sum(info.n∇ₓf for (_, _, info) in res),
                n∇ᵤf = sum(info.n∇ᵤf for (_, _, info) in res),
            ),
        )
    end

    function ez_euler_loss_and_grad_many(x0_batch, policy_params, dt)
        _aggregate_batch_results(map(x0_batch) do x0
            ez_euler_bptt(x0, policy_params, dt)
        end)
    end

    function ez_loss_and_grad_many(
        x0_batch,
        policy_params,
        solvealg,
        sensealg;
        fwd_solve_kwargs = Dict(),
    )
        # Using tmap here gives a segfault. See https://github.com/tro3/ThreadPools.jl/issues/18.
        _aggregate_batch_results(
            map(x0_batch) do x0
                ez_loss_and_grad(
                    x0,
                    policy_params,
                    solvealg,
                    sensealg,
                    fwd_solve_kwargs = fwd_solve_kwargs,
                )
            end,
        )
    end

    (
        aug_dynamics = aug_dynamics,
        loss_pullback = loss_pullback,
        ez_loss_and_grad = ez_loss_and_grad,
        ez_loss_and_grad_many = ez_loss_and_grad_many,
        ez_euler_bptt = ez_euler_bptt,
        ez_euler_loss_and_grad_many = ez_euler_loss_and_grad_many,
    )
end
