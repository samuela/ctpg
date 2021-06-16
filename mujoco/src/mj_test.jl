using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, DiffEqSensitivity
using UnicodePlots
using LyceumMuJoCo, MuJoCo, Shapes
using Shapes, LyceumBase.Tools
using UnsafeArrays, Random
using Distributions, LinearAlgebra, Distances
using Zygote
using Base: @propagate_inbounds
using LineSearches
#using DifferentialEquations: ZygoteAdjoint

import Random: seed!

include("mj_utils.jl")
include("mjderiv.jl")
#include("models/acrobot.jl"); envt = Acrobot
#include("models/reacher.jl"); envt = Reacher
#include("models/armhand.jl"); envt = ArmHandPickup # QUAT
#include("models/hopper_local.jl"); envt = Hopper
include("models/pointmass.jl"); envt = PointMassV2

Zygote.refresh()

setupenv = envt()

nh = 128 #56 #128
infeat = setupenv.sim.m.nq+setupenv.sim.m.nv
act = relu #tanh
policy = FastChain(FastDense(infeat, nh, act),
                   FastDense(nh, nh, act),
                   FastDense(nh, setupenv.sim.m.nu,
                             initW=(x...)->Flux.glorot_uniform(x...).*1e-2) # smaller outputs
                   #FastDense(nh, setupenv.sim.m.nu)
                   )

policy = FastChain(FastDense(infeat, setupenv.sim.m.nu, initW=(x...)->zeros(Float32, x...)))

# siren
#unifrand(ins, x...) = Float32.(rand(Uniform(-sqrt(1/ins), sqrt(1/ins)), x...))
#sireninit(ins, omega, x...) = Float32.(rand(Uniform(-sqrt(6/ins)/omega, sqrt(6/ins)/omega), x...))
#sirenlayer(ins, outs, omega) = FastDense(ins, outs, x->sin(omega*x),
#                                         initW=(x...)->sireninit(ins, omega, x...),
#                                         initb=(x...)->unifrand(ins, x...))
#policy = FastChain(FastDense(infeat, nh,
#                             x->sin(30.0f0*x),
#                             initW=(x...)->Float32.(rand(Uniform(-1/infeat, 1/infeat), x...)),
#                             initb=(x...)->unifrand(infeat, x...)),
#                   sirenlayer(nh, nh, 30.0f0),
#                   FastDense(nh, setupenv.sim.m.nu, initW=(x...)->sireninit(nh, 30.0f0, x...)))

init_policy_params = initial_params(policy)
@info "policy size " size(init_policy_params)

testenv, mjd, f, jacf = get_mjsys(envt, policy)

reset!(testenv)
u0 = [fast_getstate(testenv); 0.0]

T = 2.0 # 5.0 # hopper should be 8
maxiters = 4e3
tspan = (0, T)
# to use adaptive methods, must provide jacobian function
#prob = ODEProblem(ODEFunction{false}(f, jac=jacf), u0, (0, T), init_policy_params )
prob = ODEProblem(f, u0, tspan, init_policy_params)

dt = timestep(testenv) / testenv.sim.skip # undo skip
#solveargs = (adaptive=false, dt=dt) # works good
#solveargs = (abstol=1e-1, reltol=1e-1, dt=dt, force_dtmin=true, dtmin=dt/10) # works good
#solveargs = (abstol=1e-3, reltol=1e-2, dt=dt)#, dtmax=timestep(testenv)) # works good
#solveargs = (dt=dt, ) # works decently
#solverargs = (nothing, )
#solveargs = (dt=dt, force_dtmin=true)
#solveargs = (dt=dt, dtmin = dt/10000, dtmax = dt*100)#, force_dtmin=true)

solver = Tsit5()
#solver = Vern7()
#solver = BS3()
#solver = AutoTsit5(Rosenbrock23(autodiff=false))
#solver = TRBDF2()
#solver = Rodas5()
#solver = Rosenbrock23()
@time rollout = solve(prob, solver, u0=u0, p=init_policy_params; solveargs...)

function loss(p, batchsize=1)
    #b = 0.0
    b = Zygote.bufferfrom(zeros(batchsize))
    #rolls = Zygote.bufferfrom(Vector{Matrix}(undef, batchsize))

    BLAS.set_num_threads(2)
    Threads.@sync for i=1:batchsize
    #for i=1:batchsize
        Threads.@spawn begin
            env = mjd.envs[Threads.threadid()]
            randreset!(env)
            newstart = [fast_getstate(env); 0.0]
            rollout = Array(solve(ODEProblem(f, newstart, tspan, p),
                                  solver,
                                  u0 = newstart,
                                  p  = p;
                                  solveargs..., 
                                  sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), # this line isn't needed.... ?
                                  maxiters = maxiters,
                                  saveend=true))
            cost = rollout[end, end]
            b[i] = cost
            #b += cost # is this different?
        end
    end
    
    sum(copy(b))/batchsize # divide for our reference
    #min(b/batchsize, 100.0) #, copy(rolls)
end

# check gradients / loss
#println("  gradient time, $(Threads.nthreads()) threads:")
#@time testloss = Zygote.gradient(p->loss(p, Threads.nthreads()), init_policy_params)
#println("gradient norm: ", norm(testloss))

##### save inits to test conditions / env setup. visualize it.
#@time saverollouts("/tmp/ode_init_$envt.jlso", mjd, prob,
#                   policy, init_policy_params, T, 10)

niter = 500
losses=Vector{Float64}(undef, 0)
function callback(p, loss_val;
                  N=3, doplot = false)

    push!(losses, loss_val)

    skip = div(niter, 5)
    #print('.')
    println(loss_val)
    
    if doplot || mod1(length(losses), skip) == skip
        tspan = (0, T)
        tsteps = 0.0f0:0.1f0:(T)
        evals = zeros(length(tsteps), N)
        costs = []
        for m=1:N
            env = mjd.envs[Threads.threadid()]
            randreset!(env)
            newstart = [fast_getstate(env); 0.0]
            s = solve(ODEProblem(f, newstart, tspan, p),
                      solver,
                      u0 = newstart,
                      p  = p;
                      solveargs...,
                      maxiters = maxiters,
                      saveat=tsteps)
            rollout = Array(s)

            push!(costs, rollout[end,:])

            dx  = zeros(size(rollout,1))
            for i=1:size(rollout,2)
                x = rollout[:,i]
                f(x, p, tsteps[i]) # should set env's d
                ctrl = policy(x[1:end-1], p)[1]
                evals[i, m] = geteval(getstate(env), ctrl, getobs(env), env)
            end
        end
        plt = lineplot(tsteps, evals[:,1], title="evals", width=60, height=6)
        cplt = lineplot(tsteps, costs[1], title="costs", width=60, height=6)
        for i=2:N
            lineplot!(plt, tsteps, evals[:,i])
            lineplot!(cplt, tsteps, costs[i])
        end
        display(cplt)
        display(plt)
    end
    return false
end

bs = 16
opt = ADAM(0.01)
#opt = Descent(0.01)
#opt = Momentum(0.01)
#opt = BFGS(initial_stepnorm = 0.001)
#opt = LBFGS(; alphaguess=InitialStatic(alpha=0.01, scaled=true), 
#            linesearch=HagerZhang())

@time res1 = DiffEqFlux.sciml_train(p->loss(p, bs), init_policy_params, opt;
                                    cb=callback, maxiters=niter) # 1st order
                                    #cb=callback, maxiters=niter, allow_f_increases=true) # bfgs
callback(res1.minimizer, res1.minimum; doplot=true)
display(lineplot(losses, title="loss", xlim=(1,length(losses)), height=7))

function itertrain(p)
    @time res1 = DiffEqFlux.sciml_train(p->loss(p, bs), p, Descent(0.01); cb=callback, maxiters=100)
    @time res1 = DiffEqFlux.sciml_train(p->loss(p, bs), res1.minimizer, BFGS(initial_stepnorm=0.01); cb=callback, maxiters=niter) # bfgs
    #for i=1:10
    #    @time res1 = DiffEqFlux.sciml_train(p->loss(p, bs), res1.minimizer, BFGS(); cb=callback, maxiters=niter, allow_f_increases=true) # bfgs
    #    @time res1 = DiffEqFlux.sciml_train(p->loss(p, bs), res1.minimizer, ADAM(0.01); cb=callback, maxiters=50)
    #end
    display(res1)

    callback(res1.minimizer, 0.0; doplot=true)
    plt = lineplot(losses, title="loss", xlim=(1,length(losses)), height=7)
    res1.minimizer
end


