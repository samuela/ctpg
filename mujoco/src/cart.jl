using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, DiffEqSensitivity
using UnicodePlots
using LyceumMuJoCo, MuJoCo, Shapes
using Shapes, LyceumBase.Tools
using UnsafeArrays, Random
using Distributions, LinearAlgebra, Distances
using Zygote
using Base: @propagate_inbounds
using LineSearches
using ProgressMeter
#using DifferentialEquations: ZygoteAdjoint

import Random: seed!

include("mj_utils.jl")
<<<<<<< HEAD
include("rff.jl")
=======
>>>>>>> 87a74c89cca19651bc12697b0cb707076f2b0373
include("models/cartpolev2.jl"); envt = CartpoleSwingupV2

Zygote.refresh()

mjd = MJDerivEnvs(envt;
                  nthreads=Threads.nthreads(), eps=1e-6, nwarmup=1, niter=1)
setupenv = mjd.envs[1]

nh = 64 #128 #56 #128
infeat = length(obsspace(setupenv))
act = elu #tanh
policy = FastChain(FastDense(infeat, nh, act),
                   FastDense(nh, nh, act),
                   FastDense(nh, setupenv.sim.m.nu, tanh,
                            #initW=(x...)->Flux.glorot_uniform(x...).*1e-2 # smaller outputs
                            )
                   )

#rfflayer = RandomFourierFunctions{Float32}(0.8f0, infeat, 128) 
#
#policy = FastChain((x,_) -> rfflayer(x),
#                   FastDense(128, setupenv.sim.m.nu, tanh))

#policy = FastChain(FastDense(infeat, setupenv.sim.m.nu, tanh,
#                             initW=(x...)->zeros(Float32, x...)))
p0 = initial_params(policy)

#fluxpolicy = Chain(Dense(infeat, nh, act),
#                  # Dense(nh, nh, act),
#                   Dense(nh, setupenv.sim.m.nu, tanh,
#                         #)
#                         initW=(x...)->Flux.glorot_uniform(x...).*1e-2) # smaller outputs
#                  )
#fluxpolicy = Chain(Dense(infeat, setupenv.sim.m.nu, tanh,
#                         initW=(x...)->zeros(Float32, x...)))
#p0, policy = Flux.destructure(fluxpolicy)


@info "policy size " size(p0)

#f = get_mjsys(mjd, policy)
f(u,p,t) = mjsystem(u, p, t, mjd, policy)

testenv = mjd.envs[1]
reset!(testenv)
#u0 = [getstate(setupenv)[2:end]; 0.0]
u0 = [fast_getstate(testenv); 0.0]

T = 3.0 # hopper should be 8
H = T #/3 #1.0 #2.0
maxiters = 4e3
tspan = (0, H)

dt = timestep(testenv) / testenv.sim.skip # undo skip
solveargs = (dt=dt, ) # works decently

solver = Tsit5()
#solver = Euler()
prob = ODEProblem(f, u0, tspan, p0)
@time rollout = solve(prob, solver, u0=u0, p=p0; solveargs...)

function loss(p, mjd, batchsize=1, H=H)
    b = Zygote.bufferfrom(zeros(batchsize))

    #Zygote.@ignore Base.GC.enable(false)
    Zygote.@ignore BLAS.set_num_threads(1)
    Threads.@sync for i=1:batchsize
        Threads.@spawn begin
            env = mjd.envs[Threads.threadid()]
            randreset!(env)
            newstart = [fast_getstate(env); 0.0]
            if H > 0.0
                Zygote.@ignore begin
                    # do a rollout up to time H without gradients
                    # with a random restart
                    fwdpass = Array(solve(ODEProblem(f, newstart, (0, H), p),
                                          solver,
                                          u0=newstart,
                                          p = p;
                                          solveargs..., 
                                          saveat=(H,),
                                          saveend=true,
                                          maxiters=maxiters))
                    newstart = fwdpass[:,end]
                    newstart[end] = 0.0
                end
            end
            rollout = Array(solve(ODEProblem(f, newstart, (H, T), p),
                                  solver,
                                  u0 = newstart,
                                  p  = p;
                                  solveargs..., 
                                  maxiters = maxiters,
                                  #saveat=(0,T),
                                  saveend=true))
            cost = rollout[end, end]
            b[i] = cost
        end
    end
    #Zygote.@ignore Base.GC.enable(true)
    
    sum(copy(b))/batchsize # divide for our reference
end

function mpcloss(p, mjd, batchsize=1)
    b = Zygote.bufferfrom(zeros(batchsize))

    BLAS.set_num_threads(1)
    Threads.@sync for i=1:batchsize
        Threads.@spawn begin
            env = mjd.envs[Threads.threadid()]
            randreset!(env)
            newstart = [fast_getstate(env); 0.0]
            H = rand(Uniform(0.0, T))
            Zygote.@ignore begin
                # do a rollout up to time H without gradients
                # with a random restart
                fwdpass = Array(solve(ODEProblem(f, newstart, (0, H), p),
                                      solver,
                                      u0=newstart,
                                      p = p;
                                      solveargs..., 
                                      saveat=(H,),
                                      saveend=true,
                                      maxiters=maxiters))
                newstart = fwdpass[:,end]
                newstart[end] = 0.0
            end
            rollout = Array(solve(ODEProblem(f, newstart, (H, H+2.0), p),
                                  solver,
                                  u0 = newstart,
                                  p  = p;
                                  solveargs..., 
                                  maxiters = maxiters,
                                  #saveat=(0,T),
                                  saveend=true))
            cost = rollout[end, end] #+ 0.01f0*sum(abs2, p)
            b[i] = cost
        end
    end
    
    sum(copy(b))/batchsize # divide for our reference
end

# check gradients / loss
#println("  gradient time, $(Threads.nthreads()) threads:")
#@time testloss = Zygote.gradient(p->loss(p, mjd, Threads.nthreads()), p0)
#println("gradient norm: ", norm(testloss))

##### save inits to test conditions / env setup. visualize it.
#@time saverollouts("/tmp/ode_init_$envt.jlso", f, mjd, solver,
#                   policy, p0, T, 10)

losses=Vector{Float64}(undef, 0) # TODO make callback structure to save this
niter = 200
bs = 4 #16
opt = ADAM(0.03)
#opt = Descent(0.01)
#opt = Momentum(0.0001)
#@time res1 = DiffEqFlux.sciml_train(p->mpcloss(p, mjd, bs), p0, opt;
#                                    cb=(x...)->mj_callback(x..., mjd, T), 
#                                    maxiters=niter) # 1st order

#opt = BFGS()
#@time res1 = DiffEqFlux.sciml_train(p->loss(p, mjd, bs), p0, opt;
#                                    cb=(x...)->mj_callback(x..., mjd, T), 
#                                    maxiters=niter, allow_f_increases=true) # bfgs

#p0 = res1.minimizer

function mylearn(init_p, alpha, opt, niter, clip=0.1f0)
<<<<<<< HEAD
    ps = copy(init_p)
=======
    ps = init_p
>>>>>>> 87a74c89cca19651bc12697b0cb707076f2b0373
    losses = Vector{Float64}(undef, 0) #zeros(niter)
    as = zeros(niter)
    pnorm = zeros(niter)
    alpharange = exp10.(range(log10(alpha), -3.0, length=6))

    l(p) = loss(p, mjd, bs, 0.0)
    #l(p) = mpcloss(p, mjd, bs)
    @showprogress for i=1:niter
        lossval, pull = Zygote.pullback(l, ps)
        ∇ps = copy(pull(1)[1])
        
        ls = [ l(ps - a*∇ps) for a in alpharange ]
        #display(lineplot(log10.(alpharange), ls, width=60, height=3))
        minloss, aidx = findmin(ls)
        if minloss < lossval
            a = alpharange[aidx]
        else
            a = Float32(1e-6) #zero(ps[1]) #0.0
        end
        as[i] = a

        #ps = ps - a * ∇ps
        opt.eta = a
        Flux.Optimise.update!(opt, ps, ∇ps)

        pnorm[i] = norm(ps)
        clamp!(ps, -clip, clip)

        mj_callback(ps, lossval, mjd, losses, T)
    end
    mj_callback(ps, losses[end], mjd, losses, T; doplot=true)
    #display(lineplot(log10.(as), width=60, height=6, ylabel="10^"))
    display(lineplot(pnorm, width=60, height=6, title="gradnorm"))
<<<<<<< HEAD
    display(lineplot(as, width=60, height=6, ylabel="steps"))
    ps
end

#p1 = mylearn(p0, 0.03, Descent(0.03), 200, 1.0f0);
=======
    display(lineplot(as, width=60, height=6, ylabel="10^"))
    ps
end

p1 = mylearn(p0, 0.03, Descent(0.03), 200, 1.0f0);
>>>>>>> 87a74c89cca19651bc12697b0cb707076f2b0373

#=
opt = BFGS()#initial_stepnorm = 0.1)
#opt = LBFGS()
    #opt = ADAM(0.05)
for h = T:-0.2:0.0 #in [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.5, 3.0]

    println(h)
    niter=50
    opt = ADAM(0.01)
    @time res1 = DiffEqFlux.sciml_train(p->loss(p, mjd, bs, h), p0, opt;
                                        cb=(x...)->mj_callback(x..., mjd, T-h;skipiter=niter), 
                                        maxiters=niter) # 1st order
    #opt = BFGS(initial_stepnorm = 0.1)
    #@time res1 = DiffEqFlux.sciml_train(p->loss(p, mjd, bs), p0, opt;
    #                                    cb=(x...)->mj_callback(x..., mjd, h), 
    #                                    maxiters=niter, allow_f_increases=true) # bfgs
    display(res1)
    global p0 = res1.minimizer
end
mj_callback(p0, losses[end], mjd, T; doplot=true)
=#

#opt = BFGS(initial_stepnorm = 0.1;
#           alphaguess=InitialHagerZhang(α0=0.1),
#           linesearch=HagerZhang())
#opt = LBFGS(; alphaguess=InitialStatic(alpha=0.01, scaled=true), 
#            linesearch=HagerZhang())

#mj_callback(res1.minimizer, res1.minimum, mjd, T; doplot=true)


