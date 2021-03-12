include("deps.jl")

import Random: seed!

include("mj_utils.jl")
include("models/pointmass.jl"); envt = PointMassV2

Zygote.refresh()

mjd = MJDerivEnvs(envt;
                  nthreads=Threads.nthreads(), eps=1e-6, nwarmup=1, niter=1)
setupenv = mjd.envs[1]
nu = setupenv.sim.m.nu
const naug = 0 #8 # nope, doesn't help
inputshape = MultiShape(aug = VectorShape(Float64, naug),
                        state = MultiShape(pos = VectorShape(Float64, setupenv.sim.m.nq),
                                           vel = VectorShape(Float64, setupenv.sim.m.nv)),
                        cost = ScalarShape(Float64))
outputshape = MultiShape(aug = VectorShape(Float64, naug),
                         state = MultiShape(vel = VectorShape(Float64, setupenv.sim.m.nq),
                                            acc = VectorShape(Float64, setupenv.sim.m.nv)),
                         cost = ScalarShape(Float64))
#polinshape = MultiShape(obs = VectorShape(Float64, nobs),
#                        aug = VectorShape(Float64, naug))
poloutshape = MultiShape(ctrl = VectorShape(Float64, nu),
                         aug = VectorShape(Float64, naug))



nh = 32
infeat = length(obsspace(setupenv))
act = tanh
policy = FastChain(FastDense(infeat, nh, act),
                   FastDense(nh, nh, act),
                   FastDense(nh, setupenv.sim.m.nu, #x->clamp(x, -1.0f0, 1.0f0),
                            #)
                             initW=(x...)->Flux.glorot_uniform(x...).*0.01f0) # smaller outputs
                   )
#policy = FastChain(FastDense(infeat, setupenv.sim.m.nu, x->clamp(x, -1.0f0, 1.0f0),
#                             initW=(x...)->zeros(Float32, x...)))
p0 = initial_params(policy)

#fluxpolicy = Chain(Dense(infeat, nh, act),
#                   Dense(nh, nh, act),
#                   Dense(nh, setupenv.sim.m.nu, x->clamp(x, -1.0f0, 1.0f0),
#                         #)
#                         initW=(x...)->Flux.glorot_uniform(x...).*1e-2) # smaller outputs
#                  )
#fluxpolicy = Chain(Dense(infeat, setupenv.sim.m.nu, x->clamp(x, -1.0f0, 1.0f0),
#                         initW=(x...)->zeros(Float32, x...)))
#p0, policy = Flux.destructure(fluxpolicy)

@info "policy size " size(p0)

f(u,p,t) = mjsystem(u, p, t, mjd, policy)

testenv = mjd.envs[1]
reset!(testenv)
u0 = [fast_getstate(testenv); 0.0]

T = 0.5 #2.0
H = T #0.5
maxiters = 4e3
tspan = (0, H)

dt = timestep(testenv) / testenv.sim.skip # undo skip
solveargs = (dt=dt, ) # works decently

solver = VCABM() #Tsit5()
#solver = Euler()
@time rollout = solve(ODEProblem(f, u0, tspan, p0), solver, u0=u0, p=p0; dt=dt)
#@time rollout = solve(ODEProblem(f, u0, tspan, p0), Tsit5(); sensealg=SensitivityADPassThrough())

# The following is broken!!
#testloss(p) = Array(solve(ODEProblem(f, u0, tspan, p0), Tsit5(); sensealg=SensitivityADPassThrough()))[end, end]
#println("sensitivity ad")
#@time testloss(p0)
#println("gradient of sensitivity ad")
#@time g = Zygote.gradient(testloss, p0)

# check gradients / loss for number of threads
#println("ODE gradients")
#gradcheck((p,n)->odeloss(f, p, tspan, mjd; solver=solver, solveargs=solveargs, batchsize=n), p0, 8)
#println("bptt gradients")
#gradcheck((p,n)->bpttloss(f, p, tspan, mjd, dt; batchsize=n), p0, 8)

##### save inits to test conditions / env setup. visualize it.
#@time saverollouts("/tmp/ode_init_$envt.jlso", f, mjd, solver,
#                   policy, p0, T, 10)

niter = 100
losses = zeros(0)
bs = 4
#opt = ADAM(0.01)
#opt = Descent(0.1)
#opt = Momentum(0.01)
#opt = BFGS(initial_stepnorm = 0.01)
#opt = LBFGS(; alphaguess=InitialStatic(alpha=0.01, scaled=true),
#            linesearch=HagerZhang())

function dotrain(p0, niter, opt=ADAM(0.001), bs=bs)
    losses = zeros(0)
    println("Point Mass")
    @time res1 = DiffEqFlux.sciml_train(p->odeloss(f, p, tspan, mjd; batchsize=bs, dt=dt), p0, opt;
    #@time res1 = DiffEqFlux.sciml_train(p->tcostodeloss(f, p, tspan, mjd; batchsize=bs, dt=dt), p0, opt;
                                        cb=(x...)->mj_callback(x..., mjd, losses, T, niter), maxiters=niter) # 1st order
    #cb=(x...)->mj_callback(x..., mjd, losses, T), maxiters=niter, allow_f_increases=true) # bfgs
    mj_callback(res1.minimizer, res1.minimum, mjd, losses, T; doplot=true)
    res1.minimizer
end

function valuetrain(p0, niter, opt=ADAM(0.001), bs=bs)
    losses = zeros(0)
    nh = 64
    act = tanh
    env = mjd.envs[Threads.threadid()]
    infeat = length(fast_getstate(env))
    valuef = Chain(Dense(infeat, nh, act),
                   Dense(nh, nh, act),
                   Dense(nh, 1)
                  )

    l(p, v) = valueloss(f, p, tspan, mjd, v; batchsize=bs, solver=solver, dt=dt)
    p = copy(p0)
    prog = Progress(niter)

    valueopt = ADAM(0.001)
    valuelosses = zeros(0)
    N = 30
    xA = zeros(infeat, N*N)
    r = range(-1.35, 1.35, length=N)
    xA[1,:] = repeat(r', N)
    xA[2,:] = vec(repeat(r', N)')
    #display(reshape(xA[1,:], N, N))
    #display(reshape(xA[2,:], N, N))

    for i=1:niter
        loss, pull = Zygote.pullback(p->l(p, valuef), p)

        grad = pull(1)[1]
        ng = norm(grad)
        if ng == 0.0
            @info "No gradient"
        end

        Flux.update!(opt, p, grad)

        #### value
        data = []
        c = Threads.Condition()
        Threads.@threads for j=1:64
        #for j=1:64
            env = mjd.envs[Threads.threadid()]
            randreset!(env)
            x0 = fast_getstate(env)
            u0 = [zeros(naug);
                  x0;
                  0.0 # NOTE should just be value = H-horizon cumulative reward
                 ]
            rollout = Array(oderollout(f, u0, tspan, p;
                                       solver=solver,
                                       sensealg=InterpolatingAdjoint(),
                                       dt=dt))
            #rollout = Array(solve(ODEProblem(f, u0, tspan, p), solver, u0=u0, p=p; dt=dt))
            lock(c)
            push!(data, (rollout[1:end-1,1], rollout[end,end])) # TODO map this better
            unlock(c)
        end
        lossf(x,y) = Flux.mse(valuef(x), y)
        evalcb() = push!(valuelosses, mapreduce(d->lossf(d...), +, data))
        for e=1:2
            Flux.train!(lossf, Flux.params(valuef), data, valueopt;
                       )#cb=Flux.throttle(evalcb, 5))
        end
        push!(valuelosses, mapreduce(d->lossf(d...), +, data))
        if mod1(i, 10) == 10
            display(heatmap(reshape(valuef(xA), N, N), width=N, height=N, title="value"))
        end
        # TODO heatmap of value function for pointmass to see if it's decent
        #### value

        mj_callback(p, loss, mjd, losses, T, niter)
        ProgressMeter.next!(prog; showvalues=[(:loss, loss), (:grad, ng)])
    end

    display(lineplot(valuelosses))
    p, valuef
end


function _getpointmassdemos(N)
    s,_,_,r = getdemos(envt, N, Int(T/timestep(setupenv)))
    env = mjd.envs[1]
    nqnv = env.sim.m.nq+env.sim.m.nv
    t, q = mj2ode(s, r, nqnv)
end

function clonetest(p0, N, niter, opt=ADAM(0.001), tq = _getpointmassdemos(N))
    t, q = tq[1], tq[2]
    tq = zip(t, q)

    losses = zeros(0)
    res1 = DiffEqFlux.sciml_train(p->refloss(f, p, tspan, mjd, tq), p0, opt;
                                  cb=(x...)->ref_callback(x..., mjd, losses, zip(t,q), niter),
                                  maxiters=niter) # 1st order
    res1.minimizer
end

function collotest(p0, N, niter, opt=ADAM(0.001), tq = _getpointmassdemos(N))
    t, q = tq[1], tq[2]

    #du_u_t = zip(batchcollocation(t, q)..., t)
    du, u = batchcollocation(t, q)

    du = reduce(hcat, du)
    u = reduce(hcat, u)
    #N = size(du, 2)
    #u = vcat(u, zeros(1,N))
    ndata = size(du, 2)

    losses = zeros(0)
    function stocloss(p) # prevents blowup, but not good?
        #idx = rand(1:ndata, 10) # take random index of data
        i = rand(1:ndata)
        colloloss(f, p, du[:,i], u[:,i], dt)
    end

    #res1 = DiffEqFlux.sciml_train(p->colloloss(f, p, du, u, dt), p0, opt;
    numiter = niter * ndata
    res1 = DiffEqFlux.sciml_train(stocloss, p0, opt;
                                  cb=(x...)->ref_callback(x..., mjd, losses, zip(t,q), numiter),
                                  maxiters=numiter) # 1st order
    res1.minimizer
end

function mylearn(init_p)
    ps = init_p
    losses = Vector{Float64}(undef, 0) #zeros(niter)

    l(p) = odeloss(f, p, tspan, mjd; batchsize=bs)
    for i=1:niter
        loss, pull = Zygote.pullback(l, ps)

        ps = ps - 0.01 * pull(1)[1]
        mj_callback(ps, loss, mjd, losses, T, niter)
    end
    mj_callback(ps, losses[end], mjd, losses, T, niter; doplot=true)
    p2
end

function pntlearn(init_p, alpha, niter; bptt=false)
    ps = init_p
    losses = Vector{Float64}(undef, 0) #zeros(niter)
    as = zeros(niter)
    alpharange = exp10.(range(log10(alpha), -3.0, length=7))

    if bptt
        println("doing bptt!")
    else
        println("doing ODE adjoints!")
    end
    l(p) = bptt ? bpttloss(f, p, tspan, mjd, dt; batchsize=bs) : odeloss(f, p, tspan, mjd; batchsize=bs)

    @showprogress for i=1:niter
        lossval, pull = Zygote.pullback(l, ps)
        ∇ps = pull(1)[1]

        ls = [ l(ps - a*∇ps) for a in alpharange ]
        #display(lineplot(log10.(alpharange), ls, width=60, height=3))
        minloss, aidx = findmin(ls)
        if minloss < lossval
            a = alpharange[aidx]
            as[i] = a
        else
            a = zero(ps[1]) #0.0
        end

        ps = ps - a * ∇ps
        #ps = ps - 0.01 * gradp
        mj_callback(ps, lossval, mjd, losses, T)
    end
    mj_callback(ps, losses[end], mjd, losses, T; doplot=true)
    display(lineplot(as, width=60, height=6, title="α"))
    ps
end

#p1 = pntlearn(p0, 0.01, 50)
