include("deps.jl")

using Shapes

import Random: seed!

include("mj_utils.jl")
include("models/reacher.jl"); envt = Reacher

Zygote.refresh()

mjd = MJDerivEnvs(envt;
                  nthreads=Threads.nthreads(), eps=1e-4, nwarmup=1, niter=1)
setupenv = mjd.envs[1]

nh = 32
const naug = 0 #8 # nope, doesn't help
nobs = length(obsspace(setupenv)) #+ naug
nu = setupenv.sim.m.nu
act = tanh
policy = FastChain(FastDense(nobs + naug, nh, act),
                   FastDense(nh, nh, act),
                   FastDense(nh, nu + naug,
                             #initW=(x...)->Flux.glorot_uniform(x...).*1e-1, # smaller outputs
                             #initb=Flux.glorot_uniform
                            )
                   )
# TODO need shapes for pol input AND output, as well as shapes for state input/output
#policy = FastChain(FastDense(nobs, setupenv.sim.m.nu, #x->clamp(x, -1.0f0, 1.0f0),
#                             initW=(x...)->zeros(Float32, x...)))

p0 = initial_params(policy)
@info "policy size " size(p0)

#f = get_mjsys(mjd, policy)
f(u,p,t) = mjsystem(u, p, t, mjd, policy)

testenv = mjd.envs[1]
reset!(testenv)
#u0 = [getstate(setupenv)[2:end]; 0.0]
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

u0 = zeros(length(inputshape))
inputshape(u0).state .= fast_getstate(testenv)
inputshape(u0).cost = -getreward(getstate(testenv),
                                         getaction(testenv),
                                         getobs(testenv),
                                         testenv)

T = 2.0
H = T
maxiters = 4e3
tspan = (0, H)

dt = timestep(testenv) / testenv.sim.skip # undo skip
dt = dt / 2
solveargs = (dt=dt, ) # works decently

solver = Tsit5()
#solver = TRBDF2(autodiff=false,diff_type=Val{:forward})
prob = ODEProblem(f, u0, tspan, p0)
@time rollout = solve(prob, solver, u0=u0, p=p0; solveargs...)

# check gradients / loss
#println("ODE gradients")
#gradcheck((p,n)->odeloss(f, p, tspan, mjd; solver=solver, dt=dt, batchsize=n), p0, 8)
#println("bptt gradients")
#gradcheck((p,n)->bpttloss(f, p, tspan, mjd, dt; batchsize=n), p0, 8)

##### save inits to test conditions / env setup. visualize it.
#@time saverollouts("/tmp/ode_init_$envt.jlso", f, mjd, solver,
#                   policy, p0, T, 10)

#losses=Vector{Float64}(undef, 0) # TODO make callback structure to save this
#niter = 200
bs = 4

#solver=Euler()
function dotrain(p0, niter, opt=ADAM(0.01), bs=bs)
    losses = zeros(0)
    #@time res1 = DiffEqFlux.sciml_train(p->odeloss(f, p, tspan, mjd; batchsize=bs, solver=solver, dt=dt, sensealg=BacksolveAdjoint()),
    #                                    p0, opt;
    #                                    cb=(x...)->mj_callback(x..., mjd, losses, T, niter), maxiters=niter) # 1st order
    #mj_callback(res1.minimizer, res1.minimum, mjd, losses, T; doplot=true)
    #res1.minimizer
    #grad(p_in) = Zygote.gradient(p->odeloss(f, p, tspan, mjd; batchsize=bs, sensealg=BacksolveAdjoint()), p_in)
    l(p) = odeloss(f, p, tspan, mjd; batchsize=bs, solver=solver, dt=dt)
    #l(p) = tcostodeloss(f, p, tspan, mjd; batchsize=bs, solver=solver, dt=dt)
    p = copy(p0)
    @showprogress for i=1:niter
        #g = grad(p)
        loss, pull = Zygote.pullback(l, p)

        Flux.update!(opt, p, pull(1)[1])
        #clamp!(p, -1.0f0, 1.0f0)
        #Flux.Optimise.apply!(ClipNorm(2.0f0), 0, p)

        mj_callback(p, loss, mjd, losses, T, niter)
    end
    p
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
    r = range(-0.2, 0.2, length=N)
    xA[1,:] = repeat(r', N)
    xA[2,:] = vec(repeat(r', N)')
    xA[3,:] .= 0.2
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

function mylearn(p0, opt, nf, niter, clip=0.1f0; bptt=false)
    ps = copy(p0)
    losses = Vector{Float64}(undef, 0) #zeros(niter)
    lvs = Vector{Float64}(undef, 0)
    as = zeros(niter)
    pnorm = zeros(niter)

    alpha = opt.eta
    alpharange = exp10.(range(log10(alpha), -3.0, length=6))

    #l(p) = odeloss(f, p, tspan, mjd; batchsize=bs)
    l(p) = tcostodeloss(f, p, tspan, mjd; batchsize=bs, solver=solver, dt=dt)
    #end
    #l(p) = bptt ? bpttloss(f, p, tspan, mjd, dt; batchsize=bs) : odeloss(f, p, tspan, mjd; batchsize=bs)
    prog = Progress(niter)
    for i=1:niter
        lossval, pull = Zygote.pullback(l, ps)
        ∇ps = pull(1)[1]

        ls_nf = 0
        ls = [ begin
                  alphals = l(ps - a*∇ps)
                  #ls_nf += pop!(nf)
                  #pop!(lvs)
                  alphals
              end for a in alpharange ]
        #nf[end] += ls_nf # add the linesearch function evals
        push!(lvs, lossval)
        push!(nf, fwd_counter[])

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
        #clamp!(ps, -clip, clip)

        mj_callback(ps, lossval, mjd, losses, T; skipiter=1000)
        ProgressMeter.next!(prog; showvalues=[(:loss, lossval), (:evals, nf[end])])
    end
    #display(lineplot(pnorm, width=60, height=6, title="gradnorm"))
    #display(lineplot(as, width=60, height=6, ylabel="steps"))
    ps, lvs #losses
end

function _getreacherdemos(N)
    s,_,_,r = getdemos(envt, N, Int(T/timestep(setupenv)), 8, 16)
    env = mjd.envs[1]
    nqnv = env.sim.m.nq+env.sim.m.nv
    t, q = mj2ode(s, r, nqnv)
end

function clonetest(p0, N, niter, opt=ADAM(0.001), tq = _getreacherdemos(N))
    t, q = tq

    tq = zip(t, q)

    losses = zeros(0)
    res1 = DiffEqFlux.sciml_train(p->refloss(f, p, tspan, mjd, tq), p0, opt;
                                  cb=(x...)->ref_callback(x..., mjd, losses, zip(t,q), niter),
                                  maxiters=niter) # 1st order
    res1.minimizer
end

function collotest(p0, N, niter, opt=ADAM(0.001), tq = _getreacherdemos(N))
    t, q = tq

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
