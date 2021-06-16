
using MuJoCo, LyceumMuJoCo
using Zygote: @adjoint, @nograd
using StaticArrays
using LyceumBase: RealVec
using PrettyTables
using ThreadPools
# x = state 
# u = controls
#=
function fast_dx(env::AbstractMuJoCoEnvironment, mjd::MJDerivEnvs)
m, d = env.sim.m, env.sim.d
nq, nv, nu = Int(m.nq), Int(m.nv), Int(m.nu)

# assume for now x = [qpos; qvel] and not full statespace(env)(x)
# we could do time correctly but qacc_warmstart would fuck up
# TODO missing actuator activations....?

end
=#

struct MJDerivEnvs{E}
    envs::E

    tstate::Vector{Vector{Float64}} # scratch space
    tobs::Vector{Vector{Float64}}

    eps::Float64
    nwarmup::Int
    niter::Int

    function MJDerivEnvs(envtype;
                         nthreads=Threads.nthreads(),
                         eps=1e-6,
                         nwarmup=3,
                         niter=30)
        envs = tconstruct(envtype, nthreads)
        env = envs[1]
        new{typeof(envs)}(envs,
                          [zeros(length(statespace(env))) for _=1:nthreads],
                          [zeros(length(obsspace(env))) for _=1:nthreads],
                          eps, nwarmup, niter)
    end
end

#@nograd time, reset!, randreset!, getstate, getobs, getaction, timestep
#@nograd time #, reset!, randreset!#, getstate, getobs, getaction, timestep
#@nograd getstate!, getobs!
@nograd BLAS.set_num_threads

struct NonParametricPolicy{D}
    a::Matrix{D} # nu x T actions
    t::Vector{D} # 1:T timesteps
    dt::D
end
function NonParametricPolicy(nu, dt, T::Int) # T is number of timesteps
    D = typeof(dt)
    t = D.(collect(0:dt:((T-1)*dt)))
    NonParametricPolicy{D}(zeros(D, nu, T), t, dt)
end
function NonParametricPolicy(nu, dt, T::AbstractFloat) # T is final time
    D = typeof(dt)
    t = D.(collect(0:dt:T))
    NonParametricPolicy{D}(zeros(D, nu, length(t)), t, dt)
end

function (policy::NonParametricPolicy)(t)
    i1 = floor(Int, t/policy.dt) + 1
    i2 = ceil(Int, t/policy.dt) + 1
    if i1 == i2 # same index
        action = policy.a[:,i1]
    else
        w1 = (t-policy.t[i1])/policy.dt
        w2 = (policy.t[i2]-t)/policy.dt
        action = w1 * policy.a[:,i1] + w2 * policy.a[:,i2] # weighted interpolation
    end
    action
end

function (policy::NonParametricPolicy)(t, p)
    i1 = floor(Int, t/policy.dt) + 1
    i2 = ceil(Int, t/policy.dt) + 1
    (r,c) = size(policy.a)
    rp = reshape(p, r, c)
    if i1 == i2 # same index
        action = rp[:,i1]
    else
        w1 = (t-policy.t[i1])/policy.dt
        w2 = (policy.t[i2]-t)/policy.dt
        action = w1 * rp[:,i1] + w2 * rp[:,i2] # weighted interpolation
    end
    action
end

#struct EnsemblePolicy{D<:Tuple}
#    over::FS
#end
struct Ensemble{I,G,FS<:Tuple}
    latent::I
    gate::G
    over::FS
end

function Ensemble(in, z, out, n_alts)
    latent = Dense(in, z, tanh)
    #gate = Chain(Dense(z, n_alts, sigmoid), x->mapslices(y->y./norm(y,1), x; dims=1))
    gate = Chain(Dense(z, n_alts, sigmoid))
    over = Tuple(Dense(z, out) for _ in 1:n_alts)
    return Ensemble{typeof(latent),typeof(gate),typeof(over)}(latent, gate, over)
end

Flux.@functor Ensemble

function (mo::Ensemble)(input::AbstractArray)
    latent = mo.latent(input)
    weights = mo.gate(latent)
    weights = weights ./ sum(weights; dims=1)
    #weights = mapslices(y->y./norm(y,1), weights; dims=1)
    #weights = Tuple( @view weights[i:i,:] for i=1:length(mo.over) ) # group data
    #println((zip(mo.over, weights)))
    #println(weights)
    #println(mapreduce(identity, +, weights))
    #mapreduce((f) -> f[2].*f[1](latent), +, zip(mo.over, weights))
    n_alts = length(mo.over)
    out = mo.over[1](latent) .* weights[1:1, :] 
    for i = 2:n_alts #(i,o) in enumerate(mo.over)
        out += mo.over[i](latent) .* weights[i:i, :]
    end
    out
end

outdims(l::Ensemble, isize) = outdims(first(l.over), isize)


policyfunc(u, t, aug, p, policy::Function)  = vec(policy(p)(vcat(u, aug))) # flux
policyfunc(u, t, aug, p, policy::FastChain) = policy(vcat(u, aug), p) # fastchain
policyfunc(u, t, aug, p, policy::NonParametricPolicy) = policy(t, p)
policyfunc(u, t, p, policy::Function)  = policy(p)(u) # flux
policyfunc(u, t, p, policy::FastChain) = policy(u, p) # fastchain
policyfunc(u, t, p, policy::NonParametricPolicy) = policy(t, p)


@propagate_inbounds function fast_setstate!(env::AbstractEnvironment, state::RealVec)
    setstate!(env, state)
end

@propagate_inbounds function fast_setstate!(env::AbstractMuJoCoEnvironment, state::RealVec)
    sim = env.sim
    #mj_resetData(sim.m, sim.d)
    LyceumMuJoCo.copystate!(sim, state)
    forward!(sim)
    #shaped = statespace(sim)(state)
    #@uviews shaped begin copyto!(sim.d.qacc_warmstart, shaped.qacc_warmstart) end
    sim.d.qacc_warmstart .= 0.0
    sim
end

@propagate_inbounds function fast_getstate!(state::RealVec, sim::MJSim)
    #@boundscheck checkaxes(statespace(sim), state)
    shaped = statespace(sim)(state)
    @uviews shaped begin _copyshaped!(shaped, sim.d) end
    state
end

function copytolyceumstate!(state::RealVec, u::RealVec, t=0.0)
    state .= zero(eltype(state))
    state[1] = t
    odestate = inputshape(u).state
    n = length(odestate) #- 1
    for i=1:n
        state[1+i] = odestate[i]
    end
    state
end

function fast_getstate(env::AbstractMuJoCoEnvironment)
    [env.sim.d.qpos; env.sim.d.qvel]
end
function fast_getstate(env::AbstractEnvironment)
    getstate(env)[2:end] # assume time is index 1
end

function lyceum2ode(states, env)
    sp = statespace(env)
    N = size(states,2)
    roll = zeros(env.sim.m.nq + env.sim.m.nv, N)
    for i=1:N
        s = view(states, :, i)
        roll[:,i] = [sp(s).qpos; sp(s).qvel]
    end
    roll
end

#function forward(x, ẋ, u, mjd; warm=0.0, skip=MuJoCo.MJCore.mjSTAGE_NONE)
#    ẍr = MVector{length(ẋ)+1, Float64}(undef) # similar(ẋ) with reward
#    fwdrwd!(ẍr, x, ẋ, u, mjd; warm=warm, skip=skip)
#end

fwd_counter=Threads.Atomic{Int}(0)

# calculates the accelerations and reward for an input position, velocity, and control
# x, ẋ, u are pos, vel, ctrl
# warm can be used to warmstart mujoco's forward calculations
# skip reduces computation in mujoco
function fwdrwd!(ẍr, x, ẋ, u, mjd; warm=0.0, skip=MuJoCo.MJCore.mjSTAGE_NONE)
    svec = mjd.tstate[Threads.threadid()]
    ovec = mjd.tobs[Threads.threadid()]
    env  = mjd.envs[Threads.threadid()]
    m, d = env.sim.m, env.sim.d
    d.qpos .= x 
    d.qvel .= ẋ
    d.ctrl .= u
    d.qacc_warmstart .= warm # stability / reproducibility

    mj_forwardSkip(m, d, skip, false) # true) # TODO don't skip obs?
    Zygote.@ignore fwd_counter[] += 1

    for i=1:m.nv
        ẍr[i] = d.qacc[i]
    end
    cost = -getreward(getstate!(svec, env), u, getobs!(ovec, env), env)
    #cost = -runningcost(getstate!(svec, env), u, getobs!(ovec, env), env)
    ẍr[end] = cost

    ẍr
end

"""
    mjsystem(u, p, t, mjd, policy)

A dynamics systems f(u, p, t) representation for MuJoCo models. MuJoCo structs
are stored in mjd. Takes in a policy as the controller function which is assumed to be
a FastChain or Chain function, but can be a general function if it matches the signature.
"""
# similar to fwdrwd, but more streamlined, since we pass the observations 
# of the system to the policy. Thus we do our mj_fowards, etc in a different
# order.
function mjsystem(u, p, t, mjd, policy)

    svec = mjd.tstate[Threads.threadid()]
    ovec = mjd.tobs[Threads.threadid()]
    env  = mjd.envs[Threads.threadid()]
    m, d = env.sim.m, env.sim.d

    ushape = inputshape(u)
    copytolyceumstate!(svec, u, t)

    # set state (no warmstart)
    fast_setstate!(env, svec) # calls fwd

    # get obs,
    #     controls,
    #     and apply to dynamics
    getobs!(ovec, env)
    p_out = poloutshape(policyfunc(ovec, t, ushape.aug, p, policy)) # fastchain, flux combined
    #p_out = poloutshape(policyfunc(ovec, t, p, policy)) # fastchain, flux combined
    ctrl = p_out.ctrl
    setaction!(env.sim, ctrl) # calls fwd and propagates d.qacc

    cost = -getreward(svec, ctrl, getobs!(ovec, env), env) # pre-allocate this
    #cost = -runningcost(svec, ctrl, getobs!(ovec, env), env) # pre-allocate this

    #if nq == nv # TODO this stuff for quats!!
    #    du[1:nq] .= ẋ # not semi-implicit
    #else
    #    d.qpos .= 0.0 # clear for storage, integrate vel for position change
    #    MuJoCo.MJCore.mj_integratePos(m, d.qpos, ẋ, 1.0)
    #    du[1:nq] .= d.qpos # in case nq > nv
    #end

    vcat(p_out.aug, d.qvel, d.qacc, cost) # TODO no quats here...
end

# u is the [state; reward] vector
# p is the parameters of the system / policy
# t is time
# mjd contains mjd structures
# policy is the NN policy structure, either FastChain or Chain
@adjoint function mjsystem(u, p, t, mjd, policy)
    svec = mjd.tstate[Threads.threadid()]
    ovec = mjd.tobs[Threads.threadid()]
    env  = mjd.envs[Threads.threadid()]
    m, d = env.sim.m, env.sim.d
    nq, nv, nu, no = Int.( (m.nq, m.nv, m.nu, length(obsspace(env))) )

    ushape = inputshape(u)
    copytolyceumstate!(svec, u, t)

    # set state (no warmstart)
    fast_setstate!(env, svec) # calls fwd

    getobs!(ovec, env) # needs δoδstate
    centerobs = SVector{no, Float64}(ovec)

    # policypullback will get gradients for observations, parameters
    _p, policypullback = Zygote.pullback((o,a,p)->policyfunc(o, t, a, p, policy),
                                         ovec, ushape.aug, p)
    p_out = poloutshape(_p)
    ctrl = p_out.ctrl
    #if any(isnan.(ctrl))
    #    println("u", round.(u, digits=2))
    #    println("s", round.(svec, digits=2))
    #    println("o", round.(ovec, digits=2))
    #    println("p", round.(p, digits=2))
    #    error("Policy output nans")
    #end

    # TODO convert to statespace, etc
    x = ushape.state.pos #u[1:nq]
    ẋ = ushape.state.vel #u[(nq+1):(nq+nv)]

    ẍr = MVector{nv+1, Float64}(undef) # + 1 for reward
    x̂r = MVector{nv+1, Float64}(undef)
    fwdrwd!(ẍr, x, ẋ, ctrl, mjd) # gets [accelerations, reward]

    warm = SVector{nv, Float64}(d.qacc_warmstart)

    # This is the (maybe?) block representation of the jacobian for du/du
    # so we can optimze and take advantage of some sparsity
    #[zeros(nq,nq) Matrix(I,nq,nq) zeros(nq, 1);
    # δaδq         δaδv            zeros(nv+1, 1)]
        
    # δu below corresponds to change in controls
    # δa below corresponds to (accel, reward)
    # l should be (vel, accl, reward), so we may just need (accel, reward)
    function du_vjp(l) # l should be size(du) = (vel, accel, reward)
        #svec = mjd.tstate[Threads.threadid()]
        #ovec = mjd.tobs[Threads.threadid()]
        #env  = mjd.envs[Threads.threadid()]
        #copytolyceumstate!(svec, u, t)
        #fast_setstate!(env, svec) # calls fwd

        lshape = outputshape(l)
        l_accrwd = vcat(lshape.state.acc, lshape.cost) #l[nv+1+naug:end] # just (accel, reward)
        eps = mjd.eps

        # change in acceleration wrt pos, vel, ctrl
        δaδq = MVector{nq, Float64}(undef) # TODO quats?
        δaδv = MVector{nv, Float64}(undef) # TODO preallocate into mjd?
        δaδu = MVector{nu, Float64}(undef) 

        # change in observations wrt pos & vel
        δoδq = MVector{nq, Float64}(undef) # TODO quats?
        δoδv = MVector{nv, Float64}(undef)

        fwdrwd!(x̂r, x, ẋ, ctrl, mjd;  # fixes issues with bptt?
                warm=warm)#, skip=MuJoCo.MJCore.mjSTAGE_VEL)

        # finite difference for controls
        for i=1:nu
            ctrl[i] += eps
            fwdrwd!(x̂r, x, ẋ, ctrl, mjd; 
                    warm=warm, skip=MuJoCo.MJCore.mjSTAGE_VEL)
            δaδu[i] = dot((x̂r - ẍr), l_accrwd) / eps
            ctrl[i] -= eps
        end
        δuδo, δuδaug, δuδp = policypullback(vcat(δaδu, lshape.aug)) # input to policy is obs and params: o, aug, p
        
        # finite difference for velocities
        for i=1:nv
            ẋ[i] += eps
            fwdrwd!(x̂r, x, ẋ, ctrl, mjd;
                    warm=warm, skip=MuJoCo.MJCore.mjSTAGE_POS)
            δaδv[i] = lshape.state.vel[i] + (dot((x̂r - ẍr), l_accrwd) / eps) # velocity contributes to this term
            if δuδo == nothing # for non-parametric policy
                δoδv[i] = 0.0
            else
                getobs!(ovec, env)
                δoδv[i] = (dot(ovec - centerobs, δuδo) / eps)
            end
            ẋ[i] -= eps
        end
        #println(t)
        #println("xhat v ", round.(x̂r; digits=5))
        #println("xddt v ", round.(ẍr; digits=5))

        # finite difference for positions
        for i=1:nq ############ TODO QUAAAAAAAAATS omgggggggg
            x[i] += eps
            fwdrwd!(x̂r, x, ẋ, ctrl, mjd;
                    warm=warm)
            δaδq[i] = dot((x̂r - ẍr), l_accrwd) / eps
            if δuδo == nothing # for non-parametric policy
                δoδv[i] = 0.0
            else
                getobs!(ovec, env)
                δoδq[i] = dot(ovec - centerobs, δuδo) / eps
            end
            x[i] -= eps
        end

        # needs to return x/du, x/dp, x/dt
        dadx = vcat(δaδq, δaδv)
        dodx = vcat(δoδq, δoδv)
       
        #println("l ", round.(l; digits=5))
        #println("dadu ", round.(δaδu; digits=5))
        #println("dudp ", norm(δuδp))
        #println("dadx ", round.(dadx; digits=5))

        #(vcat(_δuδo[end], (dodx + dadx), 0.0), δuδp, 0*l, nothing, nothing) # du, dp, dt
        if δuδaug == nothing
            (vcat((dodx + dadx), 0.0), δuδp, 0.0, nothing, nothing) # du, dp, dt
        else
            (vcat(δuδaug, (dodx + dadx), 0.0), δuδp, 0.0, nothing, nothing) # du, dp, dt
        end
        #(vcat((dodx + dadx), 0.0), δuδp, 0*l, nothing, nothing) # du, dp, dt
        # NOTE dodx doesn't seem to contribute much to learning?
        #(vcat((dadx), 0.0), δuδp, 0*l, nothing, nothing) # du, dp, dt
    end

    vcat(p_out.aug, ẋ, ẍr), du_vjp
end

# ode state u and action a
function odeterminalcost(u, mjd)
    svec = mjd.tstate[Threads.threadid()]
    ovec = mjd.tobs[Threads.threadid()]
    env  = mjd.envs[Threads.threadid()]

    # state from ode to lyceum
    copytolyceumstate!(svec, u)
    fast_setstate!(env, svec) # calls fwd
    getobs!(ovec, env)

    -terminalcost(svec, zeros(length(actionspace(env))), ovec, env)
end

@adjoint function odeterminalcost(u, mjd)
    tcost = odeterminalcost(u, mjd)
    function tcost_pb(l)
        eps = mjd.eps
        nu = length(u)
        δcδu = MVector{nu, Float64}(undef) # TODO quats?
        for i=1:nu
            u[i] += eps
            δcδu[i] = dot((odeterminalcost(u, mjd)-tcost), l) / eps
            u[i] -= eps
        end
        (δcδu, nothing)
    end

    tcost, tcost_pb
end

"""
    oderollout(f, x0, tspan, p)

Rolls out a function f with a keyword supplied solver (Tsit5() default).
f is the f(u, p, t) function for a dynamical system
x0 is the initial state of the system, and tspan runs the system for a window of time.
p is the parameters of f.

Returns the saved states of the rollout.

"""
function oderollout(f, x0, tspan, p;
                    solver=Tsit5(),
                    sensealg=InterpolatingAdjoint(),
                    kwargs...)
    rollout = Array(solve(ODEProblem(f, x0, tspan, p),
                          solver,
                          u0 = x0,
                          p  = p;
                          sensealg=sensealg,
                          maxiters = 4e4, # heuristic to catch bad runs
                          #reltol=1e-4, # reacher
                          #abstol=1e-4, 
                          #reltol=1e-5,
                          #abstol=1e-6,
                          reltol=1e-3,
                          abstol=1e-4,
                          saveend=true,
                          kwargs...
                          ))
end

# broken
function dtrollout(f, x, tspan, p, dt)
    #x = copy(x0)
    #for t=first(tspan):dt:last(tspan)
    #nstep = floor(Int, last(tspan)/dt)
    nstep = Zygote.@ignore collect(first(tspan):dt:last(tspan))
    for t in nstep
        x += dt * f(x, p, t) # something broken with passing in time
    end
    #Zygote.@ignore println(round.(x, digits=3))
    cost = x[end]
    #x
end

"""
    bpttloss(f, p, tspan, mjd, dt)

The same as odeloss, but takes in a dt argument for the dt.
The sensealg keyword points to BacksolveAdjoint() but should be SensitivityADPassThrough(), which currently breaks.

See also: bpttloss
"""
function bpttloss(f, p, tspan, mjd, dt;
                  batchsize=1)
    #odeloss(f, p, tspan, mjd;
    #        batchsize=batchsize,
    #        solver=Euler(),
    #        #sensealg=BacksolveAdjoint(),
    #        sensealg=SensitivityADPassThrough(),
    #        dt=dt
    #       )
    
    b=0.0
    for i=1:batchsize
        newstart = Zygote.@ignore begin
            env = mjd.envs[Threads.threadid()]
            randreset!(env)
            [#zeros(naug);
             fast_getstate(env);
             #-getreward(getstate(env),
             ##-runningcost(getstate(env),
             #           getaction(env),
             #           getobs(env),
             #           env)
             0.0
            ]
        end
        b += dtrollout(f, newstart, tspan, p, dt) #[end]
    end
    b / batchsize
end

"""
    odeloss(f, p, tspan, mjd)

A loss function for rolling out MuJoCo models, based on the mjd struct.
Keyword arguments include solver and sensealg that are passed to the ODE.

See also: bpttloss
"""
function odeloss(f, p, tspan, mjd;
                 batchsize=1,
                 solver=Tsit5(),
                 sensealg=InterpolatingAdjoint(),
                 kwargs...) # change to kwargs...?

    BLAS.set_num_threads(1)
    #starts = Vector{Vector{Float64}}(undef,0)
    #Threads.@sync for i=1:batchsize
    #    Threads.@spawn begin
    #Zygote.@ignore for i=1:batchsize
    #    begin
    #        env = mjd.envs[Threads.threadid()]
    #        randreset!(env)
    #        newstart = [fast_getstate(env); 0.0]
    #        push!(starts, newstart)
    #    end
    #end  

    # TODO change to tmap when threading works again...
    #b = map(x->oderollout(f, x, tspan, p;
    #                      solver=solver,
    #                      sensealg=sensealg,
    #                      kwargs...)[end,end],
    #        starts)
    #mean(b)
    #b = 0.0 
    #for s in starts
    #    b += oderollout(f, s, tspan, p;
    #                    solver=solver,
    #                    sensealg=sensealg,
    #                    kwargs...)[end,end]
    #end
    #b / batchsize

    #b = Zygote.bufferfrom(zeros(batchsize))
    b = 0.0
    #Threads.@sync for i=1:batchsize
    #Threads.@spawn begin
    for i=1:batchsize
        begin
            env = Zygote.@ignore mjd.envs[Threads.threadid()]
            Zygote.@ignore randreset!(env)
            #newstart = [fast_getstate(env); 0.0]
            newstart = Zygote.@ignore [#zeros(naug);
                                       fast_getstate(env);
                                       #-getreward(getstate(env),
                                       ##-runningcost(getstate(env),
                                       #           getaction(env),
                                       #           getobs(env),
                                       #           env)
                                       0.0
                                      ]
            cost = oderollout(f, newstart, tspan, p;
                              solver=solver,
                              sensealg=sensealg,
                              kwargs...)[end,end]
            #b[i] = cost
            b += cost
        end
    end
    #sum(copy(b))/batchsize #+ 0.5*norm(p) # divide for our reference
    b/batchsize
    #mean(copy(b))
end

function mpcloss(f, p, tspan, mjd;
                 batchsize=1,
                 solver=Tsit5(),
                 sensealg=InterpolatingAdjoint(),
                 kwargs...) # change to kwargs...?

    BLAS.set_num_threads(1)
    b = Zygote.bufferfrom(zeros(batchsize))
    Threads.@sync for i=1:batchsize
    Threads.@spawn begin
    #for i=1:batchsize
    #    begin
            env = Zygote.@ignore mjd.envs[Threads.threadid()]
            Zygote.@ignore randreset!(env)
            #newstart = [fast_getstate(env); 0.0]
            newstart = Zygote.@ignore [zeros(naug);
                                       fast_getstate(env);
                                       #-getreward(getstate(env),
                                       ##-runningcost(getstate(env),
                                       #           getaction(env),
                                       #           getobs(env),
                                       #           env)
                                       0.0
                                      ]
            newstart = Zygote.@ignore oderollout(f, newstart, (0, rand(Uniform(tspan...))), p;
                                                 solver=solver,
                                                 sensealg=sensealg,
                                                 kwargs...)[:,end]
            Zygote.@ignore newstart[end] = 0
            #Zygote.@ignore println(newstart)
            cost = oderollout(f, newstart, tspan, p;
                              solver=solver,
                              sensealg=sensealg,
                              kwargs...)[end,end]
            b[i] = cost
            #b += cost
        end
    end
    sum(copy(b))/batchsize #+ 0.5*norm(p) # divide for our reference
    #b/batchsize
    #mean(copy(b))
end

function tcostodeloss(f, p, tspan, mjd;
                      batchsize=1,
                      solver=Tsit5(),
                      sensealg=InterpolatingAdjoint(),
                      kwargs...)

    BLAS.set_num_threads(1)
    b = Zygote.bufferfrom(zeros(batchsize))
    Threads.@sync for i=1:batchsize # TODO threading is currently broken with zygote??!
    Threads.@spawn begin
    #for i=1:batchsize
    #    begin
            env = Zygote.@ignore mjd.envs[Threads.threadid()]
            Zygote.@ignore randreset!(env)
            newstart = Zygote.@ignore [zeros(naug);
                                       fast_getstate(env);
                                       -getreward(getstate(env),
                                       #-runningcost(getstate(env),
                                                  getaction(env),
                                                  getobs(env),
                                                  env)]
            rollout = oderollout(f, newstart, tspan, p;
                              solver=solver,
                              sensealg=sensealg,
                              kwargs...)
            cost = rollout[end,end] + odeterminalcost(rollout[:,end], mjd)#, last(tspan))
            b[i] = cost
            #b += cost
        end
    end
    sum(copy(b))/batchsize #+ 0.5*norm(p) # divide for our reference
    #b/batchsize
    #mean(copy(b))
end

"""
    valueloss(f, p, tspan, mjd)

A loss function for rolling out MuJoCo models, based on the mjd struct.
Keyword arguments include solver and sensealg that are passed to the ODE.

See also: bpttloss
"""
function valueloss(f, p, tspan, mjd, valuef;
                 batchsize=1,
                 solver=Tsit5(),
                 sensealg=InterpolatingAdjoint(),
                 kwargs...) # change to kwargs...?

    BLAS.set_num_threads(1)

    b = Zygote.bufferfrom(zeros(batchsize))
    Threads.@sync for i=1:batchsize
    Threads.@spawn begin
    #for i=1:batchsize
    #    begin
            env = Zygote.@ignore mjd.envs[Threads.threadid()]
            Zygote.@ignore randreset!(env)
            x0 = Zygote.@ignore fast_getstate(env);
            newstart = Zygote.@ignore [zeros(naug);
                                       x0;
                                       #-valuef(x0)[1];
                                       0.0
                                      ]
            rollout = oderollout(f, newstart, tspan, p;
                                 solver=solver,
                                 sensealg=sensealg,
                                 kwargs...)
            cost = rollout[end,end] + valuef(rollout[1:end-1,end])[1] # TODO no augment
            b[i] = cost
            #b += cost
        end
    end
    sum(copy(b))/batchsize #+ 0.5*norm(p) # divide for our reference
    #b/batchsize
    #mean(copy(b))
end
"""
    refloss(f, p, tspan, mjd, timesanddata)

Loss function for matching the policy to data, presumably from trajectory optimizers.

timesanddata should be zip(times, data) where times, data are arrays AND includes rewards, althought it is not used for loss.
"""
function refloss(f, p, tspan, mjd, timesanddata;
                   solver=Tsit5(),
                   minibatch=0, #false,
                   sensealg=InterpolatingAdjoint(),
                   kwargs...) # change to kwargs...?
    #BLAS.set_num_threads(1)
    
    function matchdata(tsteps, data)
        # TODO was getting different time windows from tspan
        N = length(tsteps)
        #tspan = Zygote.@ignore (max(tspan[1],tsteps[1]), min(tspan[2], tsteps[end]))
        #_, datai = Zygote.@ignore findmin((tsteps .- tspan[1]).^2) # find closest timestep to our span
        datai = 1
        #tspan = Zygote.@ignore (tsteps[datai], tspan[2])
        #tsteps = Zygote.@ignore filter(x->x>=tspan[1] && x<=tspan[2], tsteps)
        tspan = (tsteps[1], tsteps[end])
        newstart = data[:, datai]
        #println(tspan)
        #println(datai)
        #println(tsteps)
        rollout = oderollout(f, newstart, tspan, p;
                             solver=solver,
                             sensealg=sensealg,
                             saveat=tsteps,
                             kwargs...)
        qposqvel = rollout[1:end-1, :]  # trim off reward, works better

        i = Zygote.@ignore datai:(datai+length(tsteps)-1)
        #sum(abs2, qposqvel .- data[1:end-1, i])
        #mean(abs2, qposqvel .- data[1:end-1, i])
        #sum(exp, sum(abs2, qposqvel .- data[1:end-1, i]; dims=1))
        s = Zygote.@ignore log(10)/tspan[2]
        sum(exp.(s .* sum(abs2, qposqvel .- data[1:end-1, i]; dims=1)))
        #sum(norm.(qposqvel - data[1:end-1, i]))
        #sum(abs2, rollout .- data[:,:]) # with reward, but something's not good
    end

    #cost = sum(map(x->matchdata(x...), timesanddata)) # so we can tmap later
    #sqrt(cost)
    b = 0.0
    N = length(timesanddata)
    
    Threads.@sync for td in timesanddata # something buggy about this causes hangs
    #Threads.@sync for i=1:N # something buggy about this causes hangs
        Threads.@spawn begin
            #for td in timesanddata
            #    begin
            t, d = td #timesanddata[i]         
            b += matchdata(t, d)
        end
    end
    #sqrt(b)
    b / length(timesanddata)

end

function batchcollocation(times, datas)
    @assert length(times) == length(datas)
    dus = Vector{Matrix{Float64}}(undef,0)
    us  = Vector{Matrix{Float64}}(undef,0)
    for i=1:length(times)
        d = datas[i]
        t = times[i]
        du, u = collocate_data(d, t, GaussianKernel())
        push!(dus, du)
        push!(us, u)
    end
    dus, us
end

"""
    colloloss(f, p, du_and_u)

With dynamics function f and params p, calculate loss wrt collocated data (u', u, time).

Currently does not include the reward/cost in du and u, and assumes fixed dt between
data points.

Something seems broken; gradients blow up when in for loop. I suspect that there is some
Zygote optimization that re-uses the pullback of f, but since it's the custom one for
mujocosystem (which has state) it leads to weird numbers.

Not sure what loss function to use. https://diffeqflux.sciml.ai/dev/examples/collocation/
does a norm.
"""
function colloloss(f, p, du, u, dt) #du_u_t) # TODO could add time here?
    cost = 0.0 #zero(first(p))
    #l = size(du,2) #-1 # and thus ignore the last time
    #for i in 1:l          # TODO track time?
    u_input = u #@view u[:,i] #[u[:,i]; 0.0] # tack on empty cost for f
    du_real = du #@view du[:,i] #* dt #s[i]
    du_test = f(u_input, p, 0.0) # strip off cost
    #cost += Flux.mse(du_real.*dt, du_test) #sum(abs2, du_real .* dt .- du_test)
    cost = sum(abs2, (du_real .* dt) .- du_test)
    #b += cost
    #println(cost)
    #end
    sqrt(cost)
    #cost
end

function collo(f, p, du, u, t)
end

"""
gradcheck(loss, p, N, iter=3)

Utility function to check loss function gradient "goodness" wrt batchsize.
Input to the loss function should be p:

loss(p, n) = odeloss(f, p, tspan, mjd; batchsize=n)

"goodness" is the norm of the gradient, and is checked wrt the initial start state

"""
function gradcheck(loss, p, N, iter=3)
    times = zeros(N)
    stdtimes = zeros(N)
    grads = zeros(N)
    stdgrads = zeros(N)
    @elapsed testloss = Zygote.gradient(p->loss(p, 1), p)
    for i=1:N
        t = zeros(iter)
        g = zeros(iter)
        for j=1:iter
            t[j] = @elapsed testloss = Zygote.gradient(p->loss(p, i), p)
            g[j] = norm(testloss)
        end
        times[i] = mean(t)
        stdtimes[i] = std(t)
        grads[i] = mean(g)
        stdgrads[i] = std(g)
        println("threads: $i, gradnorm: $(round(grads[i], digits=3)),\ttime: $(times[i])")
    end
    plt = lineplot(times, height=7, title="proc time vs batchsize, mean & std",
                   color=:blue, xlim=(1,N))
    lineplot!(plt, times + stdtimes, color=:red)
    lineplot!(plt, times - stdtimes, color=:red)
    display(plt)
    plt = lineplot(grads, height=7, title="grad norm vs batchsize, mean & std",
                   color=:blue, xlim=(1,N))
    lineplot!(plt, grads + stdgrads, color=:red)
    lineplot!(plt, grads - stdgrads, color=:red)
    display(plt)
end


function testsolvers(env, tspan, p0, policy)
    choice_function(integrator) = (Int(integrator.dt<0.0001) + 1)
    solvers = [
               Euler(),
               Midpoint(),
               Heun(),
               Tsit5(),
               VCAB4(),
               VCABM4(),
               VCABM(),
               TRBDF2(autodiff=false,diff_type=Val{:forward}),
               #Rosenbrock23(autodiff=false,diff_type=Val{:forward}),
               #Rodas3(autodiff=false,diff_type=Val{:forward}),
               #Rodas4P(autodiff=false,diff_type=Val{:forward}),
               Rodas5(autodiff=false,diff_type=Val{:forward}),
               AB4(),
               ABM43(),
               #ImplicitEuler()#autodiff=false,diff_type=Val{:forward})
               #CompositeAlgorithm((Tsit5(),
               #                    Rosenbrock23(autodiff=false,diff_type=Val{:forward})),
               #                   choice_function),
               CompositeAlgorithm((Tsit5(),
                                   Rodas5(autodiff=false,diff_type=Val{:forward})),
                                  choice_function),
               CompositeAlgorithm((Tsit5(),Euler()),choice_function)
              ]
    #tsidas_alg = AutoTsit5(Rodas5())
    #using LSODA
    #lsoda()

    randreset!(env)
    u0 = [fast_getstate(env); 0.0]
    inc = 0
    rtimes = [ @elapsed while time(env) <= last(tspan)
                  setaction!(env, policyfunc(getobs(env), time(env), p0, policy))
                  LyceumBase.step!(env)
                  inc += 1
              end ]
    evals = [ inc ]
    norms = [0.0]
    uT = fast_getstate(env)
    #println(uT)

    for solver in solvers
        println(nameof(typeof(solver)))
        try
            t = @elapsed oderollout(f, u0, (0, 0.1), p0;
                                    solver=solver, dt=dt) # warmup
            t = @elapsed rollout = oderollout(f, u0, tspan, p0;
                                              solver=solver, dt=dt)
            push!(rtimes, t)
            push!(evals, size(rollout, 2))
            push!(norms, sum(abs2, rollout[1:end-1,end] .- uT))
        catch
            #printstyled("Failed: $(nameof(typeof(solver)))\n"; color=:red)
            push!(rtimes, 1e10)
            push!(evals, 0)
            push!(norms, 1e10)
        end
    end
    println()
    println(nameof(typeof(env)))
    h1 = Highlighter( (data,i,j)->j in (2,3) && data[i,j] == minimum(data[2:end,2]),
                     foreground = :red )
    h2 = Highlighter( (data,i,j)->j in (2,3) && data[i,j] == minimum(data[2:end,3]),
                     foreground = :blue )
    pretty_table(hcat(["MuJoCo/Default"; nameof.(typeof.(solvers))], rtimes, norms, evals),
                 ["Integrator", "Rollout Time", "Error (from Default)", "Function Evals"], highlighters=(h1,h2))


end

"""
mj_callback(p, loss_val, mjd, losses, T, niter=100; skipiter=5, N=3, doplot=false)

This callback function does rollouts with the policy to get reward and evaluations
of the environment to understand the progress of the policy.
"""
function mj_callback(p, loss_val, mjd, losses, T, niter=100;
    skipiter=5, N=3, doplot=false)

    push!(losses, loss_val)

    skip = niter > skipiter ? div(niter, skipiter) : niter

    dt = timestep(mjd.envs[1])

    if doplot || mod1(length(losses), skip) == skip
        tspan = (0, T)
        tsteps = 0.0:dt:(T)
        evals = zeros(length(tsteps), N)
        ctrls = zeros(length(tsteps), N)
        costs = []
        for m=1:N
            env = mjd.envs[Threads.threadid()]
            svec = mjd.tstate[Threads.threadid()]
            ovec = mjd.tobs[Threads.threadid()]

            randreset!(env)
            #newstart = [fast_getstate(env); 0.0]
            newstart = zeros(length(inputshape))
            inputshape(newstart).state .= fast_getstate(env)
            inputshape(newstart).cost = -getreward(getstate(env),
            #inputshape(newstart).cost = -runningcost(getstate(env),
                                                   getaction(env),
                                                   getobs(env),
                                                   env)
            s = solve(ODEProblem(f, newstart, tspan, p),
                      solver, # TODO global
                      u0 = newstart,
                      p  = p;
                      maxiters = maxiters,
                      solveargs...,
                      saveat=tsteps)
            times = s.t
            rollout = Array(s)

            push!(costs, rollout[end,:])

            dx  = zeros(size(rollout,1))
            for i=1:size(rollout,2)
                r = rollout[:,i]

                copytolyceumstate!(svec, r)
                fast_setstate!(env, svec)
                getobs!(ovec, env)
                ctrl = poloutshape(policyfunc(ovec, times[i], zeros(naug), p, policy)).ctrl
                setaction!(env, ctrl)

                ctrls[i, m] = ctrl[1]
                evals[i, m] = geteval(svec, ctrl, ovec, env)
            end
        end
        plt = lineplot(tsteps, evals[:,1], title="evals", width=60, height=6)
        cplt = lineplot(tsteps, costs[1], title="costs", width=60, height=6)
        ctpl = lineplot(tsteps, ctrls[:,1], title="ctrls", width=60, height=6)
        for i=2:N
            lineplot!(plt, tsteps, evals[:,i])
            lineplot!(cplt, tsteps, costs[i])
        end
        display(cplt)
        display(ctpl)
        display(plt)
        display(lineplot(losses, title="loss", xlim=(1,length(losses)), height=7))
    end
    return false
end

"""
"""
function ref_callback(p, loss_val, mjd, losses, timesanddata, niter=100;
    skipiter=5, N=length(timesanddata), 
    ndim=2,
    doplot=false)

    push!(losses, loss_val)

    skip = niter > skipiter ? div(niter, skipiter) : niter

    dt = timestep(mjd.envs[1])

    if doplot || mod1(length(losses), skip) == skip
        idx = 0
        for (tsteps, data) in timesanddata
            index = rand(1:length(timesanddata))

            env = mjd.envs[Threads.threadid()]
            svec = mjd.tstate[Threads.threadid()]
            ovec = mjd.tobs[Threads.threadid()]

            newstart = data[:,1]
            tspan = (tsteps[1], tsteps[end])
            s = solve(ODEProblem(f, newstart, tspan, p),
                      solver, # TODO global
                      u0 = newstart,
                      p  = p;
                      maxiters = maxiters,
                      solveargs...,
                      saveat=tsteps)
            rollout = Array(s)

            ylim = round.(extrema(data[1:ndim,:]))
            plt = lineplot(tsteps, data[1,:], title="data $idx",
                           width=60, height=6, ylim=ylim, color=:blue)
            lineplot!(plt, tsteps, rollout[1,:], color=:red, name="data")
            for d=2:ndim
                lineplot!(plt, tsteps, data[d,:], color=:blue)
                lineplot!(plt, tsteps, rollout[d,:], color=:red)
            end
            display(plt)

            plt = lineplot(tsteps, data[end,:], title="costs $idx (broken plot?)", width=60, height=6, color=:blue, name="ref cost")
            lineplot!(plt, tsteps[1:end-1], diff(rollout[end,:]).+rollout[end,1], color=:red, name="rollout")
            display(plt)
            idx += 1
            if idx > N
                break
            end
        end
        display(lineplot(losses, title="loss", xlim=(1,length(losses)), height=7))
    end
    return false
end

# TODO take in the env??
"""
mj2ode(rollout::Vector{Matrix}, rewards::Vector{Vector{Float64}}, nqnv::Integer)

Take a vector of MuJoCo state sequences matrices, and slice and dice to get the 
time steps and [qvel qvel x 1:T] matrix that ODEProblem produces.
"""
function mj2ode(rollouts::Vector{Matrix{Float64}},
                rewards::Vector{Vector{Float64}}, nqnv::Integer)
    @assert length(rollouts) == length(rewards)
    times = [ roll[1,:] for roll in rollouts ]
    qposqvelrwd = [ vcat(rollouts[i][2:nqnv+1,:], -rewards[i]') for i=1:length(rewards) ] 
    times, qposqvelrwd
end

function ode2mj(rollout::AbstractMatrix, ts::AbstractVector,
                policyf, env)
    N = length(ts)
    traj = (
            states = zeros(statespace(env), N),
            obses  = zeros(obsspace(env), N),
            acts   = zeros(actionspace(env), N),
            rews   = zeros(rewardspace(env), N),
            evals  = zeros(evalspace(env), N)
           )
    for t=1:N
        r = @view rollout[:,t]
        st = view(traj.states, :, t)
        at = view(traj.acts, :, t)
        ot = view(traj.obses, :, t)

        copytolyceumstate!(st, r, ts[t])
        fast_setstate!(env, st)
        getobs!(ot, env)

        at .= poloutshape(policyf(ot, (t-1)*timestep(env), inputshape(r).aug)).ctrl
        setaction!(env, at)

        traj.rews[t] = -getreward(st, at, ot, env)
        #traj.rews[t] = -runningcost(st, at, ot, env)
        #if t == N
        #    traj.rews[t] = -terminalcost(st, at, ot, env)
        #end
        traj.evals[t] = geteval(st, at, ot, env)
    end
    traj
end

using JLSO
function saverollouts(file, f, mjd, solver, policy, polparams, T, N)
    trajs = Vector{NamedTuple}(undef, N)
    rolls = Vector{Matrix}(undef, N)
    ts    = Vector{Vector}(undef, N)
    Threads.@sync for i=1:N
        Threads.@spawn begin
            env = mjd.envs[Threads.threadid()]
            tspan = 0.0:timestep(env):T
            if i==1 println("num timesteps: $(length(tspan))") end
            randreset!(env)
            newstart = zeros(length(inputshape))
            inputshape(newstart).state .= fast_getstate(env)
            inputshape(newstart).cost = -getreward(getstate(env),
            #inputshape(newstart).cost = -runningcost(getstate(env),
                                                   getaction(env),
                                                   getobs(env),
                                                   env)
            rollout = solve(ODEProblem(f, newstart, (0.0,T), polparams),
                            solver,
                            u0 = newstart,
                            p  = polparams;
                            solveargs...,
                            saveat=tspan)
            ts[i] = rollout.t
            rolls[i] = Array(rollout)
        end
    end

    Threads.@sync for (i,roll) in enumerate(rolls)
        Threads.@spawn trajs[i] = ode2mj(roll, ts[i],
                                         (o,t,a)->policyfunc(o, t, a, polparams, policy),
                                         mjd.envs[Threads.threadid()])
    end

    JLSO.save(file,
              :traj => getproperty.(trajs, :states),
              :policy => policy,
              :polparams => polparams,
              compression=:none)
    return rolls, trajs
end

function loadrollout(f)
    traj = JLSO.load(f)[:traj]
end

using LyceumAI
function getdemos(envt, N, T, H=128, K=32, sigma=0.8, lambda=0.2)
    env = envt()
    mppi = MPPI(
                env_tconstructor = n -> tconstruct(envt, n),
                covar = Diagonal(sigma^2 * I, size(actionspace(env), 1)),
                lambda = lambda,
                H = H,
                K = K,
                gamma = 1.,
                #clamps = ctrlrange # TODO might need later?
               )
    obs    = Matrix{Float64}[]
    act    = Matrix{Float64}[]
    states = Matrix{Float64}[]
    rewards = Vector{Float64}[]

    for i=1:N
        reset!(mppi)
        randreset!(env)
        opt = ControllerIterator((a, s, o) -> getaction!(a, s, mppi), env;
                                 T = T, plotiter = T)
        @time for _ in opt # runs iterator
    end
    traj = opt.trajectory 
    push!(obs, traj.observations) # matrix
    push!(act, traj.actions)
    push!(states, traj.states)
    push!(rewards, traj.rewards)
end
JLSO.save("/tmp/ode_mppi.jlso",
          :traj => states,
          :policy => policy,
          :polparams => zeros(1),
          compression=:none)
#reduce(hcat, states), reduce(hcat, obs), reduce(hcat, act)
states, obs, act, rewards
                     end


                     #=
                     @with_kw mutable struct Args
                     lr::Float64 = 1e-2	# Learning rate
                     seqlen::Int = 50	# Length of batchseqences
                     nbatch::Int = 50	# number of batches text is divided into
                     throttle::Int = 30	# Throttle timeout
                     end
                     =#

                     # To visualize
                     #using LyceumMuJoCoViz
                     #visualize(env, trajectories=loadrollout(f))

