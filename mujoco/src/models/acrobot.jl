using LyceumMuJoCo, MuJoCo
using Base: @propagate_inbounds
using UnsafeArrays
using Random
using Shapes
using LinearAlgebra

include("reward.jl")

#------------------Constructing Environment-------------------#
struct Acrobot{Sim, OSpace} <: AbstractMuJoCoEnvironment
    sim::Sim
    obsspace::OSpace

    tip::Int

    function Acrobot(sim::MJSim)
        m = sim.m
        ospace = MultiShape(
                            horizontal = VectorShape(Float64, 2),
                            vertical   = VectorShape(Float64, 2),
                            sinqpos       = VectorShape(Float64, m.nq),
                            cosqpos       = VectorShape(Float64, m.nq),
                            qvel       = VectorShape(Float64, m.nv),
                           )
        tip = jl_name2id(m, MuJoCo.MJCore.mjOBJ_SITE, "tip")
        trg = jl_name2id(m, MuJoCo.MJCore.mjOBJ_SITE, "target")
        new{typeof(sim), typeof(ospace)}(
            sim,
            ospace,
            tip,
        )
    end
end

function LyceumBase.tconstruct(::Type{Acrobot}, n::Integer)
    modelpath = joinpath(@__DIR__, "dm_acrobot.xml")
    Tuple(Acrobot(s) for s in tconstruct(MJSim, n, modelpath, skip=1))
end
Acrobot() = first(tconstruct(Acrobot, 1))

@inline LyceumMuJoCo.getsim(env::Acrobot) = env.sim
@inline LyceumMuJoCo.obsspace(env::Acrobot) = env.obsspace

@propagate_inbounds function LyceumMuJoCo.getobs!(obs, env::Acrobot)
    checkaxes(obsspace(env), obs)
    d    = env.sim.d 
    xmat = env.sim.dn.xmat
    qvel = d.qvel
    qpos = d.qpos

    h1 = xmat[:x, :z, :upper_arm]
    h2 = xmat[:x, :z, :lower_arm]
    v1 = xmat[:z, :z, :upper_arm]
    v2 = xmat[:z, :z, :lower_arm]
    scale = 1e-8

    @views @uviews obs qvel qpos begin
        o = obsspace(env)(obs)
        o.horizontal[1] = h1   #+ randn()*scale
        o.horizontal[2] = h2   #+ randn()*scale
        o.vertical[1]   = v1   #+ randn()*scale
        o.vertical[2]   = v2   #+ randn()*scale
        o.sinqpos         .= sin.(qpos)
        o.cosqpos         .= cos.(qpos)
        o.qvel         .= qvel #.+ randn(length(qvel)).*scale
    end
    obs
end

@propagate_inbounds function LyceumMuJoCo.getreward(state, action, obs, env::Acrobot)
    d = env.sim.d
    osp = obsspace(env)(obs)

    #reward = h1+h2 - 0.01*norm(action)^2
    #tip = obsspace(env)(obs).dist

    #radius = env.sim.mn.site_size[1, :target]

    to_target = norm(SPoint3D(d.site_xpos, 1) - SPoint3D(d.site_xpos, 2))

    #reward = tolerance(to_target, bounds=(0, radius), margin=4) # margin = 0 for sparse

    reward = -(to_target^2) - 0.01*norm(d.qvel)^2

    #exp(log(10)/2.0 * reward) # goal at T
    #exp(log(2)/2.0 * reward)
    #exp(log(10)/1.0 * time(env)) * (reward)
    #reward = -tip - 0.01*norm(d.qvel)^2 #+ h1 + h2
    #reward
    #1.001^(time(env)/timestep(env)) * reward # doesn't overshoot; something something value is critically damped?
    0.99^(time(env)/timestep(env)) * reward # doesn't overshoot; something something value is critically damped?
end

function runningcost(state, action, obs, env::Acrobot)
    d = env.sim.d
    - 0.01*norm(d.qvel)^2
end
function terminalcost(state, action, obs, env::Acrobot)
    d = env.sim.d
    to_target = norm(SPoint3D(d.site_xpos, 1) - SPoint3D(d.site_xpos, 2))
    -(to_target^2)
end


@propagate_inbounds function LyceumMuJoCo.geteval(state, action, obs, env::Acrobot)
    #obsspace(env)(obs).dist
    d = env.sim.d
    to_target = norm(SPoint3D(d.site_xpos, 1) - SPoint3D(d.site_xpos, 2))
end

@propagate_inbounds function LyceumMuJoCo.reset!(env::Acrobot)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    forward!(env.sim)
    env
end

@propagate_inbounds function LyceumMuJoCo.randreset!(rng::AbstractRNG, env::Acrobot)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    d = env.sim.d
    #mag = 1.1
    mag = pi
    perturb!(rng, Uniform(-mag, mag), d.qpos)
    perturb!(rng, Uniform(-5, 5), d.qpos) # also have to perturb velocity
    #d.qvel[1:2] .= rand(rng, Uniform(-0.005, 0.005), 2)
    #d.qvel[1:2] .= rand(rng, Uniform(-0.8, 0.8), 2)
    #env.sim.d.qpos[1] += pi
    forward!(env.sim)
    #reset!(env) # reset to bottom
    #d.qpos[1] = 3.14
    env
end

function LyceumMuJoCo.isdone(state, ::Any, env::Acrobot)
    return false
end

