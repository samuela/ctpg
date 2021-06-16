using LyceumMuJoCo
using LyceumBase.Tools
using UnsafeArrays
using Random
using Shapes
using Distributions, LinearAlgebra
using Base: @propagate_inbounds

#------------------Constructing Environment-------------------#
struct Reacher{Sim, OSpace} <: AbstractMuJoCoEnvironment
    sim::Sim
    obsspace::OSpace
    randreset_distribution::Uniform{Float64}

    finger::Int
    target::Int

    function Reacher(sim::MJSim)
        m = sim.m
        ospace = MultiShape(
                            #theta_cos = VectorShape(Float64, 2),
                            #theta_sin = VectorShape(Float64, 2),
                            qpos      = VectorShape(Float64, m.nq-2),
                            qvel      = VectorShape(Float64, m.nv-2),
                            dist      = VectorShape(Float64, 3),
                            #dist      = ScalarShape(Float64),
                           )
        finger = jl_name2id(m, MuJoCo.MJCore.mjOBJ_GEOM, "finger")
        target = jl_name2id(m, MuJoCo.MJCore.mjOBJ_GEOM, "target")
        new{typeof(sim), typeof(ospace)}(
            sim,
            ospace,
            Uniform(-0.1, 0.1),
            finger,
            target
        )
    end
end

function LyceumBase.tconstruct(::Type{Reacher}, n::Integer)
    modelpath = joinpath(@__DIR__, "reacher.xml")
    Tuple(Reacher(s) for s in tconstruct(MJSim, n, modelpath, skip=1))
end
Reacher() = first(tconstruct(Reacher, 1))

@inline LyceumMuJoCo.getsim(env::Reacher) = env.sim
@inline LyceumMuJoCo.obsspace(env::Reacher) = env.obsspace

@propagate_inbounds function LyceumMuJoCo.getobs!(obs, env::Reacher)
    checkaxes(obsspace(env), obs)
    d    = env.sim.d 
    gx   = d.geom_xpos
    qpos = d.qpos
    qvel = d.qvel
    finger = SPoint3D(gx, env.finger)
    target = SPoint3D(gx, env.target)
    @views @uviews obs qpos qvel begin
        o = obsspace(env)(obs)
        #o.theta_cos .= cos.(qpos[1:2])
        #o.theta_sin .= sin.(qpos[1:2])
        o.qpos      .= qpos[1:2] #[3:4]
        o.qvel      .= qvel[1:2] #[1:2]
        o.dist      .= target - finger
        #o.dist       = norm(finger - target)
    end
    obs
end

@propagate_inbounds function LyceumMuJoCo.getreward(state, action, obs, env::Reacher)
    d = env.sim.d 
    gx = d.geom_xpos

    finger = SPoint3D(gx, env.finger)
    target = SPoint3D(gx, env.target)
    #centerT = Tolerance{Float64}(margin=2) # dm_control wants sparse reward

    dist = norm(finger - target)#^2
    reward = -dist #- 0.001*norm(action)^2
    #reward * 1.9^(10time(env))
    0.999^(time(env)/timestep(env)) * reward # doesn't overshoot; something something value is critically damped?
    #1.001^(time(env)/timestep(env)) * reward # doesn't overshoot; something something value is critically damped?
    #reward
end

function runningcost(state, action, obs, env::Reacher)
    #-1e3 * norm(action)^2
    0
end
function terminalcost(state, action, obs, env::Reacher)
    getreward(state, action, obs, env)
end

@propagate_inbounds function LyceumMuJoCo.geteval(state, action, obs, env::Reacher)
    norm(obsspace(env)(obs).dist)
end

@propagate_inbounds function LyceumMuJoCo.reset!(env::Reacher)
    reset!(env.sim)
    env
end

@propagate_inbounds function LyceumMuJoCo.randreset!(rng::AbstractRNG, env::Reacher)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    d = env.sim.d
    mn = env.sim.mn
    
    d.qpos[1] = rand(rng, Uniform(-pi, pi))
    d.qpos[2] = rand(rng, Uniform(-2.79253, 2.79253))

    angle = rand(rng, Uniform(0, 2pi))
    radius = rand(rng, Uniform(0.05, 0.2))
    #mn.geom_pos[1,:target] = radius * sin(angle) # fucking breaks things
    #mn.geom_pos[2,:target] = radius * cos(angle)
    d.qpos[3] = radius*sin(angle)
    d.qpos[4] = radius*cos(angle)
    #d.qpos[3] = -0.1
    #d.qpos[4] = -0.1

    forward!(env.sim)
    env
end

function LyceumMuJoCo.isdone(state, ::Any, env::Reacher)
    return false
end

