using LyceumMuJoCo, MuJoCo
using Base: @propagate_inbounds
using UnsafeArrays
using Random
using Shapes
using LinearAlgebra

include("reward.jl")

#------------------Constructing Environment-------------------#
struct PlanarWalker{Sim, OSpace} <: AbstractMuJoCoEnvironment
    sim::Sim
    obsspace::OSpace

    move_speed::Float64

    function PlanarWalker(sim::MJSim, move_speed=1.0)
        m = sim.m
        ospace = MultiShape(
                            orientations = VectorShape(Float64, 2*(m.nbody-1)),
                            velocity = VectorShape(Float64, m.nv),
                            height = ScalarShape(Float64),
                           )
        new{typeof(sim), typeof(ospace)}(
            sim,
            ospace,
            move_speed
        )
    end
end

function LyceumBase.tconstruct(::Type{PlanarWalker}, n::Integer)
    modelpath = joinpath(@__DIR__, "walker.xml")
    Tuple(PlanarWalker(s) for s in tconstruct(MJSim, n, modelpath, skip=1))
end
PlanarWalker() = first(tconstruct(PlanarWalker, 1))

@inline LyceumMuJoCo.getsim(env::PlanarWalker) = env.sim
@inline LyceumMuJoCo.obsspace(env::PlanarWalker) = env.obsspace

@propagate_inbounds function LyceumMuJoCo.getobs!(obs, env::PlanarWalker)
    checkaxes(obsspace(env), obs)
    d, dn    = env.sim.d, env.sim.dn
    xmat = env.sim.d.xmat
    qvel = d.qvel

    h = dn.xpos[:z, :torso]

    @views @uviews obs qvel xmat begin
        o = obsspace(env)(obs)
        #o.orientations  .= vec(view(xmat, [1,3], 2:env.sim.m.nbody)) # just 'xx', 'xz' ?
        for i=1:2:14
            idx = div(i, 2) + 2
            o.orientations[i]   = xmat[1,idx]
            o.orientations[i+1] = xmat[3,idx]
        end
        o.height         = h
        o.velocity      .= qvel
    end
    obs
end

@propagate_inbounds function LyceumMuJoCo.getreward(state, action, obs, env::PlanarWalker)
    d, dn = env.sim.d, env.sim.dn
    osp = obsspace(env)(obs)

    standT = Tolerance{Float64}(bounds=(1.2, typemax(Float64)),
                                margin=1.2/2.0)
    moveT = Tolerance{Float64}(bounds=(env.move_speed, typemax(Float64)),
                               margin=env.move_speed/2.0,
                               value_at_margin=0.5,
                               sigmoid=linear)

    upright = (1.0 + dn.xmat[:z, :z, :torso]) / 2.0
    stand_reward = (3 * standT(osp.height) + upright) / 4.0

    if env.move_speed > 0.0
        reward = stand_reward * (5.0 * moveT(d.sensordata[1]) + 1.0) / 6.0
    else
        reward = stand_reward
    end
    reward #* 1.9^(time(env))
    #norm(action)^2
end

function runningcost(state, action, obs, env::PlanarWalker)
    norm(action)^2
end
function terminalcost(state, action, obs, env::PlanarWalker)
    getreward(state, action, obs, env)
end

@propagate_inbounds function LyceumMuJoCo.geteval(state, action, obs, env::PlanarWalker)
    obsspace(env)(obs).height
    #d = env.sim.d
    #to_target = norm(SPoint3D(d.site_xpos, 1) - SPoint3D(d.site_xpos, 2))
end

@propagate_inbounds function LyceumMuJoCo.reset!(env::PlanarWalker)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    forward!(env.sim)
    env
end

@propagate_inbounds function LyceumMuJoCo.randreset!(rng::AbstractRNG, env::PlanarWalker)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    d = env.sim.d
    mag = 0.05 #1.1
    #mag = pi
    perturb!(rng, Uniform(-mag, mag), d.qpos) # TODO not correct; approx for now
    forward!(env.sim)

    #reset!(env)
    env
end

function LyceumMuJoCo.isdone(state, ::Any, env::PlanarWalker)
    return false
end

