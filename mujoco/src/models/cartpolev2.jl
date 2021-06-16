using LyceumMuJoCo
using UnsafeArrays
using Random
using Shapes
using Statistics
#------------------Constructing Environment-------------------#

include("reward.jl")

struct CartpoleSwingupV2{S<:MJSim,O<:MultiShape} <: AbstractMuJoCoEnvironment
    sim::S
    obsspace::O
    function CartpoleSwingupV2(sim::MJSim)
        #=
        obsspace = MultiShape(
            pos = MultiShape(
                cart = ScalarShape(Float64),
                pole_zz = ScalarShape(Float64),
                pole_xz = ScalarShape(Float64),
            ),
            vel = MultiShape(
                cart = ScalarShape(Float64),
                pole_ang = ScalarShape(Float64),
            ),
        )
        =#
        obsspace = MultiShape(p1=ScalarShape(Float64),
                              p2=ScalarShape(Float64),
                              v1=ScalarShape(Float64),
                              v2=ScalarShape(Float64)
                             )
        env = new{typeof(sim), typeof(obsspace)}(sim, obsspace)
        LyceumMuJoCo.reset!(env)
    end
end

LyceumMuJoCo.getsim(env::CartpoleSwingupV2) = env.sim

LyceumMuJoCo.obsspace(env::CartpoleSwingupV2) = env.obsspace

CartpoleSwingupV2() = first(tconstruct(CartpoleSwingupV2, 1))
function LyceumBase.tconstruct(::Type{CartpoleSwingupV2}, n::Integer)
    modelpath = joinpath(@__DIR__, "cartpole.xml")
    Tuple(CartpoleSwingupV2(s) for s in tconstruct(MJSim, n, modelpath, skip=1))
end

# this bullshit
function LyceumMuJoCo.setstate!(env::CartpoleSwingupV2, state)
    sim = env.sim
    LyceumMuJoCo.copystate!(sim, state)
    forward!(sim)
    shaped = statespace(sim)(state)
    @uviews shaped begin copyto!(sim.d.qacc_warmstart, shaped.qacc_warmstart) end
    sim
end

# overwriting some of the functions for the environment
function LyceumMuJoCo.isdone(state, action, obs, env::CartpoleSwingupV2)
    #@info abs(state[1])
    return abs(obs[1])>5.0# || abs(obs[4]) > 13. #|| abs(obs[4]) > 10.
    #return false
end

# initializing mujoco
function LyceumMuJoCo.reset!(env::CartpoleSwingupV2)
    LyceumMuJoCo.reset_nofwd!(env.sim)
   
    qpos = env.sim.dn.qpos
    @uviews qpos begin
        qpos[:hinge_1] = pi
    end

    forward!(env.sim)

    env
end

function LyceumMuJoCo.randreset!(rng::Random.AbstractRNG, env::CartpoleSwingupV2)
    LyceumMuJoCo.reset_nofwd!(env.sim)

    qpos = env.sim.dn.qpos
    mag = 0.1
    #mag = pi
    @uviews qpos begin
        qpos[:slider] = 2mag * rand(rng) - mag
        qpos[:hinge_1] = pi + 2mag * rand(rng) - mag
        #qpos[:hinge_1] = pi + 2pi * rand(rng) - pi
    end
   
    #randn!(rng, env.sim.d.qvel)
    #env.sim.d.qvel .*= 0.1

    forward!(env.sim)

    #reset!(env)
    #env.sim.d.qpos[2] = 3.1
    env
end


function LyceumMuJoCo.getobs!(obs, env::CartpoleSwingupV2)
    @uviews obs begin
        sobs = obsspace(env)(obs)
        #sobs.pos.cart = env.sim.dn.qpos[:slider]
        #sobs.pos.pole_zz = env.sim.dn.xmat[:z, :z, :pole_1]
        #sobs.pos.pole_xz = env.sim.dn.xmat[:x, :z, :pole_1]
        #sobs.vel.cart = env.sim.d.qvel[1]
        #sobs.vel.pole_ang = env.sim.d.qvel[2]
        sobs[1] = env.sim.d.qpos[1]
        sobs[2] = env.sim.d.qpos[2]
        sobs[3] = env.sim.d.qvel[1]
        sobs[4] = env.sim.d.qvel[2]
    end
    obs
end

function LyceumMuJoCo.geteval(state, action, obs, env::CartpoleSwingupV2)
    #obsspace(env)(obs).pos.pole_zz
    getreward(state,action,obs,env)
end


function LyceumMuJoCo.getreward(state, action, obs, env::CartpoleSwingupV2)
    sobs = obsspace(env)(obs)
    #=

    centerT = Tolerance{Float64}(margin=2)
    ctrlT   = Tolerance{Float64}(margin=1, value_at_margin=0)
    velT    = Tolerance{Float64}(margin=5)

    upright = (sobs.pos.pole_zz + 1.0) / 2.0

    centered = (1.0 + centerT(sobs.pos.cart)) / 2.0

    small_control = (4.0 + ctrlT(first(action))) / 5.0

    small_velocity = (1.0 + velT(sobs.vel.pole_ang)) / 2.0
   
    upright * small_control * small_velocity * centered #- margin # including control costs too / margin is for keeping unvisited states cost high
    =#

    x = obs[1]
    theta = obs[2]
    up_reward = cos(theta)
    distance_penalty_reward = -0.01 * (x^2)
    #ctrl_cost = -0.01 * norm(action)^2
    #vel_cost = -0.001 * sobs.v2^2
    reward = up_reward + distance_penalty_reward #+ vel_cost #+ ctrl_cost

    #0.99^(time(env)/timestep(env)) * reward
    #1.001^(time(env)/timestep(env)) * reward
end

