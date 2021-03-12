"""
This file is for Gym acrobot environment https://github.com/openai/gym/tree/master/gym/envs/classic_control.
These are based on https://github.com/WilsonWangTHU/mbbl
"""

using LyceumBase
using Shapes
using Distributions
using Base: @propagate_inbounds

struct Acrobot{D,SP,OP,AP,RP} <: AbstractEnvironment
    state::D
    obs::D
    action::D
    time::D
    nextobs::D
    ctrlrange::D

    sp::SP
    op::OP
    ap::AP
    rp::RP
    function Acrobot{T}() where T<:AbstractFloat
        nobs = 6
        nstate = 4
        naction = 1

        ctrlrange = zeros(T, 2)
        ctrlrange[1] = -1.;  ctrlrange[2] = 1.
        
        sp = VectorShape(T, nstate)
        op = VectorShape(T, nobs)
        ap = VectorShape(T, naction)
        rp = ScalarShape(T)
        s = zeros(T, nstate)
        o = zeros(T, nobs)
        a = zeros(T, naction)
        env = new{typeof(s), 
        typeof(sp), typeof(op), typeof(ap), typeof(rp)}(s,o,a,zeros(T,1),
                                                        zeros(T, nobs),
                                                        ctrlrange,
                                                        sp,op,ap,rp)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{Acrobot}, n::Integer)
    Tuple(Acrobot{Float64}() for _ in 1:n)
end
Acrobot() = first(tconstruct(Acrobot, 1))

@inline LyceumBase.statespace(env::Acrobot) = env.sp
@inline LyceumBase.obsspace(env::Acrobot) = env.op
@inline LyceumBase.actionspace(env::Acrobot) = env.ap
@inline LyceumBase.rewardspace(env::Acrobot) = env.rp


@inline function LyceumBase.getstate!(state, env::Acrobot) 
    #state .= env.state
    state[1] = env.state[1] # unroll for speed
    state[2] = env.state[2]
    state[3] = env.state[3]
    state[4] = env.state[4]
end
@inline function LyceumBase.setstate!(env::Acrobot, state) 
    #env.state .= state
    env.state[1] = state[1] # unroll for speed
    env.state[2] = state[2]
    env.state[3] = state[3]
    env.state[4] = state[4]
end

@inline LyceumBase.getaction!(action, env::Acrobot) = action .= env.action
@inline LyceumBase.setaction!(env::Acrobot, action) = env.action .= action

@propagate_inbounds function LyceumBase.isdone(state, action, obs, env::Acrobot)
    false
end
@propagate_inbounds function LyceumBase.getobs!(obs, env::Acrobot)
    s = env.state
    o = env.obs
    o[2], o[1] = sincos(s[1])
    o[4], o[3] = sincos(s[2])
    o[5] = s[3]
    o[6] = s[4]
    
    obs .= env.obs
end
@propagate_inbounds function LyceumBase.getreward(s, a, o, env::Acrobot) 
    h1 = o[1]  # Height of first arm
    h2 = o[1] * o[3] - o[2] * o[4]  # Height of second arm
    return - (h1 + h2)
    
end
@propagate_inbounds function LyceumBase.geteval(s, a, o, env::Acrobot)
    h1 = o[1]  # Height of first arm
    h2 = o[1] * o[3] - o[2] * o[4]  # Height of second arm
    return - (h1 + h2)
   
end
@propagate_inbounds function LyceumBase.reset!(env::Acrobot)
    randreset!(env)
end
@propagate_inbounds function LyceumBase.randreset!(env::Acrobot)
    env.state .= rand(Uniform(-0.1, 0.1), 4)
    getobs!(env.obs, env)
    env.time[1] = 0.0
    env
end

@inline function _wrappi(v)
    diff = 2pi
    while v >  π; v -= diff; end;
    while v < -π; v += diff; end;
    v
end

@propagate_inbounds function LyceumBase.step!(env::Acrobot)
    
    state, act, ctrlrange = env.state, env.action, env.ctrlrange
    """
        corresponds to the one in Benchmark paper
        if action[0] < -.33:
            action = 0
        elif action[0] < .33:
            action = 1
        else:
            action = 2
        Also in Gym env
        AVAIL_TORQUE = [-1., 0., +1] , torque = self.AVAIL_TORQUE[a]
    """
    act_d = 0.0
    if act[1] < -0.33; act_d = -1.0; elseif act[1] < 0.33; act_d = 0.0; else; act_d = 1.0; end;

    # BOOK default
    ns = MArray( _rk4(_dsdt, state, act_d, SA_F64[0, 0.2]) )

    #wrap function/ maybe there is more efficient way
    ns[1] = _wrappi(ns[1])
    ns[2] = _wrappi(ns[2])
    ns[3] = clamp(ns[3], -4 * π, 4 * π)
    ns[4] = clamp(ns[4], -9 * π, 9 * π)
    
    setstate!(env, ns)
    env.time[1] += timestep(env)
    env
end

function _dsdt(s::AbstractArray, a)#, t::AbstractFloat)
        m1 = 1.; m2 = 1.
        l1 = 1.; lc1 = 0.5; lc2 = 0.5
        I1 = 1.; I2 = 1.
        g = 9.8
        theta1, theta2, dtheta1, dtheta2 = s
        sintheta2, costheta2 = sincos(theta2)

        d1       = m1 * lc1^2 + m2 * (l1^2 + lc2^2 + 2 * l1 * lc2 * costheta2) + I1 + I2
        d2       = m2 * (lc2^2 + l1 * lc2 * costheta2) + I2
        phi2     = m2 * lc2 * g * cos(theta1 + theta2 - π / 2.)
        phi1     = - (m2 * l1 * lc2 * dtheta2^2 * sintheta2) - (2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sintheta2)
                    + ((m1 * lc1 + m2 * l1) * g * cos(theta1 - π / 2) + phi2)
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1^2 * sintheta2 - phi2) / (m2 * lc2^2 + I2 - d2^2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return SVector{4, eltype(s)}(dtheta1, dtheta2, ddtheta1, ddtheta2)
end

function _rk4(derivs::Function, y0::AbstractArray, a, t::AbstractArray)
    #y0 in our case should always be length 4, since we include the action
    yout = SVector{4, eltype(y0)}(y0)
    for i in 1:(length(t)-1)
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout
        k1 = derivs(y0, a)#, thist)
        k2 = derivs(y0 + dt2 * k1, a)#, thist + dt2)
        k3 = derivs(y0 + dt2 * k2, a)#, thist + dt2)
        k4 = derivs(y0 + dt * k3, a)#, thist + dt)
        yout = y0 + dt / 6.0 * (k1 + (2 * k2) + (2 * k3) + k4)
    end
    return yout # don't store intermediate computations, just return final
end
@inline LyceumBase.timestep(env::Acrobot) = 1.0
@inline Base.time(env::Acrobot) = env.time[1]

