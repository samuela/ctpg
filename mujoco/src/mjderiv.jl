

using BangBang
using StaticArrays
using MuJoCo

struct MJDeriv
    m::jlModel
    dmain::jlData
    datas::Vector{jlData}
    daccdpos::Matrix{Float64} 
    daccdvel::Matrix{Float64}
    daccdfrc::Matrix{Float64}

    eps::Float64
    nwarmup::Int
    niter::Int

    function MJDeriv(envtype;
                     nthreads=Threads.nthreads(),
                     eps=1e-6,
                     nwarmup=3,
                     niter=30)

        envs = tconstruct(envtype, nthreads+1) # first data is dmain
        datas = [ envs[i].sim.d for i=2:nthreads+1]

        nv = Int(envs[1].sim.m.nv)

        daccdpos = zeros(Float64, nv, nv)
        daccdvel = zeros(Float64, nv, nv)
        daccdfrc = zeros(Float64, nv, nv)

        new(envs[1].sim.m, envs[1].sim.d, datas,
            daccdpos, daccdvel, daccdfrc,
            eps, nwarmup, niter)
    end
end

function relnorm(res, base)
    l1r = norm(res,1)
    l1b = norm(base,1)
    m = MuJoCo.MJCore.mjMINVAL
    log10( max(m, (l1r / max(m, l1b)) ))
end

# problems: G1 F0 G0 F1
function checkderiv(m::jlModel, d::jlData,
                    G0::Matrix{Float64}, G1::Matrix{Float64}, G2::Matrix{Float64},
                    F0::Matrix{Float64}, F1::Matrix{Float64}, F2::Matrix{Float64})
    nv = Int(m.nv)

    error = zeros(8)

    # G2*F2 - I
    mat = G2*F2 - Matrix(I, nv, nv)
    error[1] = relnorm(mat, G2)

    # G2 - G2'
    mat = G2 - G2'
    error[2] = relnorm(mat, G2)

    # G1 - G1'
    mat = G1 - G1'
    error[3] = relnorm(mat, G1)

    # F2 - F2'
    mat = F2 - F2'
    error[4] = relnorm(mat, F2)

    # G1 + G2*F1
    mat = G1 + ( G2 * F1 )
    error[5] = relnorm(mat, G1)

    # G0 + G2*F0
    mat = G0 + G2 * F0
    error[6] = relnorm(mat, G0)

    # F1 + F2*G1
    mat = F1 + F2 * G1
    error[7] = relnorm(mat, F1)

    # F0 + F2*G0
    mat = F0 + F2 * G0
    error[8] = relnorm(mat, F0)

    return error
end

function get_acceljac(mjd::MJDeriv, state, ctrl) # TODO could pass in dmain...
    d = mjd.dmain           # set dmain; TODO there are other fields...
    nq = length(d.qpos)     # could also get qacc_warmstart once, here, instead of in threads...
    nv = length(d.qvel)
    d.qpos .= state[1:nq]
    d.qvel .= state[(nq+1):end]
    d.ctrl .= ctrl
    # set d.time?

    r = splitrange(nv, length(mjd.datas))

    #Threads.@threads for irange in r
    Threads.@sync for irange in r # any different than @threads?
        Threads.@spawn fwdworker(mjd, irange, true)
    end
end


function getquat!(q, qpos, idx)
    for i=1:4
        q[i] = qpos[idx + i]
    end
    q
end
function setquat!(qpos, q, idx)
    for i=1:4
        qpos[idx + i] = q[i]
    end
    qpos
end

function bad_fwdworker(mjd::MJDeriv, dmain, x, u)

    m, d = mjd.m, mjd.datas[Threads.threadid()]
    block1, block2, block3 = mjd.daccdpos, mjd.daccdvel, mjd.daccdfrc

    nv = Int(m.nv) #length(d.qvel)
    eps = mjd.eps
    nwarmup = mjd.eps

    # copy state and control from dmain to thread-specific d
    @set!! d.time = dmain.time
    @inbounds begin
        copyto!(d.qpos,           dmain.qpos)
        copyto!(d.qvel,           dmain.qvel)
        copyto!(d.qacc,           dmain.qacc)
        copyto!(d.qacc_warmstart, dmain.qacc_warmstart)
        copyto!(d.qfrc_applied,   dmain.qfrc_applied)
        copyto!(d.xfrc_applied,   dmain.xfrc_applied)
        copyto!(d.ctrl,           dmain.ctrl)
    end

    # run full computation at center point (usually faster than copying dmain)
    mj_forward(m, d)
    #for rep=1:nwarmup
    #    mj_forwardSkip(m, d, MuJoCo.MJCore.mjSTAGE_VEL, true)
    #end

    return nothing
end
function fwdworker(mjd::MJDeriv, irange, getfrc::Bool=true)

    m, dmain, d = mjd.m, mjd.dmain, mjd.datas[Threads.threadid()]
    block1, block2, block3 = mjd.daccdpos, mjd.daccdvel, mjd.daccdfrc

    nv = Int(m.nv) #length(d.qvel)
    eps = mjd.eps
    nwarmup = mjd.eps

    # copy state and control from dmain to thread-specific d
    @set!! d.time = dmain.time
    @inbounds begin
        copyto!(d.qpos,           dmain.qpos)
        copyto!(d.qvel,           dmain.qvel)
        copyto!(d.qacc,           dmain.qacc)
        copyto!(d.qacc_warmstart, dmain.qacc_warmstart)
        copyto!(d.qfrc_applied,   dmain.qfrc_applied)
        copyto!(d.xfrc_applied,   dmain.xfrc_applied)
        copyto!(d.ctrl,           dmain.ctrl)
    end

    # run full computation at center point (usually faster than copying dmain)
    mj_forward(m, d)
    for rep=1:nwarmup
        mj_forwardSkip(m, d, MuJoCo.MJCore.mjSTAGE_VEL, true)
    end

    # save output for center point and warmstart (needed in forward only)
    center    = copy(d.qacc) #SVector{nv, Float64}(d.qacc)
    warmstart = copy(d.qacc_warmstart) #SVector{nv, Float64}(d.qacc_warmstart)
    angvel    = zeros(3)
    quat      = zeros(4)

    # finite-difference over force or acceleration: skip = mjSTAGE_VEL
    if getfrc
        for i in irange
            # perturb selected target
            qfrc_i = d.qfrc_applied[i]
            d.qfrc_applied[i] += eps

            # evaluate dynamics, with center warmstart
            copyto!(d.qacc_warmstart, warmstart)
            mj_forwardSkip(m, d, MuJoCo.MJCore.mjSTAGE_VEL, true)

            # undo perturbation
            d.qfrc_applied[i] = qfrc_i

            # compute column i of derivative 2
            for j=1:nv
                block3[j,i] = (d.qacc[j] - center[j])/eps
            end
        end
    end

    # finite-difference over velocity: skip = mjSTAGE_POS
    for i in irange
        # perturb velocity
        qvel_i = d.qvel[i]
        d.qvel[i] += eps

        # evaluate dynamics, with center warmstart
        copyto!(d.qacc_warmstart, warmstart)
        mj_forwardSkip(m, d, MuJoCo.MJCore.mjSTAGE_POS, true)

        # undo perturbation
        d.qvel[i] = qvel_i

        # compute column i of derivative 1
        for j=1:nv
            block2[j,i] = (d.qacc[j] - center[j])/eps
        end
    end

    # finite-difference over position: skip = mjSTAGE_NONE
    for i in irange
        # get joint id for this dof
        jid = m.dof_jntid[i] + 1

        # get quaternion address and dof position within quaternion (-1: not in quaternion)
        quatadr = -1
        if m.jnt_type[jid] == MuJoCo.MJCore.mjJNT_BALL
            quatadr = m.jnt_qposadr[jid]
            dofpos = i - m.jnt_dofadr[jid]
        elseif m.jnt_type[jid] == MuJoCo.MJCore.mjJNT_FREE && i>=m.jnt_dofadr[jid]+4
            quatadr = m.jnt_qposadr[jid] + 3
            dofpos = i - m.jnt_dofadr[jid] - 3
        end

        # apply quaternion or simple perturbation
        if quatadr>=0
            angvel .= 0.0
            angvel[dofpos] = eps # already +1 from i
            getquat!(quat, d.qpos, quatadr)
            MuJoCo.MJCore.mju_quatIntegrate(quat, angvel, 1.0)
            # TODO mujoco.jl didn't expose mju_quatIntegrate, yet
            setquat!(d.qpos, quat, quatadr)
        else
            d.qpos[i] += eps
        end

        # evaluate dynamics, with center warmstart
        copyto!(d.qacc_warmstart, warmstart)
        mj_forwardSkip(m, d, MuJoCo.MJCore.mjSTAGE_NONE, true)

        # undo perturbation
        copyto!(d.qpos, dmain.qpos) # since quat

        # compute column i of derivative 0
        for j=1:nv
            block1[j,i] = (d.qacc[j] - center[j])/eps
        end
    end

    return nothing
end

function invworker(m::jlModel, dmain::jlData, d::jlData,
                   irange,
                   block1::Matrix{Float64}, block2::Matrix{Float64}, block3::Matrix{Float64})

    nv = length(d.qvel)
    eps = 1e-6
    nwarmup = 3

    # copy state and control from dmain to thread-specific d
    @set!! d.time = dmain.time
    @inbounds begin
        copyto!(d.qpos,           dmain.qpos)
        copyto!(d.qvel,           dmain.qvel)
        copyto!(d.qacc,           dmain.qacc)
        copyto!(d.qacc_warmstart, dmain.qacc_warmstart)
        copyto!(d.qfrc_applied,   dmain.qfrc_applied)
        copyto!(d.xfrc_applied,   dmain.xfrc_applied)
        copyto!(d.ctrl,           dmain.ctrl)
    end

    # run full computation at center point (usually faster than copying dmain)
    mj_inverse(m, d)

    # save output for center point and warmstart (needed in forward only)
    center    = copy(d.qfrc_inverse) #SVector{nv,Float64}(d.qfrc_inverse)
    warmstart = copy(d.qacc_warmstart) #SVector{nv,Float64}(d.qacc_warmstart)
    angvel    = zeros(3)
    quat      = zeros(4)

    # finite-difference over force or acceleration: skip = mjSTAGE_VEL
    for i in irange
        # perturb selected target
        qacc_i = d.qacc[i]
        d.qacc[i] += eps

        # evaluate dynamics, with center warmstart
        mj_inverseSkip(m, d, MuJoCo.MJCore.mjSTAGE_VEL, true)

        # undo perturbation
        d.qacc[i] = qacc_i

        # compute column i of derivative 2
        block3[:,i] = (d.qfrc_inverse - center)/eps
    end

    # finite-difference over velocity: skip = mjSTAGE_POS
    for i in irange
        # perturb velocity
        qvel_i = d.qvel[i]
        d.qvel[i] += eps

        # evaluate dynamics, with center warmstart
        mj_inverseSkip(m, d, MuJoCo.MJCore.mjSTAGE_POS, true)

        # undo perturbation
        d.qvel[i] = qvel_i

        # compute column i of derivative 1
        block2[:,i] = (d.qfrc_inverse - center)/eps
    end

    # finite-difference over position: skip = mjSTAGE_NONE
    for i in irange
        # get joint id for this dof
        jid = m.dof_jntid[i] + 1

        # get quaternion address and dof position within quaternion (-1: not in quaternion)
        quatadr = -1
        if m.jnt_type[jid] == MuJoCo.MJCore.mjJNT_BALL
            quatadr = m.jnt_qposadr[jid]
            dofpos = i - m.jnt_dofadr[jid]
        elseif m.jnt_type[jid] == MuJoCo.MJCore.mjJNT_FREE && i>=m.jnt_dofadr[jid]+4
            quatadr = m.jnt_qposadr[jid] + 3
            dofpos = i - m.jnt_dofadr[jid] - 3
        end

        # apply quaternion or simple perturbation
        if quatadr>=0
            angvel .= 0.0 
            angvel[dofpos] = eps # already +1 from i
            getquat!(quat, d.qpos, quatadr)
            MuJoCo.MJCore.mju_quatIntegrate(quat, angvel, 1.0)
            setquat!(d.qpos, quat, quatadr)
        else
            d.qpos[i] += eps
        end

        # evaluate dynamics, with center warmstart
        mj_inverseSkip(m, d, MuJoCo.MJCore.mjSTAGE_NONE, true)

        # undo perturbation
        copyto!(d.qpos, dmain.qpos)

        # compute column i of derivative 0
        block1[:,i] = (d.qfrc_inverse - center)/eps
    end

    return nothing
end


