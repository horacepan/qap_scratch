using LinearAlgebra
using MAT
using Printf
include("./admm_utils.jl")

function lower_bound(
        L::Array{Float64, 2},
        J::BitArray{2},
        Vhat::Array{Float64,2},
        That::Array{Float64,2},
        Z::Array{Float64,2},
        n::Int,
        scale::Float64)
    Q, _ = qr(That');
    Q = Q[:, 1:end-1];

    Uloc = [Vhat Q];
    Zloc = Uloc' * Z * Uloc;

    W11 = Zloc[1:(n-1)*(n-1)+1, 1:(n-1)*(n-1)+1];
    W11 = (W11 + W11') / 2.;
    W12 = Zloc[1:(n-1)*(n-1)+1, (n-1)*(n-1)+2:end];
    W22 = Zloc[(n-1)*(n-1)+2:end, (n-1)*(n-1)+2:end];

    Dw, Uw = eigen(W11); # Dw = vals, Uw = eigvecs
    neg_idx = Dw .< 0;
    W11 = Uw[:, neg_idx] * Diagonal(Dw[neg_idx]) * Uw[:, neg_idx]';

    Zp = Uloc * [W11 W12; W12' W22] * Uloc';
    Zp = (Zp + Zp') / 2.;

    Yp = zeros(size(L));
    Yp[(L + Zp .< 0)] .= 1;
    Yp[J] .= 0; Yp[1, 1] = 1;
    lbd = sum((L + Zp) .* Yp) * scale;
    return lbd;
end

function admm_qap(A, B, C=nothing, args=nothing)
    maxit = args["maxit"];
    tol = args["tol"];
    gamma = args["gamma"];
    lowrank = args["lowrank"];

    n, _ = size(A);
    L = make_L(A, B, C);
    Vhat = make_vhat(n);
    That = make_that(n);
    J = make_gangster(n);

    normL = norm(L);
    Vhat_nrows, _ = size(Vhat);
    scale = normL / (n*n)
    L = L / scale; # rescaled for numerical stability
    beta = n / 3.;
    lbd = -Inf;
    ubd = Inf;

    # initial variables
    Y0 = make_y0(n);
    R0 = make_r0(n, Vhat, Y0);
    Z0 = Y0 - (Vhat * R0 * Vhat');
    Y = Y0;
    R = R0;
    Z = Z0;

    for i in 1:maxit
        R_pre_proj = Vhat' * (Y + Z/beta) * Vhat;
        R_pre_proj = (R_pre_proj + R_pre_proj') / 2.;
        R_evals, R_evecs = eigen(R_pre_proj);

        if !lowrank
            pos_idx = R_evals .> 0;
            if sum(pos_idx) > 0
                vhat_u = Vhat * R_evecs[:, pos_idx];
                VRV = vhat_u * Diagonal(R_evals[pos_idx]) * vhat_u';
            else
                VRV = zeros(size(Y)); # shouldnt happen very often
            end
        else
            if R_evals[end] > 0
                vhat_u = Vhat * R_evecs[:, end:end];
                VRV = vhat_u * R_evals[end] * vhat_u';
            else
                VRV = zeros(size(Y))
            end
        end

        # update Y
        Y = VRV - ((L + Z) / beta);
        Y = (Y + Y') / 2.;
        Y[J] .= 0; Y[1, 1] = 1;
        clamp!(Y, 0, 1);
        pR = Y - VRV;

        # update Z
        Z = Z + (gamma*beta*(Y - VRV));
        Z = (Z + Z') / 2.;
        Z[abs.(Z) .< tol] .= 0;

        # pretty print progress
        if mod(i, 100) == 0 && args["verbose"]
            lbd = max(lbd, lower_bound(L, J, Vhat, That, Z, n, scale));
            ly = sum(L .* Y) * scale;
            # ubd = min(ubd, upper_bound(Y, A, B));
            #@printf("Iter %d | Lower bound: %.3f | Upper bound: %.3f", i, lbd, ubd);
            @printf("Iter %d | Lower bound: %.3f\n", i, ly);
        end
    end

    if (ubd != nothing && lbd > ubd) || lbd > ubd
       ubd = max(lbd, ubd);
    end
    @printf("Final: %.3f\n", sum(L.*Y)*scale);
    return lbd, Y;
end
