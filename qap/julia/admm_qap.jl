function admm_qap(A, B, C=nothing, ub=nothing, args=nothing)
    maxit = args.maxit || 1000;
    tol = args.tol || 1e-6;
    gamma = args.gamma || 1.618;
    lowrank = args.lowrank || false;

    n, _ = size(A);
    L = make_L(A, B, C);
    Vhat = make_Vhat(n);
    J = make_gangster(n);
    Y0 = make_y0(n);
    R0 = make_r0(n, Vhat, Y0);
    Z0 = Y0 - (Vhat * R0 * Vhat');

    norm = norm(L);
    Vhat_nrows, _ = size(Vhat);
    L = L * (n*n / normL);
    beta = n / 3.;
    lbd = -Inf;
    ubd = Inf;

    for i in 1:maxit
        R_pre_proj = Vhat' * (Y + Z/beta) * Vhat;
        R_pre_proj = (R_pre_proj + R_pre_proj') / 2.;
        R_evals, R_evecs = eigen(R_pre_proj);

        if !lowrank
            pos_idx = S .> 0;
            if sum(pos_idx) > 0
                vhat_u = Vhat * R_evecs[:, pos_ix];
                VRV = vhat_u * diagm(S[pos_idx]) * vhat_u';
            else
                VRV = zeros(size(Y)); # shouldnt happen very often
            end
        else
            if S[end] > 0
                vhat_u = Vhat * R_evecs[:, end:end];
                VRV = vhat_u * R_evals[end] * vhat_u';
            else
                VRV = zeros(size(Y))
            end
        end

        # update Y
        Y = VRV - ((L + Z) / beta);
        Y = (Y + Y') / 2.;
        Y[J] = 0; Y[1, 1] = 1;
        clamp!(Y, 0, 1);
        pR = Y - VRV;

        # update Z
        Z = Z + gamma * beta * (Y - VRV);
        Z = (Z + Z') / 2.;
        Z[abs.(Z) .< tol] .= 0;

        # pretty print progress
        if div(i, 100) == 0 && args.verbose
            scale = normL / (n*n);
            lbd = max(lbd, lower_bound(L, J, Vhat, Z, n, scale=scale));
            ubd = min(ubd, upper_bound(Y, A, B));
            @printf("Iter %d | Lower bound: %.3f | Upper bound: %.3f", i, lbd, ubd);
        end
    end

    if (ub != nothing && lbd > ubd) || lbd > ubd
       ubd = max(lbd, ubd);
    end

    return lbd, ubd;
end
