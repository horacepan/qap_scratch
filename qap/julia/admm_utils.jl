using LinearAlgebra
using SparseArrays

function fnorm(mat)
   return vecnorm(mat);
end

function psd_project(mat)
    eigs = eigen(mat);
    pos_idx = eigs.values .> 0;
    vals = eigs.values[pos_idx];
    vecs = eig.vectors[:, pos_idx];
    return vecs * diagm(vals) * vecs';
end

function make_L(A, B, C=nothing)
    L = kron(B, A);
    return L;
end

function make_L(A, B, C=nothing)
    if C == nothing
       C = zeros(size(A));
    end

    n, _ = size(A);
    L = zeros(1+n*n, 1+n*n);
    L[2:end, 1] = reshape(C, :, 1)
    L[1, 2:end] = reshape(C, 1, :);
    L[2:end, 2:end] = kron(B, A);
    return L;
end

function make_y0(n)
    n2 = n*n;
    yhat = zeros(n2+1, n2+1);
    nI = n * I(n) .- 1.0;

    yhat[1, 1] = 1.0;
    yhat[1, 2:end] .= 1. / n;
    yhat[2:end, 1] .= 1. / n;
    yhat[2:end, 2:end] .= (1. / n2) .+ (1 / (n2 * (n-1))) * kron(nI, nI);
    return yhat;
end


function make_r0(n, Vhat, Yhat)
    R0 = Vhat' * Yhat * Vhat;
    R0 = (R0 + R0') / 2.;
    return R0;
end

function make_vhat(n)
    V = [I(n-1); -ones(1, n-1)];
    qr_res = qr(V);
    V = Array(qr_res.Q);

    n, n1 = size(V);
    vxv = kron(V, V);
    r1 = zeros(1, n1*n1+1);
    r1[1, 1] = sqrt(0.5);
    r2 = [ones(n*n, 1) * sqrt(0.5)/n vxv];
    println(size(r1), size(r2));
    Vhat = [r1; r2];
end

function make_that(n)
    In = I(n);
    en = ones(n, 1);
    kron_ier = kron(In, en');
    kron_eri = kron(en', In);
    krons = [kron_ier; kron_eri];
    That = [-ones(2*n, 1) krons];
    return That
end
