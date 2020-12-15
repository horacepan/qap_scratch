using LinearAlgebra
using Statistics
using Random
using Printf
using NPZ

function rand_perm(n)
    perm = randperm(n);
    mat = zeros(n, n);
    for (i, j) in enumerate(perm)
        mat[i, j] = 1;
    end
    return mat;
end

function _outer(A, B)
    return reshape(A, :, 1) * reshape(B, :, 1)';
end

function operm(p)
   return _outer(p, p);
end

function load_cmat(n)
    prefix = "./../cmats/";
    c2 = npzread(prefix * string(n-2) * "11.npy");
    c1 = npzread(prefix * string(n-1) * "1.npy");
    mat = zeros(n*n, n*n);

    d, _ = size(c2);
    mat[begin:d, begin:d] = c2;
    mat[d+1:end, d+1:end] = c1;
    return mat;
end

function vec2kron(vp)
    k, _ = size(vp);
    n = Int(sqrt(k));
    output = zeros(k, k);
    for j in 1:n
        for i in 1:n
            row_idx = i + (j-1)*n;
            row = vp[row_idx, :];
            output[(i-1)*n+1: i*n, (j-1)*n+1: j*n] = reshape(row, n, n);
        end
    end
    return output;
end

function kron2vec(kp)
    k, _ = size(kp);
    n = Int(sqrt(k));
    output = zeros(k, k);

    for j in 1:n
        for i in 1:n
            row_idx = i + (j-1)*n;
            _block = view(kp, (i-1)*n+1:i*n, (j-1)*n+1:j*n);
            output[row_idx, :] = reshape(_block, 1, :);
        end
    end
    return output;
end

function kron2block(mat)
    k, _ = size(mat);
    n = Int(sqrt(k));
    d = k - n;
    output = zeros(size(mat));

    for j in 1:k
        for i in 1:k
            # figure out the appropriate x,y,z,w indices
            x = div(i-1, n) + 1;
            y = div(j-1, n) + 1;
            z = i - ((x-1) * n);
            w = j - ((y-1) * n);

            if (y == w) && (x == z)
                output[x + d, y + d] = mat[i, j];
            elseif y != w && x != z
                if z > x
                    t1 = (x-1) * (n-1) + z - 1
                else
                    t1 = (x-1) * (n-1) + z
                end

                if w > y
                    t2 = (y-1) * (n-1) + w - 1
                else
                    t2 = (y-1) * (n-1) + w
                end
                output[t1, t2] = mat[i, j];
            end
        end
    end
    return output;
end

function block2kron(mat)
    k, _ = size(mat);
    n = Int(sqrt(k));
    d = k - n;
    output = zeros(k, k);

    for t1 in 1:(k - n)
        for t2 in 1:(k-n)
            x = div(t1-1, n-1) + 1;
            rem_x = t1 - (x-1)*(n-1);
            if rem_x >= x
                z = rem_x + 1;
            else
                z = rem_x ;
            end
            i = z + ((x-1) * n);
            y = div(t2-1, n-1) + 1;
            rem_y = t2 - (y-1)*(n-1);
            if rem_y >= y
                w = rem_y + 1;
            else
                w = rem_y;
            end
            j = w + ((y-1) * n);

            output[i, j] = mat[t1, t2];
        end
    end

    for y in 1:n
        for x in 1:n
            z = x;
            w = y;
            i = z + ((x-1) * n);
            j = w + ((y-1) * n);
            output[i, j] = mat[x + d, y + d];

        end
    end

    return output;
end

function vec2irrep(mat, cmat, cinv)
    kron_mat = vec2kron(mat);
    blocked = kron2block(kron_mat);
    irr = cmat * blocked * cinv;
    return irr;
end

function kron2irrep(kron_mat, cmat, cinv)
    blocked = kron2block(kron_mat);
    irr = cmat * blocked * cinv;
    return irr;
end

function irrep2kron(irr_mat, cmat, cinv)
    blocked = cinv * irr_mat * cmat;
    kron_mat = block2kron(blocked);
    return kron_mat;
end

function irrep2vec(irr_mat, cmat, cinv)
    blocked = cinv * irr_mat * cmat;
    kron_mat = block2kron(blocked);
    vec = kron2vec(kron_mat);
    return vec;
end

function test_vec_kron()
    n = 10;
    mat = rand(n, n);
    v = reshape(mat, :, 1) * reshape(mat, :, 1)';
    k = kron(mat, mat);
    println("kron2vec: ", isapprox(kron2vec(k), v));
    println("vec2kron: ", isapprox(vec2kron(v), k));
end

function test_irrep()
    n = 10;
    c = load_cmat(n);
    ci = inv(c);
    p = rand_perm(n);
    vp = operm(p);
    kp = kron(p, p);
    println("irrep2vec : ", isapprox(vp, irrep2vec(vec2irrep(vp, c, ci), c, ci)));
    println("irrep2kron: ", isapprox(kp, irrep2kron(kron2irrep(kp, c, ci), c, ci)));
end
