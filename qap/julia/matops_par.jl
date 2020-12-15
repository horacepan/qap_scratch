using LinearAlgebra
using Statistics
using Random
using Printf
using NPZ
include("./matops.jl");

function vec2kron_t(vp)
    k, _ = size(vp);
    n = Int(sqrt(k));
    output = zeros(k, k);
    Threads.@threads for j in 1:n
        for i in 1:n
            row_idx = i + (j-1)*n;
            row = vp[row_idx, :];
            output[(i-1)*n+1: i*n, (j-1)*n+1: j*n] = reshape(row, n, n);
        end
    end
    return output;
end

function kron2vec_t(kp)
    k, _ = size(kp);
    n = Int(sqrt(k));
    output = zeros(k, k);

    Threads.@threads for j in 1:n
        for i in 1:n
            row_idx = i + (j-1)*n;
            _block = view(kp, (i-1)*n+1:i*n, (j-1)*n+1:j*n);
            output[row_idx, :] = reshape(_block, 1, :);
        end
    end
    return output;
end

function kron2block_t(mat)
    k, _ = size(mat);
    n = Int(sqrt(k));
    d = k - n;
    output = zeros(size(mat));

    Threads.@threads for j in 1:k
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

function block2kron_t(mat)
    k, _ = size(mat);
    n = Int(sqrt(k));
    d = k - n;
    output = zeros(k, k);

    Threads.@threads for t1 in 1:(k - n)
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

function vec2irrep_t(mat, cmat, cinv)
    kron_mat = vec2kron_t(mat);
    blocked = kron2block_t(kron_mat);
    irr = cmat * blocked * cinv;
    return irr;
end

function kron2irrep_t(kron_mat, cmat, cinv)
    blocked = kron2block_t(kron_mat);
    irr = cmat * blocked * cinv;
    return irr;
end

function irrep2kron_t(irr_mat, cmat, cinv)
    blocked = cinv * irr_mat * cmat;
    kron_mat = block2kron_t(blocked);
    return kron_mat;
end

function irrep2vec_t(irr_mat, cmat, cinv)
    blocked = cinv * irr_mat * cmat;
    kron_mat = block2kron_t(blocked);
    vec = kron2vec_t(kron_mat);
    return vec;
end

function test_vec_kron_threaded()
    n = 30;
    mat = rand(n, n);
    v = reshape(mat, :, 1) * reshape(mat, :, 1)';
    k = kron(mat, mat);
    @time println("kron2vec: ", isapprox(kron2vec_t(k), v));
    @time println("vec2kron: ", isapprox(vec2kron_t(v), k));
end

function test_irrep_threaded()
    n = 60;
    c = load_cmat(n);
    ci = inv(c);
    p = rand_perm(n);
    vp = operm(p);
    kp = kron(p, p);
    @time println("irrep2vec : ", isapprox(vp, irrep2vec_t(vec2irrep_t(vp, c, ci), c, ci)));
    @time println("irrep2kron: ", isapprox(kp, irrep2kron_t(kron2irrep_t(kp, c, ci), c, ci)));
end

# println("Running with: ", Threads.nthreads(), " threads.");
# test_vec_kron_threaded();
# test_irrep_threaded();
