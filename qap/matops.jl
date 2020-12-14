using LinearAlgebra
using Statistics
using Printf

function vec_to_kron(vp)
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

function kron_to_vec(kp)
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

function kron_to_block(mat)
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
                #println("$i, $j -> $t1, $t2");
                output[t1, t2] = mat[i, j];
            end
        end
    end
    return output;
end
