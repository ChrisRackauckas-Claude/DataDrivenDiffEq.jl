"""
Data augmentation and reduction utilities for the SINDY workflow.

This module provides functions for:
- Delay embedding (time-delay coordinates)
- Hankel matrix construction
- Dimensionality reduction via truncated SVD

These are useful for HAVOK analysis and handling chaotic systems where
time-delay embeddings can reveal underlying structure.

# References

- Brunton, S. L., Brunton, B. W., Proctor, J. L., Kaiser, E., & Kutz, J. N. (2017).
  Chaos as an intermittently forced linear system. Nature Communications, 8(1), 19.
  https://doi.org/10.1038/s41467-017-00030-8

- Arbabi, H., & Mezic, I. (2017). Ergodic theory, dynamic mode decomposition,
  and computation of spectral properties of the Koopman operator.
  SIAM Journal on Applied Dynamical Systems, 16(4), 2096-2126.
"""

"""
    $(SIGNATURES)

Construct a Hankel matrix from a 1D time series vector.

Given a vector `x = [x₁, x₂, ..., xₙ]` and `num_rows` rows, constructs the Hankel matrix:

```
H = [x₁    x₂    x₃   ... xₙ₋ₘ₊₁]
    [x₂    x₃    x₄   ... xₙ₋ₘ₊₂]
    [x₃    x₄    x₅   ... xₙ₋ₘ₊₃]
    [⋮     ⋮     ⋮    ⋱  ⋮      ]
    [xₘ    xₘ₊₁  xₘ₊₂ ... xₙ    ]
```

where `m = num_rows`.

# Arguments

  - `x`: Input vector (1D time series)
  - `num_rows`: Number of rows in the Hankel matrix (number of delays + 1)

# Returns

  - Hankel matrix of size `(num_rows, length(x) - num_rows + 1)`

# Example

```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
H = hankel_matrix(x, 4)
# Returns:
# 4×7 Matrix:
#  1.0  2.0  3.0  4.0  5.0  6.0  7.0
#  2.0  3.0  4.0  5.0  6.0  7.0  8.0
#  3.0  4.0  5.0  6.0  7.0  8.0  9.0
#  4.0  5.0  6.0  7.0  8.0  9.0  10.0
```
"""
function hankel_matrix(x::AbstractVector, num_rows::Integer)
    n = length(x)
    @assert num_rows >= 1 "num_rows must be at least 1"
    @assert num_rows <= n "num_rows ($num_rows) cannot exceed length of x ($n)"

    num_cols = n - num_rows + 1
    H = similar(x, num_rows, num_cols)

    for j in 1:num_cols
        for i in 1:num_rows
            H[i, j] = x[i + j - 1]
        end
    end

    return H
end

"""
    $(SIGNATURES)

Create delay-embedded coordinates from a data matrix.

Given data matrix `X` with states as rows and time samples as columns,
creates new coordinates by stacking time-delayed versions of each state.

For a single state `x = [x(t₁), x(t₂), ...]` with `num_delays` delays and
delay interval `τ`, the embedded coordinates are:

```
[x(t),      x(t+τ),      ...]
[x(t-τ),    x(t),        ...]
[x(t-2τ),   x(t-τ),      ...]
  ⋮           ⋮           ⋱
[x(t-mτ),   x(t-(m-1)τ), ...]
```

where `m = num_delays`.

# Arguments

  - `X`: Data matrix of size `(n_states, n_samples)`
  - `num_delays`: Number of time delays to add (resulting in `num_delays + 1`
    copies of each state at different delays)

# Keyword Arguments

  - `τ::Integer = 1`: Delay interval in samples (default: 1 sample)

# Returns

  - Embedded data matrix of size `(n_states * (num_delays + 1), n_samples - num_delays * τ)`

# Example

```julia
# Single state variable
X = reshape(1.0:10.0, 1, 10)  # 1×10 matrix
X_embed = delay_embedding(X, 2)
# Creates 3×8 matrix with original and 2 delayed copies

# Multiple state variables
X = randn(3, 100)  # 3 states, 100 samples
X_embed = delay_embedding(X, 3, τ = 2)
# Creates 12×94 matrix (3 states × 4 time copies)
```

See also: [`hankel_matrix`](@ref), [`reduce_dimension`](@ref)
"""
function delay_embedding(X::AbstractMatrix, num_delays::Integer; τ::Integer = 1)
    n_states, n_samples = size(X)
    @assert num_delays >= 0 "num_delays must be non-negative"
    @assert τ >= 1 "delay interval τ must be at least 1"
    @assert n_samples > num_delays * τ "Not enough samples for requested delays"

    new_n_states = n_states * (num_delays + 1)
    new_n_samples = n_samples - num_delays * τ

    X_embedded = similar(X, new_n_states, new_n_samples)

    for d in 0:num_delays
        offset = num_delays * τ - d * τ
        for i in 1:n_states
            row_idx = d * n_states + i
            for j in 1:new_n_samples
                X_embedded[row_idx, j] = X[i, j + offset]
            end
        end
    end

    return X_embedded
end

"""
    $(SIGNATURES)

Delay embedding for a 1D time series vector.

Convenience method that converts a vector to a 1-row matrix, applies
delay embedding, and returns the result.

# Arguments

  - `x`: Input vector (1D time series)
  - `num_delays`: Number of time delays to add

# Keyword Arguments

  - `τ::Integer = 1`: Delay interval in samples

# Returns

  - Embedded data matrix of size `(num_delays + 1, length(x) - num_delays * τ)`

# Example

```julia
x = collect(1.0:10.0)
X_embed = delay_embedding(x, 2)
# Returns 3×8 matrix
```
"""
function delay_embedding(x::AbstractVector, num_delays::Integer; τ::Integer = 1)
    X = reshape(x, 1, length(x))
    return delay_embedding(X, num_delays; τ = τ)
end

"""
    $(SIGNATURES)

Compute a truncated singular value decomposition of data matrix `X`.

# Arguments

  - `X`: Data matrix
  - `rank`: Number of singular values/vectors to retain

# Returns

  - Named tuple `(U = U_r, S = S_r, V = V_r)` where:

      + `U_r` is the first `rank` columns of U
      + `S_r` is a vector of the first `rank` singular values
      + `V_r` is the first `rank` columns of V

# Example

```julia
X = randn(100, 50)
result = truncated_svd(X, 5)
X_approx = result.U * Diagonal(result.S) * result.V'
```

See also: [`reduce_dimension`](@ref), [`optimal_shrinkage`](@ref)
"""
function truncated_svd(X::AbstractMatrix, rank::Integer)
    @assert rank >= 1 "rank must be at least 1"
    max_rank = minimum(size(X))
    rank = min(rank, max_rank)

    U, S, V = svd(X)
    return (U = U[:, 1:rank], S = S[1:rank], V = V[:, 1:rank])
end

"""
    $(SIGNATURES)

Reduce the dimensionality of data by projecting onto the top singular vectors.

Projects data matrix `X` onto its first `rank` left singular vectors,
reducing the number of effective dimensions while preserving the most
important features.

# Arguments

  - `X`: Data matrix of size `(n_features, n_samples)`
  - `rank`: Number of dimensions to retain

# Returns

  - Reduced data matrix of size `(rank, n_samples)`
  - The reduced coordinates `Y = U_r' * X` where `U_r` contains the first `rank` left singular vectors

# Example

```julia
X = randn(100, 50)  # 100 features, 50 samples
X_reduced = reduce_dimension(X, 5)  # Reduce to 5 dimensions
# X_reduced is 5×50
```

See also: [`truncated_svd`](@ref), [`optimal_shrinkage`](@ref)
"""
function reduce_dimension(X::AbstractMatrix, rank::Integer)
    result = truncated_svd(X, rank)
    return result.U' * X
end

"""
    $(SIGNATURES)

Reduce the dimensionality of data using automatic rank selection.

Uses the optimal singular value hard threshold from Gavish & Donoho (2014)
to automatically determine the number of significant singular values,
then projects onto those dimensions.

# Arguments

  - `X`: Data matrix of size `(n_features, n_samples)`

# Returns

  - Reduced data matrix with automatically selected dimensionality

# Example

```julia
X = randn(100, 50) + 0.1 * randn(100, 50)  # Low-rank + noise
X_reduced = reduce_dimension(X)  # Automatically selects rank
```

See also: [`reduce_dimension(X, rank)`](@ref), [`optimal_shrinkage`](@ref)
"""
function reduce_dimension(X::AbstractMatrix)
    m, n = minimum(size(X)), maximum(size(X))
    U, S, V = svd(X)
    τ = optimal_svht(m, n)
    threshold = τ * median(S)
    rank = count(s -> s >= threshold, S)
    rank = max(rank, 1)  # Keep at least one dimension
    return U[:, 1:rank]' * X
end
