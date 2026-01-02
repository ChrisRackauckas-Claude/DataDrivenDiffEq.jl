using DataDrivenDiffEq
using DataDrivenDiffEq: collocate_data
using LinearAlgebra
using Statistics

@testset "Optimal Shrinkage" begin
    t = collect(-2:0.01:2)
    U = [cos.(t) .* exp.(-t .^ 2) sin.(2 * t)]
    S = Diagonal([2.0; 3.0])
    V = [sin.(t) .* exp.(-t) cos.(t)]
    A = U * S * V'
    σ = 0.5
    Â = A + σ * randn(401, 401)
    n_1 = norm(A - Â)
    B = optimal_shrinkage(Â)
    optimal_shrinkage!(Â)
    @test norm(A - Â) < n_1
    @test norm(A - B) == norm(A - Â)
end

@testset "Collocation" begin
    x = 0:0.1:10.0
    y = permutedims(x)
    z = ones(1, length(x))
    # This list does not cover all kernels since some 
    # are singular
    for m in [
        EpanechnikovKernel(),
        UniformKernel(),
        TriangularKernel(),
        GaussianKernel(),
        LogisticKernel(),
        SigmoidKernel(),
        SilvermanKernel()
    ]
        ẑ, ŷ, x̂ = collocate_data(y, x, m)
        @test ẑ≈z atol=1e-1 rtol=1e-1
        @test ŷ≈y atol=1e-1 rtol=1e-1
        @test x̂≈x atol=1e-1 rtol=1e-1
    end

    x = 0:0.1:10.0
    y = permutedims(sin.(x))
    z = permutedims(cos.(x))

    for m in InterpolationMethod.([
        LinearInterpolation,
        QuadraticInterpolation,
        CubicSpline
    ])
        ẑ, ŷ, x̂ = collocate_data(y, x, m)
        @test ẑ≈z atol=1e-1 rtol=1e-1
        @test ŷ≈y atol=1e-1 rtol=1e-1
        @test x̂≈x atol=1e-1 rtol=1e-1
    end
end

@testset "Hankel Matrix" begin
    # Test basic Hankel matrix construction
    x = collect(1.0:10.0)
    H = hankel_matrix(x, 4)

    @test size(H) == (4, 7)
    # First column should be [1, 2, 3, 4]
    @test H[:, 1] == [1.0, 2.0, 3.0, 4.0]
    # Last column should be [7, 8, 9, 10]
    @test H[:, end] == [7.0, 8.0, 9.0, 10.0]
    # Check Hankel structure: constant along anti-diagonals
    @test H[1, 2] == H[2, 1]
    @test H[2, 3] == H[3, 2]
    @test H[1, 4] == H[4, 1]

    # Edge case: num_rows = 1
    H1 = hankel_matrix(x, 1)
    @test size(H1) == (1, 10)
    @test vec(H1) == x

    # Edge case: num_rows = length(x)
    Hmax = hankel_matrix(x, 10)
    @test size(Hmax) == (10, 1)
    @test vec(Hmax) == x

    # Test with Float32
    x32 = Float32.(x)
    H32 = hankel_matrix(x32, 3)
    @test eltype(H32) == Float32
end

@testset "Delay Embedding" begin
    # Test with 2D matrix input
    X = reshape(1.0:20.0, 2, 10)  # 2 states, 10 samples
    X_embed = delay_embedding(X, 2)

    # Should have 2 * (2 + 1) = 6 rows, 10 - 2 = 8 columns
    @test size(X_embed) == (6, 8)

    # Check structure: first 2 rows are current time, next 2 are t-1, etc.
    # Row 1 should be state 1 at current time: X[1, 3:10]
    @test X_embed[1, :] == X[1, 3:10]
    # Row 3 should be state 1 at t-1: X[1, 2:9]
    @test X_embed[3, :] == X[1, 2:9]
    # Row 5 should be state 1 at t-2: X[1, 1:8]
    @test X_embed[5, :] == X[1, 1:8]

    # Test with delay interval τ = 2
    X_embed_tau2 = delay_embedding(X, 2, τ = 2)
    # Should have 6 rows, 10 - 2*2 = 6 columns
    @test size(X_embed_tau2) == (6, 6)

    # Test with 1D vector input
    x = collect(1.0:10.0)
    x_embed = delay_embedding(x, 2)
    @test size(x_embed) == (3, 8)
    @test x_embed[1, :] == x[3:10]
    @test x_embed[2, :] == x[2:9]
    @test x_embed[3, :] == x[1:8]

    # Test num_delays = 0 (no embedding)
    x_no_embed = delay_embedding(x, 0)
    @test size(x_no_embed) == (1, 10)
    @test vec(x_no_embed) == x
end

@testset "Truncated SVD" begin
    # Create a low-rank matrix
    U_true = randn(50, 3)
    V_true = randn(30, 3)
    S_true = [10.0, 5.0, 1.0]
    X = U_true * Diagonal(S_true) * V_true'

    # Truncate to rank 2
    result = truncated_svd(X, 2)
    @test length(result.S) == 2
    @test size(result.U) == (50, 2)
    @test size(result.V) == (30, 2)

    # Check that singular values are in decreasing order
    @test result.S[1] >= result.S[2]

    # Reconstruction should be close to best rank-2 approximation
    X_approx = result.U * Diagonal(result.S) * result.V'
    _, S_full, _ = svd(X)
    @test norm(X - X_approx) ≈ S_full[3] atol = 1e-10

    # Test rank larger than matrix dimension (should be clamped)
    result_large = truncated_svd(X, 100)
    @test length(result_large.S) == min(50, 30)
end

@testset "Reduce Dimension" begin
    # Create a low-rank matrix with noise
    U_true = randn(100, 5)
    V_true = randn(50, 5)
    S_true = [100.0, 50.0, 10.0, 5.0, 1.0]
    X_true = U_true * Diagonal(S_true) * V_true'
    X = X_true + 0.01 * randn(100, 50)

    # Reduce to specified rank
    X_reduced = reduce_dimension(X, 3)
    @test size(X_reduced) == (3, 50)

    # Verify the reduced data can be projected back to original space
    U, _, _ = svd(X)
    X_reconstructed = U[:, 1:3] * X_reduced
    # Reconstruction error should be similar to what we get from best rank-3 approximation
    _, S_full, _ = svd(X)
    theoretical_error = sqrt(sum(S_full[4:end] .^ 2))
    @test norm(X - X_reconstructed) ≈ theoretical_error atol = 1e-10

    # Test automatic rank selection
    X_auto = reduce_dimension(X)
    # Should have between 1 and min(100, 50) dimensions
    @test 1 <= size(X_auto, 1) <= 50
    # Number of samples should be preserved
    @test size(X_auto, 2) == 50
end
