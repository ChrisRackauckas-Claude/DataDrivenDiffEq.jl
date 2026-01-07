using SafeTestsets, Test, Pkg

@info "Finished loading packages"

const GROUP = get(ENV, "GROUP", "All")

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    return Pkg.develop(PackageSpec(path = subpkg_path))
end

function activate_subpkg_env(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.activate(subpkg_path)
    Pkg.develop(PackageSpec(path = subpkg_path))
    return Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "Core" || GROUP == "Downstream"
        @testset "All" begin
            @safetestset "Basis" begin
                include("./basis/basis.jl")
            end
            @safetestset "Implicit Basis" begin
                include("./basis/implicit_basis.jl")
            end
            @safetestset "Basis generators" begin
                include("./basis/generators.jl")
            end
            @safetestset "DataDrivenProblem" begin
                include("./problem/problem.jl")
            end
            @safetestset "DataDrivenSolution" begin
                include("./solution/solution.jl")
            end
            @safetestset "Utilities" begin
                include("./utils.jl")
            end
            @safetestset "CommonSolve" begin
                include("./commonsolve/commonsolve.jl")
            end
        end

        # Run JET analysis tests only in CI or when GROUP explicitly includes JET
        # Note: JET tests may take longer due to static analysis overhead
        if get(ENV, "CI", "false") == "true" || get(ENV, "RUN_JET_TESTS", "false") == "true"
            @safetestset "JET Static Analysis" begin
                include("./jet_tests.jl")
            end
        end
    else
        dev_subpkg(GROUP)
        subpkg_path = joinpath(dirname(@__DIR__), "lib", GROUP)
        Pkg.test(PackageSpec(name = GROUP, path = subpkg_path); coverage = true)
    end
end
