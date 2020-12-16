using ArgParse
using MAT
include("./admm_qap.jl")

function main(args)

    s = ArgParseSettings("argparse_example_2.jl");

    @add_arg_table s begin
        "--maxit"
            arg_type = Int
            default = 1000
        "--tol"
            arg_type = Float64
            default = 0.001
            help = "an option"     # used by the help screen
        "--gamma"
            arg_type = Float64
            default = 1.618
        "--verbose", "-v"
            action = :store_true   # this makes it a flag
            help = "verbose flag"
        "--ex"
            arg_type = String
            default = "nug12"
        "--lowrank", "-l"
            action = :store_true   # this makes it a flag
    end

    parsed_args = parse_args(args, s)
    println("Parsed args:")
    for pa in parsed_args
        println("  $(pa[1])  =>  $(pa[2])")
    end

    file = matread("../../data/" * parsed_args["ex"] * ".mat");
    A = file["A"];
    B = file["B"];
    println("Starting ...");
    _, Y = admm_qap(A, B, nothing, parsed_args);

end

main(ARGS)
