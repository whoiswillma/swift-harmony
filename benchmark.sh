function benchmark {
    echo "$1 ie_no_opt"
    time ./sharm-debug ie $1
    echo ''

    echo "$1 ie_with_opt"
    time ./sharm ie $1
    echo ''

    ./sharm ce $1
    cd sharmcompile

    swiftc *.swift -o main
    ./main
    echo "$1 ce_no_opt"
    time ./main
    echo ''

    swiftc -O -wmo *.swift -o main
    ./main
    echo "$1 ce_with_opt"
    time ./main
    echo ''

    cd ..
}

function benchmark_mc {
    echo "$1 imc_no_opt"
    time ./sharm-debug imc --silent $1
    echo ''

    echo "$1 imc_with_opt"
    time ./sharm imc --silent $1
    echo ''

    ./sharm cmc --silent $1
    cd sharmcompile

    swiftc *.swift -o main
    ./main
    echo "$1 cmc_no_opt"
    time ./main
    echo ''

    swiftc -O -wmo *.swift -o main
    ./main
    echo "$1 cmc_with_opt"
    time ./main
    echo ''

    cd ..
}

swift build
swift build -c release

#benchmark benchmarks/bs_exec.hvm
#benchmark benchmarks/qs_exec.hvm
#benchmark benchmarks/ss_exec.hvm

benchmark_mc benchmarks/bs_mc.hvm
benchmark_mc benchmarks/qs_mc.hvm
benchmark_mc benchmarks/ss_mc.hvm
benchmark_mc DinersFixed.hvm

