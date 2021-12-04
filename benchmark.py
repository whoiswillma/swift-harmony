import timeit
import subprocess
import os

def run(cmd):
    print(cmd)
    subprocess.run(cmd.split(' '))


def build_sharm():
    run('swift build')
    run('swift build -c release')


def sharm_ce(prog, with_optimization=False):
    subprocess.run(
        f'./sharm ce {prog}',
        shell=True
    )

    if with_optimization:
        subprocess.run(
            'swiftc -O -wmo *.swift -o main',
            shell=True,
            cwd=f'{os.getcwd()}/sharmcompile'
        )
    else:
        subprocess.run(
            'swiftc *.swift -o main',
            shell=True,
            cwd=f'{os.getcwd()}/sharmcompile'
        )
    
    return f'{os.getcwd()}/sharmcompile/main'


def runtimes(runnable, number, warmup=0):
    runtimes = []

    for _ in range(warmup):
        runnable()
        print('*', end='', flush=True)

    for _ in range(number):
        runtimes.append(timeit.timeit(
            runnable,
            number=1
        ))
        print('.', end='', flush=True)
    return runtimes


def benchmark(prog, number=5, warmup=1):
    results = {}

    print('ie no opt: ', end='', flush=True)
    results['ie_no_opt'] = runtimes(
        lambda: subprocess.run(f'./sharm-debug ie {prog}', shell=True, capture_output=True), 
        number=number,
        warmup=warmup
    )
    print()

    print('ie with opt: ', end='', flush=True)
    results['ie_with_opt'] = runtimes(
        lambda: subprocess.run(f'./sharm ie {prog}', shell=True, capture_output=True), 
        number=number,
        warmup=warmup
    )
    print()

    exe = sharm_ce(prog, with_optimization=False)
    print('ce no opt: ', end='', flush=True)
    results['ce_no_opt'] = runtimes(
        lambda: subprocess.run(exe, shell=True, capture_output=True), 
        number=number,
        warmup=warmup
    )
    print()

    exe = sharm_ce(prog, with_optimization=True)
    print('ce with opt: ', end='', flush=True)
    results['ce_with_opt'] = runtimes(
        lambda: subprocess.run(exe, shell=True, capture_output=True), 
        number=number,
        warmup=warmup
    )
    print()

    return results


build_sharm()
print(benchmark('benchmarks/ss.hvm'))
print(benchmark('benchmarks/qs.hvm'))

