from multiprocessing import Pool
import os


# ROOT = "/scratch/jelleluijkx/eagerx_interbotix"
ROOT = "/home/jelle/eagerx_dev/eagerx_interbotix"


def generate_command(delay_min, delay_max):
    return f"python {ROOT}/examples/finetune.py --delay-min {delay_min} --delay-max {delay_max}"

if __name__ == '__main__':
    delays_min = [0.1, 0.4]
    delays_max = [0.1, 0.2, 0.3, 0.4]

    n_cpus = os.cpu_count()
    pool = Pool(n_cpus)

    commands = []
    for delay_min in delays_min:
        for delay_max in delays_max:
            if delay_min > delay_max:
                continue
            commands.append(generate_command(delay_min, delay_max))
    pool.map(os.system, commands)
