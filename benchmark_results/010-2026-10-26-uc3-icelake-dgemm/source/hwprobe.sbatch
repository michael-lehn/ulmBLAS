#!/bin/bash
#  hwprobe.sbatch: record the hardware and software environment of one
#  compute node (no benchmarks).  Submit with the partition on the command
#  line, e.g.
#
#      sbatch -p dev_cpu_il hwprobe.sbatch
#      sbatch -p dev_cpu    hwprobe.sbatch
#
#  Output: hwprobe-PARTITION-JOBID.txt in the submission directory.
#SBATCH --job-name=hwprobe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000mb
#SBATCH --time=00:05:00
#SBATCH --output=hwprobe-%x-%j.out

OUT=$SLURM_SUBMIT_DIR/hwprobe-$SLURM_JOB_PARTITION-$SLURM_JOB_ID.txt
exec > "$OUT" 2>&1
sec() { echo; echo "=== $*"; }

sec job;     echo "host $(hostname) partition $SLURM_JOB_PARTITION job $SLURM_JOB_ID date $(date -Iseconds)"
             echo "cpus_on_node $SLURM_CPUS_ON_NODE job_cpus $SLURM_JOB_CPUS_PER_NODE"
             echo "affinity: $(taskset -cp $$)"
             grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
sec os;      uname -r; grep PRETTY /etc/os-release
sec lscpu;   lscpu
sec lscpu-e; lscpu -e=CPU,NODE,SOCKET,CORE,L1d:L1i:L2:L3,ONLINE,MAXMHZ,MINMHZ
sec cpuinfo; awk '/^$/{exit} {print}' /proc/cpuinfo
sec caches
for d in /sys/devices/system/cpu/cpu0/cache/index*; do
    echo "L$(cat $d/level) $(cat $d/type) size=$(cat $d/size) ways=$(cat $d/ways_of_associativity) line=$(cat $d/coherency_line_size) sets=$(cat $d/number_of_sets) shared=$(cat $d/shared_cpu_list)"
done
sec numa;    numactl -H 2>&1 || cat /sys/devices/system/node/node*/cpulist
sec freq
for f in scaling_driver scaling_governor scaling_min_freq scaling_max_freq cpuinfo_max_freq base_frequency; do
    echo "$f: $(cat /sys/devices/system/cpu/cpu0/cpufreq/$f 2>/dev/null)"
done
echo "intel_pstate no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"
echo "cpufreq boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
echo "amd_pstate status: $(cat /sys/devices/system/cpu/amd_pstate/status 2>/dev/null)"
echo "smt: $(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"
echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
echo "thp: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
sec load;    cat /proc/loadavg; who | wc -l
sec compilers
module purge >/dev/null 2>&1
for m in compiler/gnu/14.2 compiler/gnu/15.2 compiler/llvm/22.1; do
    module load $m >/dev/null 2>&1
    case $m in *llvm*) c=clang;; *) c=gcc;; esac
    echo "-- $m: $($c --version | head -1)"
    $c -march=native -### -E - < /dev/null 2>&1 | tr ' ' '\n' | grep -E '^"?-(march|mtune|mprefer-vector-width)' | tr '\n' ' '; echo
    module unload $m >/dev/null 2>&1
done
module load compiler/gnu/14.2 >/dev/null 2>&1
echo "gcc 14.2 -march=native: $(gcc -march=native -Q --help=target 2>/dev/null | grep -E '^ +-m(arch|tune)=' | tr -s ' ' | tr '\n' ' ')"
echo "enabled ISA: $(gcc -march=native -Q --help=target 2>/dev/null | awk '$2=="[enabled]"' | grep -oE '\-m(avx[0-9a-z_]*|fma|amx[a-z0-9-]*)' | tr '\n' ' ')"
sec tools;   for t in perf likwid-topology lstopo numactl hwloc-ls turbostat; do printf '%s: %s\n' $t "$(command -v $t)"; done
