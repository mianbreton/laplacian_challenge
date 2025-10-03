#!/bin/bash

set -e # Exit immediately if a failure is detected

echo "=== Running All Laplacian Binaries with Varying OMP_NUM_THREADS ==="
echo "Timestamp: $(date)"
echo ""

# Directories to search for executables
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
technologies=("native" "kokkos_cpu" "kokkos_gpu")
proc_binds=("false" "true" "spread")
#proc_binds=("false")
# Output log file
LOG_FILE="run_all_results_with_threads.log"
> "$LOG_FILE"

# Thread counts to test (adjust to match your machine's CPU count)
NCPUs=(1 2 4 8 16)
NCELLS=(32 64 128 256)
runs=10

# Loop over each thread count
for threads in "${NCPUs[@]}"; do
    export OMP_NUM_THREADS=$threads
    for proc_bind in "${proc_binds[@]}"; do
        export OMP_PROC_BIND=$proc_bind
        echo "--- OMP_NUM_THREADS=$threads --- OMP_PROC_BIND=$proc_bind"
        echo "# --- OMP_NUM_THREADS=$threads --- OMP_PROC_BIND=$proc_bind" >> "$LOG_FILE"
        for technology in "${technologies[@]}"; do 
            for exe in "${script_dir}/bin/laplacian_${technology}"*; do
                [[ -x "$exe" ]] || continue
                    echo "------------- Running $exe -------------"
                    echo "" >> "$LOG_FILE"
                    echo "------------- Running $exe -------------" >> "$LOG_FILE"
                    $exe --ncells ${NCELLS[@]} --runs $runs --basename $script_dir/timings/runtime_BIND_${proc_bind}_NCPUs_${threads} >> $LOG_FILE
            done
        done
        echo ""
    done
done

echo "All runs completed. Results saved to $LOG_FILE."
