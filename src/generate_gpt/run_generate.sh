#!/bin/bash

command="python blocks/datasets/generate_dataset_green_gpt.py"
num_instances=${1:-1}  # Default to 1 instance if not specified

run_instance() {
    while true; do
        $command &
        pid=$!

        while kill -0 $pid 2>/dev/null; do
            sleep 5
        done

        echo "Instance $1: Command was terminated. Restarting..."
    done
}

for i in $(seq 1 $num_instances); do
    run_instance $i &
done

wait