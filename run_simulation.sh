#!/bin/bash

# Exit immediately if any command fails
# set -e
clear

# 1. Clean up the 'run' directory at the start
echo "=========================================="
echo "Cleaning up the 'run' and 'results' directory..."
echo "=========================================="
if [ -d "run" ]; then
    # Removes everything inside 'run' but keeps the directory itself
    rm -rf run/*
else
    mkdir run
fi

if [ -d "results" ]; then
    # Removes everything inside 'results' but keeps the directory itself
    rm -rf results/*
else
    mkdir results
fi

# 2. Find all directories at the root level
for case_dir in *; do
    # Ensure it's a directory and not run, results, or another file
    if [ -d "$case_dir" ] && [ "$case_dir" != "run" ] && [ "$case_dir" != "results" ]; then
        
        # Check if the base_case folder exists before proceeding
        if [ ! -d "$case_dir/base_case" ]; then
            echo "Skipping $case_dir (No 'base_case' folder found)."
            continue
        fi

        echo ""
        echo "Processing case: $case_dir"
        echo "------------------------------------------"

        # 3. Iterate through subfolders in the case directory
        for variant_path in "$case_dir"/*; do
            if [ -d "$variant_path" ]; then
                variant_name=$(basename "$variant_path")

                # Exclude 'base_case' and any directory starting with an underscore (_)
                if [ "$variant_name" == "base_case" ] || [[ "$variant_name" == _* ]]; then
                    continue
                fi

                echo "  -> Found variant: $variant_name"

                # Define the target execution directory inside 'run/'
                target_run_dir="run/$case_dir/$variant_name"
                mkdir -p "$target_run_dir"

                # Step A: Copy all contents from base_case into the run directory
                # Using '/.' ensures hidden files (if any) are copied smoothly
                cp -r "$case_dir/base_case/." "$target_run_dir/"

                # Step B: Overwrite/add variant-specific dictionaries (e.g., turbulenceProperties)
                cp -r "$variant_path/." "$target_run_dir/"

                # 4. Execute the Allrun script in serial
                if [ -f "$target_run_dir/Allrun" ]; then
                    echo "  -> Running Allrun for $variant_name..."
                    
                    # Subshell execution (cd into target, run, then safely return automatically)
                    (
                        cd "$target_run_dir"
                        # Ensure Allrun has execution permissions
                        chmod +x Allrun
                        ./Allrun
                    )
                    
                    echo "  -> Finished $variant_name."
                else
                    echo "  -> [Warning] No Allrun script found in $target_run_dir"
                fi
            fi
        done

        # 5. Copy case-level post-processing scripts directly into run/<case>/ and execute them
        postproc_dir="$case_dir/_postProcessing"
        if [ -d "$postproc_dir" ]; then
            mkdir -p "run/$case_dir"

            for postproc_script in "$postproc_dir"/*.py; do
                [ -f "$postproc_script" ] || continue
                cp "$postproc_script" "run/$case_dir/"
            done

            run_postproc_script() {
                script_path="$1"
                script_name=$(basename "$script_path" .py)
                echo "  -> case_dir: $case_dir, script_name: $script_name, $script_path"
                if [ -f "$script_path" ]; then
                    source ../.venv/bin/activate
                    output_file="run/$case_dir/${script_name}.log"
                    echo "  -> Running post-processing script: $script_path, output will be saved to $output_file"
                    python3 "$script_path" > "$output_file" 2>&1
                fi
            }

            for script_name in postpro dashboard; do
                script_path="run/$case_dir/$script_name.py"
                if [ -f "$script_path" ]; then
                    run_postproc_script "$script_path"
                fi
            done

            for script_path in "run/$case_dir"/*.py; do
                [ -f "$script_path" ] || continue
                script_name=$(basename "$script_path" .py)
                if [ "$script_name" != "postpro" ] && [ "$script_name" != "dashboard" ]; then
                    run_postproc_script "$script_path"
                fi
            done
        fi
    fi
done

echo ""
echo "=========================================="
echo "All simulations completed successfully!"
echo "=========================================="
