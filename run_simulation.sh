#!/bin/bash

# Exit immediately if any command fails
# set -e
clear

# Helper: format elapsed seconds as HH:MM:SS
format_time() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

# Root-level directories that are not simulation cases (extend as needed)
EXCLUDED_DIRS=("run" "results" "src" "documentation")

# Parse command-line filters
RUN_ALL=true
script_start=$(date +%s)
FAILED_VARIANTS=()
FAILED_POSTPROC=()
if [ $# -gt 0 ]; then
    RUN_ALL=false
    FILTERS=()
    for arg in "$@"; do
        # Remove trailing slashes to handle inputs like "case/" or "case/model/"
        arg="${arg%/}"
        FILTERS+=("$arg")
    done
fi

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

# 1b. Build the custom SSG/LRR-omega turbulence model library (if source present)
if [ -d "src/SSGLRROmega2019" ]; then
    echo "=========================================="
    echo "Building custom turbulence model (SSGLRROmega2019)..."
    echo "=========================================="
    if [ -z "$WM_PROJECT_DIR" ]; then
        source /usr/lib/openfoam/openfoam2512/etc/bashrc
    fi
    (cd src/SSGLRROmega2019 && wmake libso) || echo "  [warn] custom model build failed"
fi

# 1c. Build the custom SALSA turbulence model library (if source present)
if [ -d "src/SpalartAllmarasSALSA" ]; then
    echo "=========================================="
    echo "Building custom turbulence model (SpalartAllmarasSALSA)..."
    echo "=========================================="
    if [ -z "$WM_PROJECT_DIR" ]; then
        source /usr/lib/openfoam/openfoam2512/etc/bashrc
    fi
    (cd src/SpalartAllmarasSALSA && wmake libso) || echo "  [warn] custom model build failed"
fi

# 2. Find all directories at the root level
for case_dir in *; do
    # Skip excluded directories (not simulation cases)
    is_excluded=false
    for excluded_dir in "${EXCLUDED_DIRS[@]}"; do
        if [ "$case_dir" == "$excluded_dir" ]; then
            is_excluded=true
            break
        fi
    done
    if [ "$is_excluded" = true ]; then
        continue
    fi

    # Ensure it's a directory
    if [ -d "$case_dir" ]; then
        
        # Check if the base_case folder exists before proceeding
        if [ ! -d "$case_dir/base_case" ]; then
            echo "Skipping $case_dir (No 'base_case' folder found)."
            continue
        fi

        # Determine if this case matches filters and has at least one matching variant
        case_has_matching_variant=false
        if [ "$RUN_ALL" = true ]; then
            case_has_matching_variant=true
        else
            for filter in "${FILTERS[@]}"; do
                case_filter="${filter%%/*}"
                if [[ "$case_dir" == $case_filter ]]; then
                    if [[ "$filter" != *"/"* ]]; then
                        case_has_matching_variant=true
                        break
                    else
                        model_filter="${filter#*/}"
                        IFS=',' read -r -a allowed_models <<< "$model_filter"
                        for variant_path in "$case_dir"/*; do
                            if [ -d "$variant_path" ]; then
                                variant_name=$(basename "$variant_path")
                                if [ "$variant_name" != "base_case" ] && [[ "$variant_name" != _* ]]; then
                                    for model in "${allowed_models[@]}"; do
                                        if [[ "$variant_name" == $model ]]; then
                                            case_has_matching_variant=true
                                            break 3
                                        fi
                                    done
                                fi
                            fi
                        done
                    fi
                fi
            done
        fi

        if [ "$case_has_matching_variant" = false ]; then
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

                # Determine if this variant matches any of the filter arguments
                variant_matched=false
                if [ "$RUN_ALL" = true ]; then
                    variant_matched=true
                else
                    for filter in "${FILTERS[@]}"; do
                        case_filter="${filter%%/*}"
                        if [[ "$case_dir" == $case_filter ]]; then
                            if [[ "$filter" != *"/"* ]]; then
                                variant_matched=true
                                break
                            else
                                model_filter="${filter#*/}"
                                IFS=',' read -r -a allowed_models <<< "$model_filter"
                                for model in "${allowed_models[@]}"; do
                                    if [[ "$variant_name" == $model ]]; then
                                        variant_matched=true
                                        break 2
                                    fi
                                done
                            fi
                        fi
                    done
                fi

                if [ "$variant_matched" = false ]; then
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
                     variant_start=$(date +%s)

                     # Subshell execution (cd into target, run, then safely return automatically)
                     (
                         cd "$target_run_dir"
                         # Ensure Allrun has execution permissions
                         chmod +x Allrun
                         ./Allrun
                     )
                     variant_rc=$?
                     variant_end=$(date +%s)
                     variant_elapsed=$((variant_end - variant_start))
                     if [ $variant_rc -ne 0 ]; then
                         echo "  -> [FAILED] $variant_name (exit $variant_rc) in $(format_time $variant_elapsed) — logs: run/$case_dir/$variant_name/log.*"
                         FAILED_VARIANTS+=("$case_dir/$variant_name")
                     else
                         echo "  -> Finished $variant_name in $(format_time $variant_elapsed) ($variant_elapsed s)."
                     fi
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
                    rc=$?
                    if [ $rc -ne 0 ]; then
                        echo "  -> [FAILED] $script_name.py (exit $rc) — log: $output_file"
                        FAILED_POSTPROC+=("$case_dir/$script_name.py")
                    fi
                fi
            }

            for script_name in postpro; do
                script_path="run/$case_dir/$script_name.py"
                if [ -f "$script_path" ]; then
                    run_postproc_script "$script_path"
                fi
            done

            for script_path in "run/$case_dir"/*.py; do
                [ -f "$script_path" ] || continue
                script_name=$(basename "$script_path" .py)
                [[ "$script_name" == _* ]] && continue
                if [ "$script_name" != "postpro" ] && [ "$script_name" != "dashboard" ]; then
                    run_postproc_script "$script_path"
                fi
            done

            if [ ${#FAILED_VARIANTS[@]} -gt 0 ]; then
                echo "  [warn] running post-processing despite failed variant(s): ${FAILED_VARIANTS[*]}"
            fi

            # dashboard runs last so it sees all images produced by the other scripts
            script_path="run/$case_dir/dashboard.py"
            if [ -f "$script_path" ]; then
                run_postproc_script "$script_path"
            fi
        fi
    fi
done

script_end=$(date +%s)
total_elapsed=$((script_end - script_start))
echo ""
echo "=========================================="
if [ ${#FAILED_VARIANTS[@]} -eq 0 ] && [ ${#FAILED_POSTPROC[@]} -eq 0 ]; then
    echo "All simulations completed successfully!"
else
    echo "Completed with failures:"
    [ ${#FAILED_VARIANTS[@]} -gt 0 ] && echo "  failed variants : ${FAILED_VARIANTS[*]}"
    [ ${#FAILED_POSTPROC[@]} -gt 0 ] && echo "  failed scripts  : ${FAILED_POSTPROC[*]}"
fi
echo "Total elapsed time: $(format_time $total_elapsed) ($total_elapsed s)"
echo "=========================================="
