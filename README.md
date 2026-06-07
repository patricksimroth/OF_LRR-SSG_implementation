# Implementation and validation of the SSG/LRR Reynolds stress model in OpenFOAM

## Introduction

This repository aims to implement and validate the SSG/LRR Reynolds stress model in OpenFOAM. The SSG and LRR models are already implemented in OpenFOAM and will be used as a basis for the implementation of the SSG/LRR model. The implementation will take place in multiple steps:

1. **Simple Test Cases**: The first step will be to implement the SSG/LRR model in simple test cases, such as backward-facing step and flat plate. This will allow us to validate the implementation and ensure that it is working correctly.

2. **2D Airfoil profile**: The next step will be to implement the SSG/LRR model in a 2D airfoil profile. 

3. **3D Swept wing**: The final step will be to implement the SSG/LRR model in a 3D swept wing to investigate its predictions of shock buffets and compare them with experimental data.


## Repository overview

The repository is structured as follows:
- `backward_facing_step`: This directory contains the case setup for the backward-facing step test case. It includes reference data in the `_validation_data` directory and post processing scripts in the `post_processing` directory. The `base_case` directory contains the base case setup for the backward-facing step test case. The other directories (`kw-SST`, `openfoam`, `SSG`, and `SA`) contain the case setups for the different turbulence models, which will be copied to the `run` directory when executing the `run_simulation.sh` script.
- `run`: This directory is created when executing the `run_simulation.sh` script. It contains the results of all the simulations for all the available test cases and each model. 
- `results`: This directory contains the results of the simulations. The user can analyze the results, using the `index.html` file, which summarizes the results and provides links to log files and plots.



## Running the code

To run the code, the user needs to execute:

```bash
./run_simulation.sh
```

This script will run the simulations for all the available test cases and each model, by copying them to the `run` directory. The results will be stored in the `results` directory, where the user can analyze and compare the results with experimental data, using the `index.html` file. 

## References

