# LIFT

## Logistics Infrastructure & Fleet Transformation


## Created by 
Brian Dietermann, M.Sc. and Anna Paper, M.Sc.  
Institute of Automotive Technology  
Department of Mobility Systems Engineering  
TUM School of Engineering and Design  
Technical University of Munich  
anna.paper@tum.de  
2025

#### Contributors  
Fabian Mayer, B.Sc. - Semester Thesis ongoing

## Installation

#### Step 1: Getting the source code
LIFT is available on [GitLab](https://gitlab.lrz.de/energysystemmodelling/lift) and can be cloned from there using 
```bash
git clone https://gitlab.lrz.de/energysystemmodelling/lift.git
```

#### Step 2: Create a clean virtual environment
It is recommended to create and activate a clean virtual environment for the installation of LIFT.
This can be done using conda:
```bash
conda create -n <name_of_virtual_environment> python=3.11
conda activate <name_of_virtual_environment>
```
or alternatively with the following command:
```bash
python -m venv <path_to_virtual_environment>
source <path_to_virtual_environment>/bin/activate
```

#### Step 3: Install package and dependencies locally
After cloning the repository, navigate to its root directory (where ```README.md``` and ```pyproject.toml``` are located) in your terminal.
Then install the package and its dependencies using one of the following commands depending on the chosen mode of installation:
##### a) Standard Installation
This copies the package into your (virtual environment’s) site-packages directory:
```bash
pip install .
```
After pulling new changes from the repository, the package has to be reinstalled using the same command to take the changes into account.

##### b) Editable Installation (recommended for development)

This links the package to your local source code, so any changes (you make or pulled from the repository) are immediately reflected without reinstalling:
```bash
pip install -e .
```
Use the editable mode if you plan to modify the code during development.


## Basic Usage - GUI mode
LIFT can be run using one of two terminal commands, given the correct virtual environment is activated:
1. Call to the main module: ```python -m lift``` (best for local execution on host machine, e.g. through a run configuration in PyCharm)
2. Call to the entry point: ```lift``` (best for remote execution on a server as it works irrespective of the current working directory as long as the correct environment is active)

## Advanced Usage - GUI mode
LIFT includes predefined options for vehicles, chargers and other assumptions. Those can be found in ```lift/definitions/*```.
To create a new type of charger or vehicle add a new entry to the dictionary in ```chargers.py``` or ```subfleets.py```, respectively.
If LIFT is installed in editable mode, the frontend will automatically include the changes made. 

## Basic Usage - Scalable mode
LIFT as being clearly divided into a frontend and backend also features the option to only use the backend without the frontend.
This allows for scalable multi-scenario investigations. For this, a ```lift.interfaces.Inputs``` instance has to be created and passed to ```lift.backend.backend.run_backend()```.
An example for a single scenario is given in ```scripts/run.py```.
