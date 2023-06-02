### Computational Environment:

- Programming language: Python
- Packages and libraries: see the `requirements.txt` file or the `environment.yml` file

## How to use this code repository

To get started, you can follow two steps:

1. Install the package, there are two options: with Conda (recommended), and with pip and venv.
2. Execute the program

### Installation instructions with Conda (recommended)

Before installing this code, make sure to have conda installed, you can use the command `conda --version` on the terminal. If you don't have conda, you can download Miniconda it from [here](https://docs.conda.io/en/latest/miniconda.html) (Miniconda will contain conda), and follow [these installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)

1. Fork and clone this repository.
2. In a shell, move to the directory containing the repository.
3. Run the following command:

```
conda env create --file environment.yml
```

4. Run the command `conda env list` you should see "f_abm_env" as one of the listed environments.

If you wish to delete the environment you can type 

```
conda remove --name f_abm_env --all
```

You can verify that the environment was deleted by typing

```
conda info --f_abm_env
```


To activate the conda virtual environment type `conda activate f_abm_env`, to deactivate it type `conda deactivate`.


### Installation instructions with pip and venv

**Note 1:** for this to work you need to make sure you have `pip` and `venv` installed in your computer. You can find [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) more information about `pip` and virtual environments.

**Note 2:** these instructions assume you have `Python 3` installed in your computer. 

If you are unsure your computer has these requirements, please follow the recommended installation instructions. 


1. Fork and clone this repository
2. Create a virtual environtment and install the rependencies usign the following commands (if you face any problems, see the [official documentation](https://docs.python.org/3/library/venv.html)):

```
python3 -m venv ./f_abm_env  % Use 'venv' to create a virual environment called 'f_abm_env'
source ./f_abm_env/bin/activate  % Activate the virtual environment
pip install -r requirements.txt  % Install the dependencies from the requirements.txt
```

3. A file called 'f_abm_env' should appear now in the current directory.

If you wish to delete the environment you can type 

```
rm -r f_abm_env
```

To activate the venv virtual environment type `source ./f_abm_env/bin/activate`, to deactivate it type `deactivate`.


### Program execution

1. Make sure you are in the main project directory, which is called `f_abm`

2. In order to use the program, it is first necessary to activate the virtual environment, if the installation was done using Conda type:

```
conda activate f_abm_env
```

to deactivate the Conda environment you can type `conda deactivate`

If the installation was done using `venv` type:

```
source ./f_abm_env/bin/activate
```

to deactivate the virtual environment you can type `deactiate`

3. To start the program type 


```
python start.py
```

and follow the instructions, all the options have default parameters, so feel free to press enter to all the questions.

4. If there were no mistakes, you can go to the directory `./f_abm/simulation_results`, there you should find a directory with the name `Simulation-YYYYMMDDHHMMSS` that contains three images, resulting from the simulation.









### Tests

The code includes a script to run tests and check that the code is working as intended. To run the test, execute the script `code_test.py` by typing in the terminal (while in the project directory)

```
python3 src/code_test.py
```

After executing this code, a histogram should appear, and also file called 'example_opinions.npy'.
