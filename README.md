
# Agent-Based Model for Opinion Formation

by Carlos Andres Devia

- [Documentation](https://f-abm.readthedocs.io/en/latest/)
- [Code of Conduct](https://github.com/CarlosAndresd/f_abm/blob/main/CODE_OF_CONDUCT.md)
- [Contributing](https://github.com/CarlosAndresd/f_abm/blob/main/CONTRIBUTING.md)

## Computational Environment:

- Programming language: Python
- Packages and libraries: see the `requirements.txt` file or the `environment.yml` file

## How to use this code repository

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


To activate the conda virtual environment type `conda activate f_abm`, to deactivate it type `conda deactivate`.


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


### Tests

The code includes a script to run tests and check that the code is working as intended. To run the test, execute the script `code_test.py` by typing in the terminal (while in the project directory)

```
python3 src/code_test.py
```

After executing this code, a histogram should appear, and also file called 'example_opinions.npy'.


### Examples


