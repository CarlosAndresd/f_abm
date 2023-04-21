
# Agent-Based Model for Opinion Formation

by Carlos Andres Devia

## Computational Environment:

- Programming language: Python
- Packages and libraries: see the `requirements.txt` file

## How to use this code repository

### Installation instructions

#### Installation instructions using conda (recommended)

Before installing this code, make sure to have conda installed, you can use the command `conda --version` on the terminal. If you don't have conda, you can download Miniconda it from [here](https://docs.conda.io/en/latest/miniconda.html) (Miniconda will contain conda), and follow [these installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)

1. Fork and clone this repository
2. In a terminal, move to the directory containing the repository
3. Run the following command:

```
conda env create --file environment.yml
```

4. Run the command`conda env list` you should see "f_abm" as one of the listed environments.

#### Installation instructions using python

1. Fork and clone this repository
2. Create a virtual environtment and install the rependencies usign the following commands (if you face any problems, see the [official documentation](https://docs.python.org/3/library/venv.html)):

```
python3 -m venv ./venv  % Create the virual environment
source ./venv/bin/activate  % Activate the virtual environment
pip install -r requirements.txt  % Install the dependencies from the requirements.txt
```


### Tests

The code includes a script to run tests and check that the code is working as intended. To run the test, execute the script `code_test.py` by typing in the terminal (while in the project directory)

```
python3 src/code_test.py
```

After executing this code, a histogram should appear, and also file called 'example_opinions.npy'.


### Examples


