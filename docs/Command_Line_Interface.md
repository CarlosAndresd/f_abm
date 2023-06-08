
## Use of the Command Line Interface

After executing the `start.py` Python script you will enter a Command Line Interface, from there you will be able to 
configure and run different simulations. This is useful because you can not only configure multiple simulations at once
but they can be run in a HPC.

This tutorial will provide a step by step guide organised in 'Parts'. From the most basic use to advance configuration. 
Feel free to jump to the Part most convenient to you. The tutorial also contains all the commands, an explanation, and 
the expected results.

## Part 1: First simulation using the default values

After executing the Python program `start.py` you will be asked to enter 8 simulation parameters. All these options have default values
which we will use in this first part. Some of the defaut options can be seen in the square brackets.

So feel free to press `ENTER` 8 times. You will see something like this:

``
 
    1. Enter name of the new simulation [Simulation-20230608133600]: (Press Enter)
    2. Enter directory where results are saved [simulation_results]:  (Press Enter)
    3. Enter number of agents [100]:  (Press Enter)
    4. Enter initial opinion characterisation [io_loc=(0.5, 0.1); io_prt=True]: (Press Enter) 
    5. Enter model [mod_lab="CB"]:  (Press Enter)
    6. Enter agent parameter characterisation [par_rep=(0.2, 0.3, 0.5); par_prt=True]: (Press Enter) 
    7. Enter underlying digraph characterisation [dig_lab="sw"; dig_tsi=[0, 1, 1, 1]; dig_cpr=0.5; dig_prt=True]: 
    8. Enter number of time-step [50]: 








	Creating initial opinions
	Initial opinions created


	Creating adjacency matrix
	Creating a Small-World Digraph with 100 agents, 1.0 positive edge ratio
	Adjacency matrix created


	Creating agent parameters
	Agent parameters created


	Running the model
	Simulation complete
``


## Part 2: Basic modifications: name, directory, number of agents, number of time steps


## Part 3: Configuration of initial opinions


## Part 4: Configuration of agent parameters


## Part 5: Configuration of underlying digraph


## Part 6: Configuration of simulation model


## Part 7: Configuration of multiple simulations
