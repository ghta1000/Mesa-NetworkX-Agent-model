# Mesa-NetworkX-Tumor-Growth-Agent-model
Simulation and analysis of multiscale graph agent-based tumor model
# Table of contents
1. [Introduction](#introduction)
	1. [Dependencies](#Mesa)
	2. [Tumor Growth Simulation](#Tumor)
  3. [Documentation](#ref_doc)

# Introduction <a name="introduction"></a>
Python data visualization provides strong support for integration with several technologies and higher programming productivity across the development lifecycle.
In this Model, the front-end integration is the strength of Python modeling based on packages such as an ABM package (Mesa), analysis, and visualization packages (Numpy, SciPy, Matplotlib).
On the other hand, the Tumor simulation is getting more orchestrated while united with machine learning packages (NetworkX) or distribution utility modules and third-parties (Pypi) and container facilities deep at the back-end.

## Dependencies<a name="Mesa"></a>

* Python 3.0 
* Mesa is an Apache2 licensed agent-based modeling (or ABM) framework in Python.
https://mesa.readthedocs.io/en/latest/

* NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
https://networkx.org/documentation/stable/index.html

## Tumor Growth Simulation<a name="Tumor"></a>

The model has two core classes. One is the tumor model states, attributes, and scheduler component, and the other is the tumor age.
Four states in the tumor model class are defined as metastasis cells, dead cells, normal cells, and inflammatory cells.
The graph generator and the clustering method for sub-graphing are returned to the tumor model class.
The inflammation of the cellular neighborhood based on Angiogenic switch rules and the help of the Finite-state machine probabilities, all selected and verified by oncologists interactively to simulate the tumor’s growing scenario model is created at the tumor agents class.
In the Mesa framework, for studying the model’s behavior under different conditions, it has needed to collect the model’s relevant data while running in which the Data_Collector class is defined for this task. The tumor model’s data collection is in the tumor model class and extracts the CSV or XML files data.  

#### Documentation <a name="ref_doc"></a>

*Tashakor G, Suppi R. Simulation and computational analysis of multiscale graph agent-based tumor model. In2019 International Conference on High Performance Computing & Simulation (HPCS) 2019 Jul 15 (pp. 298-304). IEEE.

*Tashakor G., Suppi R. (2019). Simulation and analysis of a multi-scale tumor model using agent clustered network. Proceedings of the 18th International Conference on Modelling and Applied Simulation (MAS 2019), pp. 64-71. DOI: https://doi.org/10.46354/i3m.2019.mas.009
*
