Complete the following exercises. Each exercise focuses on a different high performance computing problem in scientific computing, data science, or artificial intelligence.

Your implementations must include a serial baseline and at least one parallel implementation. The main purpose of this assignment is to demonstrate, through reproducible experiments, how parallelism improves serial runs or, when improvement is limited, to explain clearly why that happens.

Use Python multiprocessing, mpi4py, or both, according to the requirements of each exercise. Measure performance carefully. At minimum, report execution time, problem size, number of workers or processes, and a brief analysis of speedup and efficiency whenever parallelization is applied.

Use reproducible experiments. Fix random seeds when needed, document the execution environment, and describe any assumptions, simplifications, or preprocessing steps.

**Deliverables**
One folder per exercise, for example exercise_3, and exercise_4. Each folder must contain the corresponding source code, scripts, configuration files, and any small supporting files required to reproduce the experiments.

1. Any figures, tables, logs, or exported outputs that support the report, organized either inside the corresponding exercise folder or inside docs/assets.

2. A requirements file or environment description that allows the work to be reproduced on another machine.

**Exercise 3. Forest Fire Cellular Automaton Driven by NASA FIRMS Data**

Context

Cellular automata are useful models for the study of complex spatiotemporal systems. Forest fire propagation is a natural application because the state of each location depends on local neighborhood interactions, available fuel, and external ignition events.

In this exercise, you will use hotspot detections from NASA FIRMS as the data source to initialize or guide a two-dimensional forest fire automaton. The objective is to combine real satellite detections with a simplified local propagation model and analyze the computational behavior of the simulation when the domain is parallelized.

This project connects scientific computing, geospatial data processing, and distributed simulation. Suggested data and documentation: 
NASA FIRMS main portal: https://firms.modaps.eosdis.nasa.gov/
FIRMS API: https://firms.modaps.eosdis.nasa.gov/api/
FIRMS API Python guide: https://firms.modaps.eosdis.nasa.gov/academy/data_api/

Model Description

Define a two-dimensional grid in which each cell has a discrete state. A suggested state model is the following: 0 for non-burnable or outside the valid domain, 1 for susceptible vegetation, 2 for burning, and 3 for burned.

NASA FIRMS detections may be mapped onto the grid as ignition points. A simple propagation rule can be defined so that a susceptible cell becomes burning when one or more neighboring cells are burning. The ignition probability may depend on the number of burning neighbors, the fire radiative power of the nearest detection, or a simplified local intensity rule.

At each time step, burning cells consume fuel and then transition to the burned state after a specified lifetime or fuel threshold. This yields a discrete-time dynamical system driven by local interaction plus externally observed hotspot data.

For parallel execution with mpi4py, partition the grid into subdomains and exchange boundary data between neighboring processes at each iteration.

Tasks

1. Obtain hotspot data from NASA FIRMS for a selected region and time window. Document the source, filtering criteria, and variables used.

2. Transform the hotspot detections into a regular two-dimensional grid suitable for a cellular automaton.

3. Define the state space, neighborhood structure, ignition rule, and state transition logic of the automaton.

4. Implement a serial simulation and verify that the model evolves coherently over time.

5. Implement a parallel version with mpi4py using domain decomposition. Describe how the grid is split and how border information is exchanged.

6. Run simulations for different grid sizes or time horizons and compare serial and parallel runtimes.

7. Visualize the temporal evolution of the fire, either as selected snapshots or as an animation.

8. Discuss the interpretation of NASA FIRMS data in the model and explain the difference between hotspot detections and the true fire perimeter.

9. Reflect on the scientific limitations of the simplified automaton and suggest possible improvements.

**Exercise 4. Parallel K-Means Clustering**

Context

K-means is one of the most widely used unsupervised learning algorithms in data science. Although its update rule is conceptually simple, its repeated distance computations can become expensive when the dataset is large.

In this exercise, you will implement K-means on the Covertype dataset and study how the algorithm can be parallelized with distributed-memory communication. The focus is on how the work can be split across processes and how partial cluster statistics can be aggregated efficiently.

This project connects machine learning with core ideas from high performance computing, especially data partitioning, collective communication, and iterative synchronization. Suggested data source: 
Covertype dataset: https://archive.ics.uci.edu/ml/datasets/covertype

Model Description

Given a dataset X = {x_1, ..., x_N} in R^d and a chosen number of clusters k, K-means minimizes the within-cluster sum of squares by alternating between two steps: assignment and update.

In the assignment step, each observation is assigned to the nearest centroid according to Euclidean distance. In the update step, each centroid is replaced by the mean of the observations assigned to its cluster.

In a parallel implementation, each process can work on a local block of observations, compute local assignments, and accumulate local sums and counts for each cluster. These partial results are then combined through collective communication to update the global centroids.

Tasks

1. Load the Covertype dataset and describe its size, number of features, and any preprocessing steps applied.

2. Implement a serial K-means baseline and validate the correctness of the assignments and centroid updates.

3. Implement a parallel K-means version with mpi4py. Explain how the dataset is distributed and how centroids are synchronized at each iteration.

4. Use collective communication operations to aggregate local cluster statistics and update the centroids.

5. Test the algorithm with different numbers of clusters and different numbers of processes.

6. Measure runtime per iteration, total runtime, convergence behavior, and any changes in clustering quality or stability.

7. Compare the serial and parallel versions and discuss under which conditions the parallel approach becomes advantageous.

8. Identify the main communication costs and discuss how dataset size and number of clusters influence scalability.

**NOTES**

All figures, tables, and code fragments included must be readable and correctly labeled.

Whenever a design choice is made, explain why it is technically reasonable for the problem and for the selected parallel strategy.

The final repository and report should make the performance comparison easy to verify. The reader should be able to identify the serial baseline, the parallel implementation, the execution settings, and the resulting improvement or limitation in each exercise.