# History

## 0.4.0 (2025-09-26): Prepare article revision

* Added three notebooks for explicit support of https://hal.archives-ouvertes.fr/hal-03502084
* Reward-based policies like EGPD have a `alt_rewards` option to alter the rewards internally.
  Two alterations (`gentle` and `normalize`) are available.
* Default rewards are now edge-degree proportional
  to ensure default neutrality in hypergraphs.
* New policy: CRPD (Constant-Regret Primal-Dual),
  based on https://dl.acm.org/doi/10.1145/3578338.3593532 .
* New graphs in library: Nazari-Stolyar, Erdös-Rényi.
* Vis engine updated, with new default behavior (refresh button).
* Fix: Numba error when using forbidden edge without `k`.
* Chore: documentation files switched to markdown.
* Chore: package backend switched to uv.


## 0.3.3 (2024-07-15): Simplified and unified metric computation

* Extraction of metrics from simulation gathered in a unique submodule
* Most used metrics are now properties of the simulator object
* Pre-defined metrics can be selected by name on batch simulation
* Custom metrics can be used by passing their function

## 0.3.2 (2024-07-11): Improved tools for batched simulations

* Unified way to run batched of experiments
  * Construct experiments with static and variable parameters
  * Define how to extract the metrics you want
  * Start evaluation and see how it progresses with tqdm
* mutiprocess.Pool can be optionally used to parallelize the results
* Results can be optionally automatically cached
* Cf notebooks or reference for details

## 0.3.1 (2024-07-09): Improving the simulator

* Changes in the simulator API:
  * For k-filtering, the threshold parameter is now k
  * weights are now called rewards everywhere but for priority (to keep the weight/counterweight story)
  * Interleaving of rewards and forbidden edges has been improved (each can define the other if necessary)
  * reward-based policies are triggered by setting a beta parameter
* Introduction of in-package parallelization tools
* New notebook tutorial added
* Bugfix: E-Filtering now has working CCDF
* Chores

## 0.3.0 (2024-07-05): Let's boost things

* Simulator re-written almost entirely
  * Easier to read/maintain thanks to jit and data classes.
  * Roughly 40% faster than previous version.
  * Virtual queue updated with better edge-FCFM policy.
  * EGPD ported to both virtual queue and longest policies.
  * epsilon-filtering (a.k.a. epsilon-coloring) added.
* Switch to Poetry
  * Easier package maintainance
  * Pydata documentation style
  * Supported Python version: >=3.10

## 0.2.2 (2023-06-18): Improved CCDF

* Add a function to draw the (discrete) CCDFs piecewise
* New range of officially supported Python versions: 3.6 -> 3.11

## 0.2.1 (2022-02-03): Big little update

* New optimize_rates for Model. Outputs a flow that optimizes the rates according to some reward weights.
* Refactoring: policies formerly called semi-greedy are now called (semi)-filtering.
* New option weights for filtering policies. Auto-computes the forbidden edges to optimize the reward according to weights.
* Default model tolerance raised to 1e-7 for better detection of null edges.
* Tutorials modified to introduce the new features.
* The notebook used for paper https://hal.archives-ouvertes.fr/hal-03502084 is now included in the documentation.
* Bug hunt: very large simulation could overflow silently (solved by switching logs from uint32 to uint64).

## 0.2.0 (2022-01-24): Brand new API

As the package is at early stage, it had to go through a lot of refactoring.
Hopefully, the result should be more easy to use.

Major changes:

* API completely unified
    * Beginners (and intermediate) users should always go through the Model class to use the package.
    * Advanced users: the documentation reference is your oyster!
* New features
    * A few new graph models.
    * New analysis tools: left kernel, right kernel basis change and display (for simple graphs),
      injectivity/surjectivity, connected components, spanners, and vertices!
    * New policies for the simulator: *priority* and *semi-greedy*!

Minor changes:

* Default rates are proportional to degree (but you can ask for 'uniform')
* We have a logo!
* Hunt for typos in documentation.
* Notebooks tutorials have been updated to cope with the new API.


## 0.1.0 (2022-01-13): First release

* First release on PyPI.
* Refactoring: the package name for the public release is *Stochastic Matching*.
* The graph submodule has been revamped a bit:
    * Obtaining graphs by concatenation of other graphs is now more systematic and optimized.
    * Names of generators have been changes to comply with *official* terminology from Wolfram.
    * A few more generators have been added (e.g. complete, lollipop, barbell).
* Current compatibility: 3.6 -> 3.9


## 0.0.4 (2021-09-30): Misc. improvements

* Add possibility to specify node names. The names are used for all display operations
* New simulator method compute_ccdf to allow access outside show_ccdf
* Simulator method show_ccdf has two new options: indices (only show some nodes), sort
* New simulator method compute_average_queues
* New simulator method show_average_queues with three options: indices (only show some nodes), sort,
  and as_time (divide by arrival rates)
* New *Getting started* notebook.
* Doctests parameters adjusted for faster test suite execution



## 0.0.3 (2021-08-24): Simulators are back!

This update reintroduces simulators, fully revamped

* All simulators are built as a hierarchy of classes
    * Simulator is the mother abstract class
    * QueueSizeSimulator is dedicated to greedy simulators that only use the queue sizes:
      RandomNode, RandomItem, and LongestQueue.
    * QueueStateSimulator is dedicated to greedy simulators that use the age of items:
      FCFM.
    * VQSimulator is a fast implementation of the virtual queue algorithm.
* All non-abstract simulators have a string name. The new `get_simulator_classes()`
  gives a dict of them that can be used to easily select a given simulator.
* Lots of refactoring
    * nodes and edges attributes from graph renamed as incidence and co_incidence respectively.
      Older incidence attribute (plain matrix) removed.
    * spectral.py moved up to main module
    * New simulator module
    * simulator.py split into classes.py and a few other files inside the simulator module.
* Jit is disabled for the testing (allows for inner coverage inspection).
* Tutorials revamped to cope with 0.0.3.
* Old files removed from git for cleanliness


## 0.0.2 (2021-07-22): Hyper-graph release

Lots of changes in this update.

* Graph API has been fully revamped. There is now a GenericGraph class from which SimpleGraph and HyperGraph are derived.
    * Hypergraphs don't have any adjacency attribute, only incidence. They are displayed as bipartite graphs between
      nodes and edges, with an option to explicitly separate nodes from edges.
    * Simple graphs are mostly similar but not compatible with previous Graph class.
    * Hovering on edge now show edge id and description, e.g. edge 6 between nodes 1 and 4 will show "6: (1, 4)"
* TWO HYPERGRAPHS GENERATORS have been added to the mix:
    * hyper_dumbbells can create the candy and larger sweets with trivial kernel.
    * fan can create hypergraphs with more complex kernels, possibly with bipartite-like degenerescence
* The old gramian module is now called spectral.
    * All internals have switched from gramian-based computation to incidence-based computation,
      which makes it fully hypergraph ready (multiplicity / loops should work, but are not tested yet).
    * NEW MAXIMIN FLOW COMPUTATION! No need to optimize each edge, one single linear optimization is enough and
      the result maximizes the minimal edge flow.
    * is_stable method checks the graph kernel dimension and existence of a positive solution.
* Main class as been revamped as well
    * Class name is now MQ (could be changed later).
    * When displayed, flows are checked by default. Conservation law issues gives red nodes, negative edges are red,
      null edges are orange (can be disabled).
    * SIMULATION ENGINE NOT AVAILABLE IN 0.0.2, as there are a lot of things to change. It will return in 0.0.3.
* Misc:
    * Tutorials (simulation apart) have been updated. Enjoy the double degenerated fan!
    * Local coverage computation. Enforcing 100% coverage on all 0.0.2+ code.
    * Minor changes in the display module to cope with the new graph API.
