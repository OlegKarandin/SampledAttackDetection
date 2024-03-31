# Introduction

Application project to test sampled algorithm on DDOS Detection.

At some point in the development working with such large `pcap` files dynamically was deemed too computationally taxing;
as such, much of the code has been adapted to work on csv files that are generated via `pcap_to_csv.py` file.

# Structure

1. `./sampleddetection/` package to keep all relevant utilities organized.
2. `./scripts/` any helper scripts that are not part of the main program

Within the package some main modules with their responsibilities are:

1. `Reader`s : These are classes that maintain a pointer to some sort of source file, be it a pcap or csv file.
1. `WindowSampler`s: Will generally use readers and create samples fitting to our experiments
1. `Environment`s : Represent a reinforcement learning environment
1. `Datastructure`s : Will do a bulk of the work in the formation of flows and the derivation of statistics.
   Generally used by samplers to retrieve statistics once a list of packets corresponding to a window is extracted

Scritps:

1. `./windowed/` Entry point for sample experiments

# Requirements

Python libraries may be found in `requirements.txt`.

In order to run `pytest ./test` you must ensure your hell instance has `tshark` in its `$PATH`.

# Additional Requirements

Outside of those listed in `requirements.txt` you will also want to download those
listed below:

1. [CICFlowMeter for Python](https://gitlab.com/hieulw/cicflowmeter)
1. [Original Java CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter)
