import cProfile
from os import name
import pstats
import pandas as pd
import io
import os, sys

scrip_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(scrip_directory)
sys.path.append(parent_directory)
from datastructures.flowsession import SampledFlowSession
from networking.datastructures.packet_like import CSVPacket

# Create a profiler object
profiler = cProfile.Profile()

flowsession = SampledFlowSession()
flowsession.reset()
print("Flow session initialized")

# Load min Wednesday
print("Reading CSV file")
df = pd.read_csv("../../data/mini_Wednesday.csv")
raw_samples = [CSVPacket(p) for _, p in df.iloc[:100, :].iterrows()]
print("Obtained package")

# Profile the 'test_function'
print("Profiling")
profiler.enable()  # Start profiling
for sample in raw_samples:
    flowsession.on_packet_received(sample)
profiler.disable()  # Stop profiling
print("Profiling Done")

# Create a stream to capture profiling data
stream = io.StringIO()
stats = pstats.Stats(profiler, stream=stream)
stats.sort_stats("time")  # or 'cumulative', 'ncalls'

# Print the results of the profile
stats.print_stats()
data = stream.getvalue()

# Convert the timing information to milliseconds
data_lines = data.split("\n")
for line in data_lines:
    parts = line.split()
    if len(parts) > 0 and not isinstance(parts[0], int):
        print("\t".join(parts))
        continue
    if (
        len(parts) >= 5 and parts[0].replace(".", "", 1).isdigit()
    ):  # Check if it's a line with timing data
        # Assuming the default format: ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        tottime_ms = (
            float(parts[2]) * 1e6
        )  # Convert total time from seconds to milliseconds
        cumtime_ms = (
            float(parts[4]) * 1e6
        )  # Convert cumulative time from seconds to milliseconds
        parts[2] = f"{tottime_ms:.3f}"
        parts[4] = f"{cumtime_ms:.3f}"
        print(" ".join(parts))
    else:
        print(line)

# print(stream.getvalue())
