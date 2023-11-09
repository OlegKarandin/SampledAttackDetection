from pathlib import Path, PosixPath

import pyshark

# Get Pcap file
pcap_file = Path("../bigdata/Wednesday-WorkingHours.pcap")
assert pcap_file.exists(), "Could not find pcap file"

# Open file with pyshark
cap = pyshark.FileCapture(str(pcap_file))

# Iterate over packets
limit = 10
for i, packet in enumerate(cap):
    if i == limit:
        break

    print(packet)
    # Get the capture time for this packet:
    print(packet.time)

    print(packet.layers)
    print(packet.frame_info)
    print(packet.frame_info.time)
    print(packet.frame_info.time_epoch)
