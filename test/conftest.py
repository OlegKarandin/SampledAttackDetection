def pytest_addoption(parser):
    parser.addoption(
        "--mini_csv_path",
        action="store",
        default="./data/mini_Wednesday.csv",
        help="Location to Csvfile",
    )
    parser.addoption(
        "--csvfile",
        action="store",
        default="./bigdata/specific_instances/40.83.143.209_192.168.10.14_443_49461.csv",
        help="Location to Csvfile",
    )
    parser.addoption(
        "--pcappath",
        action="store",
        default="./bigdata/specific_instances/40.83.143.209_192.168.10.14_443_49461.pcap",
        help="Location to PcapFile",
    )
    # TODO: Make it so that we source these parameters from the same config file as the main python would
    parser.addoption(
        "--active_timeout",
        action="store",
        default=0.005,
        help="Maximum change in the IAT between the last three packets."
        "i.e. for packet to be added to be flow (C-B) - (B-A) < active_timeout  must be satisfied",
    )

    parser.addoption(
        "--clump_timeout",
        action="store",
        default=1.0,
        help="Maximum IAT to add a packet to an existing flow. Exceeding it creates a new flow with same key.",
    )
