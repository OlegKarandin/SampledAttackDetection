def pytest_addoption(parser):
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
