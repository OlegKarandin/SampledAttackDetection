def pytest_addoption(parser):
    parser.addoption(
        "--csvfile",
        action="store",
        default="./bigdata/Wednesday.csv",
        help="Location to Csvfile",
    )
    parser.addoption(
        "--no_of_alphanum",
        action="store",
        default="0",
        help="Number of alphanumeric characters",
    )
