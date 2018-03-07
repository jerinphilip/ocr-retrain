
def base_opts(parser):
    parser.add_argument("-c", "--config", type=str, required=True, help="/path/to/config")
    parser.add_argument("-l", "--lang", type=str, required=True, help="language to run for")
    parser.add_argument("-lmap", "--lookup", type=str, required=True, help="lookup table")