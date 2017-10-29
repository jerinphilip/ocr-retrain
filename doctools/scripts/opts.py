
def base_opts(parser):
    parser.add_argument('-o', '--output', required=True, help="Output directory")
    parser.add_argument("-c", "--config", type=str, required=True, help="/path/to/config")
    parser.add_argument("-l", "--lang", type=str, required=True, help="language to run for")
    parser.add_argument("-b", "--book", type=int, required=True, help="id of the book to run simulation on")

