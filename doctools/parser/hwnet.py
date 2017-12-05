
def read_annotation(fpath):
    with open(fpath) as fp:
        for line in fp:
            print(line.strip())
