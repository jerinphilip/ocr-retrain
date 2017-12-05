
def read_annotation(fpath):
    """
    Reads annotation.txt by praveen krishnan
    returns (basepath, imgs, truths)
    """
    base = fpath
    imgs = []
    truths = []
    with open(fpath) as fp:
        for line in fp:
            img, truth = line.strip().split()
            imgs.append(img)
            truths.append(truth)

    return (base, imgs, truths)
