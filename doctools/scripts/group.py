from doctools.parser.hwnet import read_annotation
from doctools.meta.file_locs import get_pickeled

if __name__ == '__main__':
    fpath = '/OCRData2/praveen-intermediate/0191/annotation.txt'
    base, imgs, truths = read_annotation(fpath)
    graph = get_pickeled('0191', type='edges')
    print(graph.keys())
    for i, component in enumerate(graph["components"]):
        for j in component:
            print(truths[j])
        print('-'*10)
        input()

    print(len(imgs), len(truths), base)
