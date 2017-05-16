from lxml import etree
from pprint import pprint
import cv2

"""
IIIT OCR Web System follows the following xml tree structure.

<resultset ...>
    <row>
        <field name={{name}}>{{text}}</field>
    </row>
    .
    .
    .
</resultset>

All 3, line.xml, word.xml, text.xml can be parsed using the same 
function provided below.

"""

# Parse line.xml file.
def parse_ocr_xml(xml_file):
    with open(xml_file) as f:
        print("Read", xml_file)
        root = etree.parse(f)
        rows = root.xpath("row")

        # Subfunction to extract from a row
        extract = lambda field: (field.get("name"), field.text)
        extract_field = lambda x: dict(map(extract, x.xpath("field")))
        export = list(map(extract_field, rows))
        return export

# Atoms could here be words or lines
# An image is a page.
def extract_atoms(image_location, atoms):
    image = cv2.imread(image_location)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return_val, thresholded = cv2.threshold(grayscale, 0, 1, cv2.THRESH_OTSU)
    rows, cols = thresholded.shape

    def trimRect(atom):
        rectKeys = ['rectLeft', 'rectTop', 'rectRight', 'rectBottom']
        x, y, X, Y = list(map(lambda x: int(atom[x]), rectKeys))
        #print(x, y, X, Y)
        trim_v = lambda w: lambda v: max(0, min(v, w))
        x = trim_v(cols)(x)
        X = trim_v(cols)(X)
        y = trim_v(rows)(y)
        Y = trim_v(rows)(Y)
        return (x, X, y, Y)

    def extract_atom_image(atom):
        #print(trimRect(atom))
        x, X, y, Y = trimRect(atom)
        subImg = thresholded[y:Y, x:X]
        # Truncate to height 32
        height, width = subImg.shape
        #print(subImg.shape)
        ratio = 32/height
        cv2.resize(subImg, None, fx=ratio, fy=ratio, 
                interpolation = cv2.INTER_CUBIC)
        return subImg

    atomImages = list(map(extract_atom_image, atoms))
    return atomImages

def group(text_data, atoms):
    udict = {}
    required_keys = ["ImageLoc", "Text"]
    udict["page"] = {}
    for entry in text_data:
        if not "BookCode" in udict:
            udict["BookCode"] = entry["BookCode"]
        pno = int(entry["PageNo"])
        if pno not in udict["page"]:
            udict["page"][pno] = dict((key, entry[key]) for key in required_keys)
            udict["page"][pno]["atoms"] = []
    for atom in atoms:
        pno = int(atom["PageNo"])
        #print(pno, udict.keys())
        #assert(pno in udict)
        udict["page"][pno]["atoms"].append(atom)

    # Return ultimate dict
    return udict

def images_and_truths(udict, mapping_f):
    result = []
    for pno in udict["page"]:
        extract_required = lambda x: udict["page"][pno][x]
        required_keys = ['Text', 'ImageLoc', 'atoms']
        text, imgloc, atoms = list(map(extract_required, required_keys))

        # Order atoms by Key
        atom_truths, atoms = mapping_f(text, atoms)
        atom_images = extract_atoms(prefix+imgloc, atoms)
        print(len(atom_images),  len(atom_truths))
        result.append((atom_images, atom_truths))
    return result

def line_mapping_f(text, atoms):
    lines = list(filter(lambda x: x, text.split('\n')))
    atoms = sorted(atoms, key=lambda x: x["LineNo"])
    return (lines, atoms)

def word_mapping_f(text, atoms):
    lines = list(filter(lambda x: x, text.split('\n')))
    words = list(map(lambda x: x.split(), lines))
    owords, oatoms = [], []
    for atom in atoms:
        i, j = list(map(lambda x: int(atom[x]), ["LineNo", "SerialNo"]))
        owords.append(words[i-1][j-1])
        oatoms.append(atom)
    return (owords, atoms)


if __name__ == '__main__':
    import sys
    prefix = sys.argv[1]
    files = map(lambda f: prefix + f + '.xml', ['line', 'word', 'text'])
    lines, words, text = list(map(parse_ocr_xml, files))
    ud = group(text, lines)
    result = images_and_truths(ud, line_mapping_f)
    #ud = group(text, words)
    #result = images_and_truths(ud, word_mapping_f)
    for pairs in result:
        images, truths = pairs
        for image, truth in zip(images, truths):
            print("Truth:", truth)
            cv2.imshow("window", image)
            cv2.waitKey(0)
    print(result)
