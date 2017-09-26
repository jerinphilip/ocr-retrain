from lxml import etree
import cv2
import numpy as np

# Parse line.xml file.
def parse_ocr_xml(xml_file):
    with open(xml_file, encoding='utf-8') as f:
        #print("Read", xml_file)
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
        if height is 0 or width is 0:
            subImg = np.zeros((32, 32))

        height, width = subImg.shape
        #print(subImg.shape)
        # TODO Take care of zero height
        ratio = 32/height
        resized = cv2.resize(subImg, None, fx=ratio, fy=ratio, 
                interpolation = cv2.INTER_CUBIC)
        return resized

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
        udict["page"][pno]["atoms"].append(atom)

    # Return ultimate dict
    return udict
def pages(book_dir_path):
    obtainxml = lambda f: book_dir_path + f + '.xml'
    filenames = map(obtainxml, ['line', 'word', 'text'])
    lines, words, text = list(map(parse_ocr_xml, filenames))
    pno = [entry["PageNo"] for entry in lines]
    return pno[-1]

def images_and_truths(udict, mapping_f):
    result = []
    prefix = udict["prefix"]
    for pno in udict["page"]:
        extract_required = lambda x: udict["page"][pno][x]
        required_keys = ['Text', 'ImageLoc', 'atoms']
        text, imgloc, atoms = list(map(extract_required, required_keys))

        # Order atoms by Key
        atom_truths, atoms = mapping_f(text, atoms)
        atom_images = extract_atoms(prefix+imgloc, atoms)
        #print(len(atom_images),  len(atom_truths))
        result.append((atom_images, atom_truths))
    return result

def line_mapping_f(text, atoms):
    lines = list(filter(lambda x: x, text.split('\n')))
    olines = []
    for atom in atoms:
        i = int(atom["LineNo"])
        olines.append(lines[i])
    return (olines, atoms)

def word_mapping_f(text, atoms):
    lines = list(filter(lambda x: x, text.split('\n')))
    words = list(map(lambda x: x.split(), lines))
    owords = []
    for atom in atoms:
        i, j = list(map(lambda x: int(atom[x]), ["LineNo", "SerialNo"]))
        owords.append(words[i-1][j-1])
    return (owords, atoms)


def read_book(book_dir_path):
    obtainxml = lambda f: book_dir_path + f + '.xml'
    filenames = map(obtainxml, ['line', 'word', 'text'])
    lines, words, text = list(map(parse_ocr_xml, filenames))
    ud = group(text, words)
    ud["prefix"] = book_dir_path
    
    pagewise = images_and_truths(ud, word_mapping_f)
    return pagewise


def full_text(book_dir_path):
    """ Used to extract all the vocabulary in a particular book """
    text_file = book_dir_path + 'text.xml'
    text_d = parse_ocr_xml(text_file)
    text = ""
    for d in text_d:
        text += d["Text"]
    return text

