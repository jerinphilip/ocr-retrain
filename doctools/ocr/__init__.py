from doctools.ocr import pyocr
import numpy as np
import cv2

class GravesOCR:
    def __init__(self, weights_f, lookup_f):
        self.net = pyocr.NetAPI(weights_f, lookup_f)
        # Load lookup table.
        self.table = {}
        with open(lookup_f) as lf:
            values = list(map(
                lambda x: self.stringToUnicode(x.strip()),
                lf))
            values = [None] + values
            self.table = dict(zip(values, range(len(values))))

    def asType(self, interfaceType, sequence):
        fv = interfaceType(len(sequence))
        for i in range(len(sequence)):
            fv[i] = sequence[i]
        return fv

    def test(self, sequence):
        return self.net.test(self.asType(pyocr.FloatVector, sequence))

    def train(self, sequences, targets):
        sequenceContainer = pyocr.FloatVVector(len(sequences))
        for i in range(len(sequences)):
            sequenceContainer[i] = self.asType(pyocr.FloatVector, sequences[i])

        targetsContainer = pyocr.IntVVector(len(targets))
        for i in range(len(targets)):
            targetsContainer[i] = self.asType(pyocr.IntVector, targets[i])

        return self.net.train(sequenceContainer, targetsContainer)
    
    def export(self):
        return self.net.exportModel()

    def stringToUnicode(self, ocr_output):
        codepoint = ocr_output[1:]
        codepoint_value = int(codepoint, 16)
        return chr(codepoint_value)

    def unicodeToClasses(self, string):
        return list(map(lambda x: self.table[x], string))

    def cvImgToGraves(self, img):
        vector = list(map(float, img.T.ravel()))
        return vector
    
    def recognize(self, image):
        sequence = self.cvImgToGraves(image)
        cps = self.test(sequence)
        chars = list(map(lambda x: self.stringToUnicode(x), cps))
        return ''.join(chars)



