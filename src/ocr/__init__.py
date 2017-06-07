from ocr import pyocr
import numpy as np
import cv2
import re

class GravesOCR:
    def __init__(self, weights_f, lookup_f):
        self.net = pyocr.NetAPI(weights_f, lookup_f)
        # Load lookup table.
        self.table = {}
        with open(lookup_f) as lf:
            values = list(map(
                lambda x: self.stringToUnicode(x.strip()),
                #lambda x: (x.strip('*\n')),
                lf))
            values = [None] + values
            self.table = dict(zip(values, range(len(values))))
            #print(self.table)
    def asType(self, interfaceType, sequence):

        fv = interfaceType(len(sequence))
        for i in range(len(sequence)):
            fv[i] = sequence[i]
        return fv

    def test(self, sequence):
        return self.net.test(self.asType(pyocr.FloatVector, sequence))

    def train_aux(self, sequences, targets,maxEpochs):
        sequenceContainer = pyocr.FloatVVector(len(sequences))
        
        for i in range(len(sequences)):
            sequenceContainer[i] = self.asType(pyocr.FloatVector, sequences[i])

        targetsContainer = pyocr.IntVVector(len(targets))
        for i in range(len(targets)):
            
            targetsContainer[i] = self.asType(pyocr.IntVector, targets[i])

        return self.net.train2(sequenceContainer, targetsContainer,maxEpochs)
    def no_of_characters(self,truths):
        sum=0
        for truth in truths:
          sum+=len(list(truth))
        return(sum)
    
    def train(self, images, truths):
        sequences,targets,indices=[],[],[]
        for image,truth in zip(images,truths):                     #images,truths are lists of numpy array and ground gtruth strings respectively
            sequences.append(self.cvImgToGraves(image))              #sequences=[[1,1,0,,1],[0,0,1,1]...[0,0,1,1]]
            targets.append(self.get_unicode_value(truth))            # truths=[['0084','0929','0098'].....[0067]]

        for target in targets:
            indices.append(self.unicodeToClasses(target))

        self.train_aux(sequences,indices)

    def train2(self, train_images, train_truths, val_images, val_truths, maxEpochs):
        train_sequences,train_targets,train_indices=[],[],[]
        val_sequences,val_targets,train,val_indices=[],[],[]

        for train_image,train_truth in zip(train_images,train_truths):                     #images,truths are lists of numpy array and ground gtruth strings respectively
            train_sequences.append(self.cvImgToGraves(train_image))              #sequences=[[1,1,0,,1],[0,0,1,1]...[0,0,1,1]]
            train_targets.append(self.get_unicode_value(train_truth))            # truths=[['0084','0929','0098'].....[0067]]

        for train_target in train_targets:
            train_indices.append(self.unicodeToClasses(train_target))

        for val_image,val_truth in zip(val_images,val_truths):                     #images,truths are lists of numpy array and ground gtruth strings respectively
            val_sequences.append(self.cvImgToGraves(val_image))              #sequences=[[1,1,0,,1],[0,0,1,1]...[0,0,1,1]]
            val_targets.append(self.get_unicode_value(val_truth))            # truths=[['0084','0929','0098'].....[0067]]

        for val_target in val_targets:
            val_indices.append(self.unicodeToClasses(val_target))

        self.train_aux(train_sequences,train_indices,val_sequences,val_indices, maxEpochs)




    def export(self):
        return self.net.exportModel()

    def stringToUnicode(self, ocr_output):
        codepoint = ocr_output[1:]
        codepoint_value = int(codepoint, 16)
        return chr(codepoint_value)

    def unicodeToClasses(self, string):
        
        try:
          return list(map(lambda x: self.table[x], string))
        except KeyError as e:
          print(e)
          pass

    def cvImgToGraves(self, img):
        vector = list(map(float, img.T.ravel()))
        return vector                                      # flattens a numpy array into a list
    
    def recognize(self, image):
        sequence = self.cvImgToGraves(image)
        cps = self.test(sequence)
        chars = list(map(lambda x: self.stringToUnicode(x), cps))
        return ''.join(chars)

    def get_unicode_value(self,string):
        #string=(string).encode("unicode-escape").decode()

        return(list(string))


