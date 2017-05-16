import os
import sys
import cv2
import pandas as pd
import numpy as np
from xml.dom import minidom
#df = pd.read_csv('unicode_table.csv')
path = sys.argv[1]

def parse_cordinates(PageNo,Bookcode):
    full_path = os.path.join(path,os.path.join(Bookcode,'line.xml'))     #takes the line.xml file of a certain book and parses all the 
    xmldoc = minidom.parse(full_path)
    rows = xmldoc.getElementsByTagName('row')
    a=[]
    for row in rows:
        if (row.getElementsByTagName('field')[1].firstChild.data == str(PageNo)):
            LineNo = row.getElementsByTagName('field')[3].firstChild.data
            x1 = row.getElementsByTagName('field')[5].firstChild.data
            x2 = row.getElementsByTagName('field')[6].firstChild.data
            y1 = row.getElementsByTagName('field')[7].firstChild.data
            y2 = row.getElementsByTagName('field')[8].firstChild.data
            a.append([int(LineNo),int(x1), int(y1), int(x2), int(y2)])
            # print text_file,text
            data = np.array(a)
            data = data[data[:, 2].argsort()]
    return data

def show_image(Image,line_info):
    try:
        #cv2.imshow('lineImage',Image)
        #cv2.waitKey(0)
        print (line_info)
    except Exception as e:
        print (e)



def feature_extract(Image,bbox_info,line_info):
    sequences=[]
    target = []
    try:
        image = cv2.imread(Image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        (rows, cols, chans) = image.shape
        lines = line_info.split('\n')
        linecount = -1
        for line in lines[:]:
            if len(line) > 0:
                linecount += 1
                char = []
                for each_character in line:
                    #if any(df['character'] == each_character) == True:
                     #   index = ((np.where(df['character'] == each_character)[0]))
                     #   char.append(df['unicode'].iloc[index[0]])
                    #else:
                        #print (each_character.encode("unicode-escape").decode())
                    uni_repr = each_character.encode("unicode-escape").decode()

                    char.append(uni_repr)

                a = ' '.join(char)

                target.append(a)
                LineNo = bbox_info[linecount][0]
                x1 = bbox_info[linecount][1]
                y1 = bbox_info[linecount][2]
                x2 = bbox_info[linecount][3]
                y2 = bbox_info[linecount][4]

                if (x1 < 0 or y1 < 0 or x1 > cols or y1 > rows or x2 > cols or y2 > rows or x2 - x1 < 1 or y2 - y1 < 1):
                    print("the line coordinates are exceeding image dimensions " << x1 << " " << y1 << " " << (
                    x2 - x1) << " " << (y2 - y1) << "\n")
                    continue
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                line_crop = thresh1[int(y1):int(y2), int(x1):int(x2)]
                newHeight = 32
                aspectRatio = (float(x2 - x1) / float(y2 - y1))
                newWidth = int(np.ceil(aspectRatio * newHeight))
                try:
                    resized_image = cv2.resize(line_crop, (int(newWidth), int(newHeight)), interpolation=cv2.INTER_AREA)
                    # cv2.imshow('image',line_crop)
                    #cv2.imshow('image2', resized_image)
                    #cv2.waitKey(0)
                    (cropped_rows, cropped_cols) = resized_image.shape
                except Exception as e:
                    print(int(newWidth), int(newHeight))
                    print (str((x2 - x1)) + '\n' + str(y2 - y1))
                pixels = []
                for r in range(cropped_rows):
                    for c in range(cropped_cols):
                        pixel = resized_image[r, c]
                        if pixel == 255:
                            pixels.append(str(1))
                        else:
                            pixels.append(str(0))
                A = np.array(pixels)
                B = np.reshape(A, (-1, cropped_cols))
                b = ' '.join(pixels)

                sequences.append(b)

                #show_image(resized_image,line)
        return(sequences,target)
    except Exception as e:
        print (e)


def get_data():                                  
    seqences=[]
    targets = []
    book_list = [x[0] for x in os.walk(path)]
    for each_dir in book_list[1:]:                       #picks one book at a time 
        full_path = os.path.join(each_dir, 'text.xml')   # looks for text.xml in the book directory

        if os.path.exists(full_path):
            xmldoc = minidom.parse(full_path)
            rows = xmldoc.getElementsByTagName('row')      #parses each line of the text.xml with the tag name row
            for row in rows[:2]:
                BookCode = row.getElementsByTagName('field')[0].firstChild.data
                PageNo = row.getElementsByTagName('field')[1].firstChild.data
                ImagePath = os.path.join(each_dir, row.getElementsByTagName('field')[2].firstChild.data)
                text = row.getElementsByTagName('field')[3].firstChild.data
                bbox_info = parse_cordinates(PageNo, BookCode)  # passes the PageNo. and bookID to the function parse cordinates which in turn returns the 
                                                                #bbox_cordinates of that particular page.
                s,t=feature_extract(ImagePath,bbox_info,text)   #we extract the features for each image by providing the text info , image path and the line 
                                                                # cordinate values for each page  and append them in two seperate lists 
                                                                #one for containing the pixel values and the other containg the ground truths. 
                #print (t)
                seqences.append(s)
                targets.append(t)

    return (seqences,targets)

seq,tar=get_data()
