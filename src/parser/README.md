WebToTrain.py
===


IIIT OCR Web annotation system follows the following xml tree structure.

```xml
<resultset ...>
    <row>
        <field name={{name}}>{{text}}</field>
    </row>
    .
    .
    .
</resultset>
```

All 3, line.xml, word.xml, text.xml can be parsed using the same 
function.

# Details

Now, we're provided with the following from the above files:

 * line.xml - bounding boxes for lines
 * word.xml - bounding boxes for words
 * text.xml - ground truths for the images.

There's a relationship between the 'PageNo', 'LineNo', 'SerialNo' attributes
present in the above files using which you can relate the data in the three.

We can use lines or words depending on what we want, the configurability is
extended by using a `line_mapping_f` and a `word_mapping_f`, which maps the
lines/words to the text.


The rough procedure for reading books is:

1. Parse xmls. `parse_xml`.
2. Group the data in xmls relating them to a page level. `group`.
3. Obtain (atoms, truths) grouped by pages. `images_and_truths(ud, mapping)`.

