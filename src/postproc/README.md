Error Module 
===

Since I don't want to implement abstract base
classes in python in a convoluted way, I'd specify
here what an error module should be structured
like.

```python

def ErrorModule:

    def __init__(*args, **kwargs):
        # Use kwargs['lang'] to load corresponding
        # model for a language.

    def error(self, word):
        # Define error method which predicts how
        # much a word has errored.
        # Value in [0, 1]

```

The dictionary module has been implemented as a
sample, with error values being 0 or 1.
