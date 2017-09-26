OCR Module
===

Rough API this should provide:

```python
class OCR_Module:
    def __init__(self, weights_file, lookup_file):
        # initialize class from weights 
        # saved file and lookup file

    # ([sequence], [target]) -> [errors]
    def train(self, sequences, targets):
        # Train function
        # returns errors

    # sequence -> prediction :: String
    def test(self, sequence):
        # Test.

    def export(self):
        # Export model
    
```



# Graves OCR
