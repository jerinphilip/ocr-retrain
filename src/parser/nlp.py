
def extract_words(text):
    tokens = []
    lines = text.splitlines()
    for line in lines:
        candidates = tokenize(line)
        valid = filter(lambda x:x, candidates)
        tokens.extend(valid)
    return tokens

