import re

def configurable_tokenizer(text: str, keep_punctuation: bool = True, handle_contractions: bool = False,lowercase:bool=True)->list:
    """
    Configurable tokenizer:
    - Lowercases text (optional)
    - Removes punctuation (optional)
    - Handles contractions (optional)
    """

    if lowercase:
        text = text.lower()

    if handle_contractions:
        contractions = {
            "don't": "do not",
            "can't": "can not",
            "won't": "will not",
            "i'm": "i am",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are"
        }

        for key, value in contractions.items():
            text = text.replace(key, value)

    if keep_punctuation:
        # This Regex finds words (\w+) OR individual punctuation marks ([^\w\s])
            tokens = re.findall(r"\w+|[^\w\s]", text)
    else:
        # This removes punctuation entirely and then splits
        text = re.sub(r"[^\w\s]", '', text)
        tokens = text.split()

    # tokens = text.split()
    return tokens

if __name__ == "__main__":
    sample = "I am stressed... really stressed!!!"

    print("\n--- Default ---")
    print(configurable_tokenizer(sample))

    print("\n--- Keep punctuation ---")
    print(configurable_tokenizer(sample, keep_punctuation=True))

    print("\n--- No contraction handling ---")
    print(configurable_tokenizer(sample, handle_contractions=False))

    print("\n--- No lowercase ---")
    print(configurable_tokenizer(sample, lowercase=False))