from basic_tokenizer import simple_tokenizer
from improved_tokenizer import improved_tokenizer

from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')  # punkt is a pre-trained tokenizer model that can handle contractions and punctuation more effectively than our simple and improved tokenizers.
nltk.download('punkt_tab') 
text = "I can't believe it's already 2026! Time flies..."

print("Original Text:", text)
print("Simple Tokenizer:", simple_tokenizer(text))
print("Improved Tokenizer:", improved_tokenizer(text))
print("NLTK Tokenizer:", word_tokenize(text))