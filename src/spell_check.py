from pyvi import ViTokenizer
from dict_build import stop_words, contains_special_characters_or_numbers

# Kiểm tra chính tả đoạn văn bản
def spell_check(text, dictionary):
    tokenized_text = ViTokenizer.tokenize(text)
    tokens = tokenized_text.split()
    errors = []
    
    for idx, word in enumerate(tokens):
        word_lower = word.lower()
        if word_lower not in dictionary and not contains_special_characters_or_numbers(word_lower) and word_lower not in stop_words:
            errors.append((word, idx))
    
    return errors
