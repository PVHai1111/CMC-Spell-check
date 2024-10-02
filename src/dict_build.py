import json
from pyvi import ViTokenizer
import string
from collections import Counter
import re

# Danh sách các từ dừng tiếng Việt
stop_words = set(["và", "là", "của", "có", "với", "cho", "để", "đến", "từ", "trong", "bởi", "một", "những", "các", "thì"])

# Hàm kiểm tra xem một từ có chứa ký tự đặc biệt hoặc là số hay không
def contains_special_characters_or_numbers(word):
    return bool(re.search(r'[\d@#\$%\^&\*\.,\+\-\/><\?\'\";:\|\\[\]\{\}~`\!\(\)=]', word))

# Loại bỏ các từ dừng, dấu câu và các từ chứa ký tự đặc biệt hoặc số
def remove_stopwords_and_punctuation(tokens):
    cleaned_tokens = [
        word.lower() 
        for word in tokens.split() 
        if word.lower() not in stop_words 
        and word not in string.punctuation 
        and not contains_special_characters_or_numbers(word)
    ]
    return cleaned_tokens

# Đọc tệp JSON theo từng khối
def process_json_file(file_path):
    word_freq = Counter()
    buffer_size = 1024 * 1024  # 1MB buffer size
    with open(file_path, 'r', encoding='utf-8') as file:
        buffer = ''
        while True:
            chunk = file.read(buffer_size)
            if not chunk:
                break
            buffer += chunk
            while True:
                try:
                    data, idx = json.JSONDecoder().raw_decode(buffer)
                    if isinstance(data, dict):  # Kiểm tra nếu 'data' là dict
                        word_freq.update(process_json_object(data))
                    buffer = buffer[idx:].lstrip()
                except ValueError:
                    break
    return word_freq

# Xử lý từng đối tượng JSON
def process_json_object(data):
    word_freq = Counter()
    try:
        # Đảm bảo 'data' là đối tượng dict
        if isinstance(data, dict):
            print(f"Processing object: {data}")  # Debug print
            ocr_data = data.get("OCR_data", {})
            documents = ocr_data.get("document", [])
            
            for doc in documents:
                # Kiểm tra điều kiện parentIndex
                if doc.get("parentIndex") != "0.0.0.0.0.0.0.0.0.0.0":
                    text_value = doc.get("textValue", "")
                    if text_value:
                        tokenized_document = ViTokenizer.tokenize(text_value)
                        cleaned_tokens = remove_stopwords_and_punctuation(tokenized_document)
                        word_freq.update(cleaned_tokens)
        else:
            print(f"Unexpected data type: {type(data)}")  # Debug print
    except Exception as e:
        print(f"Error processing JSON object: {e}")
    return word_freq

# Xây dựng từ điển
def build_dictionary(word_frequencies):
    # Tính số lượng từ cần loại bỏ
    num_words_to_remove = int(len(word_frequencies) * 0.3)
    
    # Sắp xếp từ theo tần suất xuất hiện tăng dần và loại bỏ 30% các từ xuất hiện ít nhất
    sorted_words = sorted(word_frequencies.items(), key=lambda item: item[1])
    filtered_words = sorted_words[num_words_to_remove:]
    
    # Xây dựng từ điển kết quả
    dictionary = {word: freq for word, freq in filtered_words}
    return dictionary
