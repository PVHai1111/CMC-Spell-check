import json
from pyvi import ViTokenizer
import string
from collections import Counter
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

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

# Đọc tệp JSON và trích xuất các câu văn
def extract_sentences_from_json(file_path):
    sentences = []
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
                        sentences.extend(process_json_object(data))
                    buffer = buffer[idx:].lstrip()
                except ValueError:
                    break
    return sentences

# Xử lý từng đối tượng JSON và trích xuất các câu văn
def process_json_object(data):
    sentences = []
    try:
        # Đảm bảo 'data' là đối tượng dict
        if isinstance(data, dict):
            ocr_data = data.get("OCR_data", {})
            documents = ocr_data.get("document", [])
            
            for doc in documents:
                # Kiểm tra điều kiện parentIndex
                if doc.get("parentIndex") != "0.0.0.0.0.0.0.0.0.0.0":
                    text_value = doc.get("textValue", "")
                    if text_value:
                        sentences.append(text_value)
    except Exception as e:
        print(f"Error processing JSON object: {e}")
    return sentences

# Hàm giả để lấy định nghĩa từ từ điển
def get_definition(word):
    # Trong thực tế, bạn có thể thay thế hàm này bằng API gọi tới từ điển trực tuyến hoặc cơ sở dữ liệu có sẵn
    fake_dictionary = {
        "nhà": "Nơi để ở",
        "xe": "Phương tiện di chuyển",
        "sách": "Vật dụng dùng để đọc",
        "tốt": "Đạt mức độ cao",
        "xấu": "Không đạt yêu cầu"
    }
    return fake_dictionary.get(word, "Không tìm thấy định nghĩa")

# Xây dựng cặp tích cực và tiêu cực
def build_positive_negative_pairs(sentences):
    positive_pairs = []
    negative_pairs = []
    
    for sentence in sentences:
        tokenized_sentence = ViTokenizer.tokenize(sentence)
        cleaned_tokens = remove_stopwords_and_punctuation(tokenized_sentence)
        
        for word in cleaned_tokens:
            definition = get_definition(word)
            
            if definition != "Không tìm thấy định nghĩa":
                # Tạo cặp tích cực
                similar_words = [w for w in cleaned_tokens if w != word and get_definition(w) == definition]
                for similar_word in similar_words:
                    positive_pairs.append((word, similar_word))
                
                # Tạo cặp tiêu cực
                different_words = [w for w in cleaned_tokens if w != word and get_definition(w) != definition]
                if different_words:
                    negative_word = random.choice(different_words)
                    negative_pairs.append((word, negative_word))
    
    return positive_pairs, negative_pairs

class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # Lấy embedding của token [CLS]

    def get_embedding(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        return self.forward(input_ids, attention_mask)

def train_contrastive_model(model, positive_pairs, negative_pairs, epochs=10, learning_rate=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MarginRankingLoss(margin=0.5)

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for pos_pair, neg_pair in zip(positive_pairs, negative_pairs):
            word_pos1, word_pos2 = pos_pair
            word_neg1, word_neg2 = neg_pair

            emb_pos1 = model.get_embedding(word_pos1)
            emb_pos2 = model.get_embedding(word_pos2)
            emb_neg1 = model.get_embedding(word_neg1)
            emb_neg2 = model.get_embedding(word_neg2)

            pos_similarity = model.cosine_similarity(emb_pos1, emb_pos2)
            neg_similarity = model.cosine_similarity(emb_neg1, emb_neg2)
            
            # Nhãn cho MarginRankingLoss: 1 cho cặp tích cực và -1 cho cặp tiêu cực
            target = torch.tensor([1], dtype=torch.float)

            loss = criterion(pos_similarity, neg_similarity, target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(positive_pairs)}")

# Đường dẫn tới tệp JSON (bạn cần thay đổi đường dẫn này)
file_path = "path_to_your_file.json"

# Trích xuất các câu văn từ tệp JSON
sentences = extract_sentences_from_json(file_path)

# Xây dựng cặp tích cực và tiêu cực
positive_pairs, negative_pairs = build_positive_negative_pairs(sentences)

# Khởi tạo mô hình
contrastive_model = ContrastiveModel()

# Huấn luyện mô hình với các cặp tích cực và tiêu cực
train_contrastive_model(contrastive_model, positive_pairs, negative_pairs)

class SpellChecker:
    def __init__(self):
        self.model = contrastive_model
        self.dictionary = fake_dictionary

    def check_spelling(self, word):
        if word in self.dictionary:
            return True
        else:
            return False
