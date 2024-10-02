from dict_build import process_json_file, build_dictionary
from spell_check import spell_check

# Đường dẫn đến tệp JSON chứa dữ liệu văn bản
file_path = r'C:\Users\Admin\Downloads\data.json'

# Thực hiện xử lý
word_frequencies = process_json_file(file_path)
print(f"Word frequencies: {word_frequencies}")  # In tần suất từ để kiểm tra

# Xây dựng từ điển
dictionary = build_dictionary(word_frequencies)
print(f"Số lượng từ trong từ điển: {len(dictionary)}")
print(dictionary)

# Đoạn văn bản cần kiểm tra chính tả
input_text = "tiền thuê đất djehd t43 giá cả htyk."

# Thực hiện kiểm tra chính tả
errors = spell_check(input_text, dictionary)

# In ra các lỗi và vị trí lỗi
if errors:
    print("Các lỗi chính tả và vị trí:")
    for error in errors:
        print(f"Từ '{error[0]}' tại vị trí {error[1]}")
else:
    print("Không có lỗi chính tả.")
