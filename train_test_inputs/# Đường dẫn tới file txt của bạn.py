# Đường dẫn tới file txt của bạn
file_path = r"c:\Users\buiti\ZoeDepth-main\train_test_inputs\nyudepthv2_test_files_with_gt.txt"

# Đọc file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Thêm giá trị vào cuối mỗi dòng
lines = [line.strip() + " 518.8579\n" for line in lines]

# Ghi lại vào file
with open(file_path, 'w') as file:
    file.writelines(lines)
