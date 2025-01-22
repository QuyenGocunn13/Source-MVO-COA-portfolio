import numpy as np
import random
import csv

tc1 = random.randint(1, 20) # Lợi nhuận (Positive)
tc2 = random.randint(1, 20) # Rủi ro (negative)
tc3 = random.randint(1, 20) # Thanh khoản (Positive)
tc4 = random.randint(1, 20) # Thời gian đầu tư (average)
tc5 = random.randint(1, 20) # Chi phí (negative)
tc6 = random.randint(1, 20) # Tính ổn định của dòng tiền (Positive)
tc7 = random.randint(1, 20) # Đa dạng hóa (average)

# Dữ liệu => Tổng hợp điể từ web
# data = [13, 2, 5, 5, 7, 1, 7] # Thay đổi sau mỗi lần vote
data = [tc1, tc2, tc3, tc4, tc5, tc6, tc7]

# Lưu vào file CSV
with open('criteria_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Value'])
    for value in data:
        writer.writerow([value])


# AHP
# Hàm đọc dữ liệu từ file CSV

def read_csv_to_array(filename):
    values = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            values.append(float(row[0]))
    return values


# Hàm quy đổi giá trị sang hệ 9
def normalize_to_nine_scale(data):
    max_value = max(data)
    min_value = min(data)
    normalized = []

    for value in data:
        if max_value == min_value:
            normalized.append(5)
        else:
            normalized_value = 1 + 8 * (value - min_value) / (max_value - min_value)
            normalized.append(round(normalized_value))

    return normalized


# Hàm tạo ma trận so sánh
def create_comparison_matrix(normalized_scores):
    n = len(normalized_scores)
    matrix = np.ones((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = normalized_scores[i] / normalized_scores[j]

    return matrix


criteria_data = read_csv_to_array('criteria_data.csv')  # Đọc dữ liệu từ file CSV

# Quy đổi dữ liệu sang hệ 9
normalized_scores = normalize_to_nine_scale(criteria_data)

# Tạo ma trận so sánh
comparison_matrix = create_comparison_matrix(normalized_scores)

# Chuẩn hóa ma trận và tính trọng số
col_sum = np.sum(comparison_matrix, axis=0)
normalized_matrix = comparison_matrix / col_sum
weights = np.mean(normalized_matrix, axis=1)
# criteria_values = np.random.rand(20, len(weights))

# Tính tỷ lệ phần trăm trọng số
weights_percent = weights / np.sum(weights) * 100

# Ma trận chứa các giá trị của tiêu chí cho từng tùy chọn đầu tư
criteria_values = np.array([
    [3, 2, 8, 6, 2, 7, 5],
    [5, 4, 3, 8, 5, 6, 4],
    [2, 1, 9, 4, 1, 9, 3],
    [6, 5, 8, 5, 3, 4, 6],
    [7, 7, 7, 5, 4, 3, 7],
    [4, 3, 5, 7, 3, 8, 4],
    [3, 2, 9, 8, 2, 9, 3],
    [6, 4, 6, 6, 4, 5, 5],
    [4, 2, 7, 7, 3, 8, 4],
    [5, 3, 6, 6, 3, 6, 5],
    [8, 7, 8, 5, 4, 3, 6],
    [7, 8, 7, 4, 5, 2, 7],
    [9, 9, 7, 3, 7, 1, 8],
    [9, 9, 6, 2, 8, 1, 8],
    [6, 3, 6, 7, 4, 7, 5],
    [2, 1, 9, 4, 1, 9, 3],
    [8, 8, 6, 3, 6, 2, 5],
    [7, 9, 6, 3, 6, 2, 5],
    [9, 9, 3, 7, 6, 1, 8],
    [8, 8, 6, 6, 5, 3, 7]
])


# Hàm fitness của danh mục
def fitness_function_investment(i, weights, criteria_values, weights_percent):
    # 0.Lợi nhuận, 1.Rủi ro, 2.Thanh khoản, 3.Thời gian đầu tư
    # 4.Chi phí, 5.Tính ổn định, 6.Đa dạng hóa
    # Positive -> TC: 0,2,5
    # Negative -> TC: 1,4
    # Average -> TC: 3,6


    if i < 0 or i >= criteria_values.shape[0]:
        raise IndexError("Index i is out of bounds for criteria_values.")

    # Lọc ra top 3 tiêu chí ưu tiên
    current_criteria_values = criteria_values[i]
    sorted_indices = np.argsort(weights)[-3:]

    # Tiêu chí tích cực
    positive_contribution = np.dot(
        [weights[0], weights[2], weights[5]],
        [current_criteria_values[0], current_criteria_values[2], current_criteria_values[5]]
    ) * np.sum(weights_percent[[0, 2, 5]])

    # Tiêu chí tiêu cực
    negative_contribution = np.dot(
        [weights[1], weights[4]],
        [current_criteria_values[1], current_criteria_values[4]]
    ) * np.sum(weights_percent[[1, 4]])

    # Kiểm tra sự đa dạng hóa
    diversification_contribution = weights[6] * current_criteria_values[6] * weights_percent[6] \
        if 6 in sorted_indices else -weights[6] * current_criteria_values[6] * weights_percent[6]

    # Kiểm tra thời gian đầu tư
    investment_time_contribution = weights[3] * current_criteria_values[3] * weights_percent[3] \
        if 3 in sorted_indices else -weights[3] * current_criteria_values[3] * weights_percent[3]

    return positive_contribution - negative_contribution + diversification_contribution + investment_time_contribution


# Xác định số biến n_bien dựa trên weights
sorted_indices = np.argsort(weights)[-3:]
n_bien = 10 if 6 in sorted_indices else 6

fitness_investment = np.round(np.abs(np.array(
    [fitness_function_investment(i, weights, criteria_values, weights_percent) for i in range(criteria_values.shape[0])
     if i < len(criteria_values)])), 6)

print("Fitness của danh mục:")
for idx, values in enumerate(fitness_investment):
    print(f"Danh mục {idx + 1}: {values}")