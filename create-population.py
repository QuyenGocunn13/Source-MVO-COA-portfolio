import numpy as np
import random
import csv

# Tham số cho thuật toán
pop_size = 200
bounds = [-5, 5]
max_iter = 500
#n = 4
switch_iter = 250

tc1 = random.randint(1, 20)
tc2 = random.randint(1, 20)
tc3 = random.randint(1, 20)
tc4 = random.randint(1, 20)
tc5 = random.randint(1, 20)
tc6 = random.randint(1, 20)
tc7 = random.randint(1, 20)

# Dữ liệu => Tổng hợp điể từ web
#data = [13, 2, 5, 5, 7, 1, 7] # Thay đổi sau mỗi lần vote
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
#criteria_values = np.random.rand(20, len(weights))

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
    if i < 0 or i >= criteria_values.shape[0]:
        raise IndexError("Index i is out of bounds for criteria_values.")

    # Lọc ra top 3 tiêu chí ưu tiên
    current_criteria_values = criteria_values[i]
    sorted_indices = np.argsort(weights)[-3:]

    # Tiêu chí tích cực
    positive_contribution = np.dot(
        [weights[0], weights[3], weights[5]],
        [current_criteria_values[0], current_criteria_values[2], current_criteria_values[5]]
    ) * np.sum(weights_percent[[0, 3, 5]])

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


# Hàm tính toán fitness cho từng cá thể
def calculate_fitness(fitness_investment, population):
    fitness_values = []  # Danh sách chứa fitness của từng cá thể

    for individual in population:
        individual_fitness = 0
        for j in range(0, len(individual), 2):  # Duyệt qua các giá trị xi và yi (xi: chẵn, yi: lẻ)
            xi_value = individual[j]  # Lấy xi (vị trí chẵn)
            yi_value = individual[j + 1]  # Lấy yi (vị trí lẻ)

            # Đảm bảo xi_value là số nguyên và nằm trong phạm vi hợp lệ
            if isinstance(xi_value, int) and 0 <= xi_value < len(fitness_investment):
                fitness_xi = fitness_investment[xi_value]  # Giá trị fitness từ danh sách/ma trận
                individual_fitness += fitness_xi * yi_value  # Tính đóng góp của xi và yi

        fitness_values.append(individual_fitness)  # Lưu tổng fitness của cá thể này vào danh sách

    return fitness_values

# Xác định số biến n_bien dựa trên weights
sorted_indices = np.argsort(weights)[-3:]
n_bien = 10 if 6 in sorted_indices else 6

# Hàm tạo quần thể
def create_population(pop_size, n_bien):
    population = []  # Quần thể chứa các cá thể

    # Khởi tạo n cá thể
    for _ in range(pop_size):
        individual = []  # Mỗi cá thể

        # Tạo các xi (số ngẫu nhiên từ 0 đến 19) và yi (giá trị ngẫu nhiên sao tổng = 100)
        sum_yi = 100  # Tổng yi cho mỗi cá thể
        for j in range(n_bien):
            if j % 2 == 0:  # Vị trí chẵn: xi
                xi = random.randint(0, 19)  # Giá trị xi ngẫu nhiên
                individual.append(xi)
            else:  # Vị trí lẻ: yi
                if j == n_bien - 1:  # If it's the last element, assign the remaining sum
                    yi = sum_yi
                else:
                    # Ensure valid range for yi while leaving enough for the remaining elements
                    max_value = min(sum_yi - (n_bien - j - 1), 99)
                    yi = random.randint(1, max_value)
                sum_yi -= yi
                individual.append(yi)

        # Thêm cá thể vào quần thể
        population.append(individual)

    return population

population = create_population(pop_size, n_bien)

fitness_investment = np.array([fitness_function_investment(i, weights, criteria_values, weights_percent) for i in range(criteria_values.shape[0]) if i < len(criteria_values)])

# Tính fitness cho từng cá thể trong quần thể
fitness = calculate_fitness(fitness_investment, population)