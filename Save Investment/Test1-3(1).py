import numpy as np
import random
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import pandas as pd
from tabulate import tabulate

# Tham số cho thuật toán
pop_size = 200
bounds = [-5, 5]
max_iter = 500
n = 4
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
with open('../criteria_data.csv', mode='w', newline='') as file:
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

criteria_data = read_csv_to_array('../criteria_data.csv')  # Đọc dữ liệu từ file CSV

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

# Định nghĩa hàm fitness
def fitness_function(i, weights, criteria_values, weights_percent):
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


def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


# Tính toán fitness cho tất cả các vũ trụ
def calculate_fitness(population):
    return np.array([rosenbrock(ind) for ind in population])


# Tính tỷ lệ chuẩn hóa NI cho MVO
def normalize_inflation_rate(fitness):
    normalized_fitness = 1 / (fitness + 1e-6)  # Tránh chia cho 0
    return normalized_fitness / np.sum(normalized_fitness)


# Lựa chọn bằng cách quay bánh xe
def roulette_wheel_selection(weights):
    cumulative_sum = np.cumsum(weights - np.min(weights))  # Tránh giá trị âm
    normalized_weights = cumulative_sum / cumulative_sum[-1]  # Chuẩn hóa
    random_number = np.random.rand()
    index = np.searchsorted(normalized_weights, random_number)
    return index


# Cập nhật dân số bằng cơ chế của MVO
def update_population_mvo(population, fitness, best_universe, WEP, bounds, TDR, pop_size, n):
    new_population = population.copy()
    best_fitness = np.min(fitness)

    # Tính NI
    NI = normalize_inflation_rate(fitness)
    SU = population[np.argsort(fitness)]  # Các vũ trụ được sắp xếp theo fitness

    for i in range(pop_size):
        if fitness[i] > best_fitness:  # Bỏ qua fitness tốt nhất
            r1 = np.random.rand()
            if r1 < NI[i]:  # Sử dụng NI để so sánh
                white_hole_index = roulette_wheel_selection(-NI)
                new_population[i] = SU[white_hole_index]
            else:  # Cơ chế Wormhole
                r2 = np.random.rand()
                if r2 < WEP:
                    for j in range(n):
                        r3 = np.random.rand()
                        r4 = np.random.rand()

                        # Kiểm tra xem best_universe có phải là scalar không
                        if np.isscalar(best_universe):  # Nếu best_universe là scalar, chuyển nó thành một mảng
                            best_universe = np.full(n, best_universe)

                        # Cập nhật giá trị cho new_population[i][j]
                        if r3 < 0.5:
                            new_population[i][j] = best_universe[j] + TDR * ((bounds[1] - bounds[0]) * r4 + bounds[0])
                        else:
                            new_population[i][j] = best_universe[j] - TDR * ((bounds[1] - bounds[0]) * r4 + bounds[0])
                else:
                    new_population[i] = population[i]

    return new_population



# Cập nhật vị trí theo COA
def update_population_coa(population, best_solution, bounds):
    N, m = population.shape
    r = random.random()
    I = random.randint(1, 2)

    new_population = np.zeros_like(population)

    # Iguana là cá thể tốt nhất (best solution)
    iguana = best_solution

    for i in range(N):
        if i >= N / 2:
            # Coati rơi xuống đất và di chuyển để bắt Iguana
            iguana_G = np.random.uniform(bounds[0], bounds[1], m)
            if rosenbrock(iguana_G) < rosenbrock(population[i]):
                new_population[i] = population[i] + r * (iguana_G - I * population[i])
            else:
                new_population[i] = population[i] + r * (population[i] - iguana_G)
        else:
            # Coati trên cây đe dọa Iguana
            new_population[i] = population[i] + r * (iguana - I * population[i])

        # Áp dụng yếu tố nhiễu
        noise = np.random.uniform(-0.001, 0.001, m)
        new_population[i] += noise

    return new_population

population = np.random.uniform(bounds[0], bounds[1], (pop_size, n))
fitness = calculate_fitness(population)

# Thuật toán lai COA-MVO
def hybrid_coa_mvo(population, fitness ,bounds, max_iter, n, switch_iter):
    # Khởi tạo dân số và tính fitness ban đầu
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    WEPmin, WEPmax = 0.2, 0.8
    TDR = 1.0

    for iter in range(max_iter):
        WEP = WEPmin + (iter / max_iter) * (WEPmax - WEPmin)
        TDR = 1 - (iter / max_iter)

        # Cập nhật dân số theo chiến lược kết hợp
        if iter < switch_iter:
            # Giai đoạn đầu: chủ yếu MVO
            population = update_population_mvo(population, fitness, best_idx, WEP, bounds, TDR, pop_size, n)
        else:
            # Giai đoạn sau: chuyển sang COA
            population = update_population_coa(population, best_solution, bounds)

        # Tính toán lại fitness
        fitness = calculate_fitness(population)

        # Cập nhật cá thể tốt nhất
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx].copy()
    return best_fitness


# Tên các danh mục đầu tư tương ứng với chỉ số
investment_names = [
    "Đầu tư vàng",
    "Bất động sản",
    "Gửi ngân hàng",
    "Cổ phiếu Top thị trường",
    "Cổ phiếu không thuộc top",
    "Trái phiếu doanh nghiệp",
    "Trái phiếu chính phủ",
    "Quỹ cổ phiếu",
    "Quỹ trái phiếu",
    "Quỹ hỗn hợp cân bằng",
    "Crypto (BTC, ETH)",
    "ALT Coin",
    "Meme Coin",
    "Shit Coin",
    "Quỹ đầu tư bất động sản (REITs)",
    "Chứng chỉ tiền gửi (CDs)",
    "Hợp đồng tương lai",
    "Hợp đồng quyền chọn",
    "Vốn đầu tư mạo hiểm",
    "Đầu tư công nghệ tài chính (Fintech)"
]

best_fitness = hybrid_coa_mvo(population, fitness ,bounds, max_iter, n, switch_iter)
best_fitness_2 = best_fitness.copy()
# Giá trị fitness tương ứng với các danh mục đầu tư
fitness_investment = best_fitness_2 + np.array([fitness_function(i, weights, criteria_values, weights_percent) for i in range(criteria_values.shape[0]) if i < len(criteria_values)])

# Tạo danh sách thay thế cho từng danh mục đầu tư
replacement_map = {
    "ALT Coin": ["Meme Coin", "Shit Coin"],
    "Meme Coin": ["ALT Coin", "Shit Coin"],
    "Crypto (BTC, ETH)": ["ALT Coin", "Meme Coin", "Shit Coin"],
    "Cổ phiếu Top thị trường": ["Cổ phiếu không thuộc top", "Quỹ cổ phiếu"],
    "Bất động sản": ["Quỹ đầu tư bất động sản (REITs)"],
}

# Xác định loại danh mục đầu tư
categories = {
    "Vàng": ["Đầu tư vàng"],
    "Bất động sản": ["Bất động sản", "Quỹ đầu tư bất động sản (REITs)"],
    "Tiền gửi": ["Gửi ngân hàng", "Chứng chỉ tiền gửi (CDs)"],
    "Cổ phiếu": ["Cổ phiếu Top thị trường", "Cổ phiếu không thuộc top", "Quỹ cổ phiếu"],
    "Trái phiếu": ["Trái phiếu doanh nghiệp", "Trái phiếu chính phủ", "Quỹ trái phiếu"],
    "Quỹ": ["Quỹ hỗn hợp cân bằng", "Quỹ cổ phiếu", "Quỹ trái phiếu", "Quỹ đầu tư bất động sản (REITs)"],
    "Crypto": ["Crypto (BTC, ETH)", "ALT Coin", "Meme Coin", "Shit Coin"],
    "Hợp đồng": ["Hợp đồng tương lai", "Hợp đồng quyền chọn"],
    "Đầu tư mạo hiểm": ["Vốn đầu tư mạo hiểm", "Đầu tư công nghệ tài chính (Fintech)"]
}

# Phân loại danh mục đầu tư
classified_investments = defaultdict(list)

for investment, fitness_investment in zip(investment_names, fitness_investment):
    for category, names in categories.items():
        if investment in names:
            classified_investments[category].append((investment, fitness_investment))
            break

# Sắp xếp danh sách theo giá trị fitness (từ cao đến thấp) cho từng loại
for category in classified_investments:
    classified_investments[category].sort(key=lambda x: x[1], reverse=True)

# Tính tổng fitness của tất cả danh mục đầu tư
total_fitness = sum([inv[1] for investments in classified_investments.values() for inv in investments])

# Kiểm tra xem chỉ số "Đa dạng hóa" có nằm trong top 3 trọng số không
diversification_index = 6  # Chỉ số của "Đa dạng hóa"
top_3_indices = np.argsort(weights)[-3:]  # Lấy chỉ số của 3 trọng số lớn nhất
is_diversification_top3 = diversification_index in top_3_indices

# Hàm hiển thị kết quả với ràng buộc không có danh mục giống nhau
def display_investments_by_category(classified_investments, total_fitness, is_diversification_top3):
    # Số lượng danh mục cần hiển thị
    display_count = 5 if is_diversification_top3 else 3

    selected_investments = []  # Danh sách để lưu các danh mục đầu tư đã chọn
    used_categories = set()  # Theo dõi các loại danh mục đã sử dụng

    for category, investments in classified_investments.items():
        for investment, fitness_investment in investments:
            # Nếu Đa dạng hóa nằm trong top 3, kiểm tra xem danh mục đã chọn chưa
            if is_diversification_top3 and investment in [inv[0] for inv in selected_investments]:
                # Tìm danh mục thay thế
                alternatives = replacement_map.get(investment, [])
                found_alternative = False
                for alternative in alternatives:
                    if alternative not in [inv[0] for inv in selected_investments]:
                        investment = alternative
                        fitness_investment *= 0.2  # Giảm 30% fitness cho danh mục thay thế
                        found_alternative = True
                        break

                # Nếu không tìm thấy danh mục thay thế, bỏ qua
                if not found_alternative:
                    continue

            # Nếu chưa chọn và còn chỗ hiển thị, thêm vào danh sách hiển thị
            # Kiểm tra xem nhóm danh mục đã được chọn chưa
            category_key = next((cat for cat, names in categories.items() if investment in names), None)
            if category_key and category_key not in used_categories:
                selected_investments.append((investment, fitness_investment))  # Lưu danh mục đã chọn
                used_categories.add(category_key)  # Đánh dấu danh mục đã sử dụng

                # Dừng nếu đã đủ số danh mục cần hiển thị
                if len(selected_investments) == display_count:
                    break

        # Dừng nếu đã đủ số danh mục cần hiển thị
        if len(selected_investments) == display_count:
            break

    # Tính tổng fitness đã chọn
    total_selected_fitness = sum(inv[1] for inv in selected_investments)

    # Sắp xếp danh sách theo tỷ lệ phần trăm giảm dần
    selected_investments.sort(key=lambda x: x[1], reverse=True)

    # Lợi nhuận và rủi ro ước tính cho từng danh mục
    criteria_estimated = [
        "5 - 8% (1 năm)", # Đầu tư vàng
        "8 - 12% (1 năm)", # Bất động sản
        "1 - 3% (1 năm)", # Gửi ngân hàng
        "8 - 12% (1 năm)", # Cổ phiếu Top thị trường
        "5 - 10% (1 năm)", # Cổ phiếu không thuộc top
        "5 - 7% (1 năm)", #  Trái phiếu doanh nghiệp
        "2 - 4% (1 năm)", # Trái phiếu chính phủ
        "7 - 10% (1 năm)", # Quỹ cổ phiếu
        "3 - 5% (1 năm)", # Quỹ trái phiếu
        "5 - 8% (1 năm)", # Quỹ hỗn hợp cân bằng
        "10 - 20% (1 năm)", # Crypto (BTC, ETH)
        "15 - 25% (1 năm)", # ALT Coin
        "20 - 50% (1 năm)", # Meme Coin
        "30 - 100% (1 năm)", #Shit Coin
        "6 - 9% (1 năm)", # Quỹ đầu tư bất động sản (REITs)
        "1 - 2% (1 năm)", # Chứng chỉ tiền gửi (CDs)
        "5 - 15% (1 năm)", # Hợp đồng tương lai
        "10 - 30% (1 năm)", # Hợp đồng quyền chọn
        "15 - 25% (1 năm)", # Vốn đầu tư mạo hiểm
        "10 - 20% (1 năm)",  # Đầu tư công nghệ tài chính (Fintech)
    ]

    risk_estimated = [
        "5 - 10% (1 năm)", # Đầu tư vàng
        "10 - 15% (1 năm)", # Bất động sản
        "1 - 2% (1 năm)", # Gửi ngân hàng
        "10 - 15% (1 năm)", # Cổ phiếu Top thị trường
        "15 - 20% (1 năm)", # Cổ phiếu không thuộc top
        "8 - 12% (1 năm)", #  Trái phiếu doanh nghiệp
        "3 - 5% (1 năm)", # Trái phiếu chính phủ
        "10 - 15% (1 năm)", # Quỹ cổ phiếu
        "5 - 8% (1 năm)", # Quỹ trái phiếu
        "8 - 12% (1 năm)", # Quỹ hỗn hợp cân bằng
        "25 - 40% (1 năm)", # Crypto (BTC, ETH)
        "30 - 50% (1 năm)", # ALT Coin
        "40 - 60% (1 năm)", # Meme Coin
        "50 - 80% (1 năm)", #Shit Coin
        "10 - 15% (1 năm)", # Quỹ đầu tư bất động sản (REITs)
        "1 - 2% (1 năm)", # Chứng chỉ tiền gửi (CDs)
        "15 - 30% (1 năm)", # Hợp đồng tương lai
        "20 - 50% (1 năm)", # Hợp đồng quyền chọn
        "30 - 60% (1 năm)", # Vốn đầu tư mạo hiểm
        "15 - 25% (1 năm)",  # Đầu tư công nghệ tài chính (Fintech)
    ]

    # Tạo danh sách để chứa dữ liệu bảng
    table_data = []

    # Tính toán tỷ lệ phần trăm và thêm vào dữ liệu bảng
    for idx, (investment, fitness_investment) in enumerate(selected_investments):
        percent = (fitness_investment / total_selected_fitness * 100) if total_selected_fitness > 0 else 0
        table_data.append([
            f"{idx + 1}",
            investment,
            f"{percent:2.2f}%",
            criteria_estimated[idx],
            risk_estimated[idx]
        ])

    # Hiển thị bảng
    print("\nTop danh mục đầu tư theo loại và tỷ lệ phần trăm của từng danh mục:")
    print(tabulate(table_data, headers=["STT","Danh mục đầu tư", "Trọng số (%)", "Lợi nhuận ước tính", "Rủi ro ước tính"], tablefmt="fancy_grid"))

# Hiển thị ma trận so sánh
# Tên cột
headers = ["Criterias", "Expected Return", "Risk", "Liquidity", "Investment Time", "Cost", "Stability of Cash Flow", "Diversification"]
index = ["Expected Return", "Risk", "Liquidity", "Investment Time", "Cost", "Stability of Cash Flow", "Diversification"]
data = []

for i in range(comparison_matrix.shape[0]):
    temp_val = list(comparison_matrix[i, :])
    temp_val.insert(0, index[i])
    data.append(temp_val)

print("Kết quả Ma Trận AHP")
print(tabulate(data, headers=headers, tablefmt='fancy_grid', numalign="center"))

print("\n")
print("Trọng số cuối cùng cho các tiêu chí")
index = ["Expected Return", "Risk", "Liquidity", "Investment Horizon", "Cost", "Stability of Cash Flow", "Diversification"]

data2 = []
for i in range(len(weights)):
    data2.append([index[i], weights[i]])

# Sắp xếp dữ liệu từ lớn đến nhỏ theo giá trị weights
data2_sorted = sorted(data2, key=lambda x: x[1], reverse=True)

print(tabulate(data2_sorted, headers=["Criterias", "Criteria Weights"], tablefmt='fancy_grid', numalign="center"))


# Gọi hàm hiển thị kết quả
display_investments_by_category(classified_investments, total_fitness, is_diversification_top3)




