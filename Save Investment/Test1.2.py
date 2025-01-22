import numpy as np
import random
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import pandas as pd
from tabulate import tabulate

# Tham số cho thuật toán
pop_size = 2000
bounds = [-5, 5]
max_iter = 1000
switch_iter = 500

tc1 = random.randint(1, 20)
tc2 = random.randint(1, 20)
tc3 = random.randint(1, 20)
tc4 = random.randint(1, 20)
tc5 = random.randint(1, 20)
tc6 = random.randint(1, 20)
tc7 = random.randint(1, 20)

# Dữ liệu => Tổng hợp điể từ web
data = [13, 2, 5, 5, 7, 1, 7] # Thay đổi sau mỗi lần vote
#data = [tc1, tc2, tc3, tc4, tc5, tc6, tc7]

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


# Xác định số biến n_bien dựa trên weights
sorted_indices = np.argsort(weights)[-3:]
n_bien = 10 if 6 in sorted_indices else 6


# Hàm tạo 1 cá thể
def create_individual(n_bien):
    n = n_bien // 2  # Số lượng (xi, yi)
    individual = [0] * n_bien  # Cá thể chứa num_vars giá trị
    # Tạo các giá trị xi duy nhất ở vị trí chẵn
    xi_values = random.sample(range(20), n)  # Lấy n giá trị không trùng từ [0, 19]
    for i, xi in enumerate(xi_values):
        individual[2 * i] = xi
    # Tạo các giá trị yi bằng phân phối Dirichlet
    yi_values = np.random.dirichlet(np.ones(n)) * 100  # Tổng các giá trị luôn là 100
    for i, yi in enumerate(yi_values):
        individual[2 * i + 1] = np.round(yi, 2)
    return individual


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
            else:
                fitness_xi = 0  # Gán giá trị mặc định nếu xi không hợp lệ

            # Đảm bảo yi_value không phải None
            if yi_value is not None:
                individual_fitness += fitness_xi * yi_value  # Tính đóng góp của xi và yi

        fitness_values.append(individual_fitness)  # Lưu tổng fitness của cá thể này vào danh sách

    return fitness_values



population = [create_individual(n_bien) for _ in range(pop_size)]

fitness_investment = np.array(
    [fitness_function_investment(i, weights, criteria_values, weights_percent) for i in range(criteria_values.shape[0])
     if i < len(criteria_values)])

# Tính fitness cho từng cá thể trong quần thể
fitness = np.abs(calculate_fitness(fitness_investment, population))


# Chuẩn hóa tỷ lệ NI cho MVO
def normalize_inflation_rate(fitness):
    if not isinstance(fitness, np.ndarray):
        fitness = np.array(fitness)
    total_fitness = np.sum(fitness)
    if total_fitness == 0:
        return np.ones_like(fitness) / len(fitness)

    epsilon = 1e-6
    normalized_fitness = 1 / (fitness + epsilon)

    normalized_fitness /= np.sum(normalized_fitness)
    return normalized_fitness


# Lựa chọn bằng cách quay bánh xe
def roulette_wheel_selection(m):
    cumulative_sum = np.cumsum(m - np.min(m))  # Tránh giá trị âm
    normalized_weights = cumulative_sum / cumulative_sum[-1]  # Chuẩn hóa
    random_number = np.random.rand()
    index = np.searchsorted(normalized_weights, random_number)
    return index

# Cập nhật dân số bằng cơ chế của MVO
def update_population_mvo(population, fitness, best_solution, WEP, bounds, TDR):
    new_population = population.copy()
    best_fitness = np.min(fitness)

    # Tính NI
    NI = normalize_inflation_rate(fitness)
    sorted_indices = np.argsort(fitness)
    SU = [population[i] for i in sorted_indices]  # Sắp xếp theo fitness

    for i in range(len(population)):
        if fitness[i] > best_fitness:  # Bỏ qua cá thể tốt nhất
            r1 = np.random.rand()
            if r1 < NI[i]:  # Cơ chế White Hole
                white_hole_index = roulette_wheel_selection(-NI)
                new_population[i] = SU[white_hole_index]
            else:  # Cơ chế Wormhole
                n = len(best_solution) // 2  # Số lượng (xi, yi)
                new_individual = population[i][:]  # Khởi tạo cá thể mới từ cá thể gốc (giữ nguyên xi)
                new_yis = []
                r2 = np.random.uniform(0, 1)
                if r2 < WEP:
                    for j in range(n):
                        r3 = np.random.uniform(0, 1)
                        r4 = np.random.uniform(0, 1)
                        if r3 < 0.5:
                            yi_value = np.round(
                                best_solution[2 * j + 1] + TDR * ((bounds[1] - bounds[0]) * r4 + bounds[0]), 2)
                            new_yis.append(yi_value)
                        else:
                            yi_value = np.round(
                                best_solution[2 * j + 1] - TDR * ((bounds[1] - bounds[0]) * r4 + bounds[0]), 2)
                            new_yis.append(yi_value)

                    # Đảm bảo tổng các giá trị yi = 100%
                    total_yi = np.sum(new_yis)
                    if total_yi != 100:
                        diff = 100 - total_yi
                        new_yis = [yi + (diff / len(new_yis)) for yi in new_yis]

                    # Kiểm tra lại kích thước của new_individual trước khi cập nhật
                    for j in range(n):
                        if 2 * j + 1 < len(new_individual):
                            new_individual[2 * j + 1] = new_yis[j]

                    new_population[i] = new_individual
        else:
            new_population[i] = population[i]  # Giữ nguyên cá thể gốc nếu là tốt nhất

    return new_population


# Cập nhật vị trí theo COA
def update_population_coa(new_population, fitness, best_solution):
    r = random.random()
    I = random.choice([1, 2])

    new_population_coa = new_population.copy()

    # Iguana là cá thể tốt nhất (best solution)
    iguana = best_solution

    for i in range(pop_size):
        if i >= pop_size / 2:  # Iguana fall down
            iguana_G = create_individual(n_bien)

            individual_fitness_G = 0
            for j in range(0, len(iguana_G), 2):  # Duyệt qua các giá trị xi và yi (xi: chẵn, yi: lẻ)
                xi_value_iguana = iguana_G[j]  # Lấy xi (vị trí chẵn)
                yi_value_iguana = iguana_G[j + 1]  # Lấy yi (vị trí lẻ)

                # Đảm bảo xi_value là số nguyên và nằm trong phạm vi hợp lệ
                if isinstance(xi_value_iguana, int) and 0 <= xi_value_iguana < len(fitness_investment):
                    fitness_xi = fitness_investment[xi_value_iguana]  # Giá trị fitness từ danh sách/ma trận
                    individual_fitness_G += fitness_xi * yi_value_iguana  # Tính đóng góp của xi và yi

            for j in range(0, len(new_population[i]), 2):  # Duyệt qua từng xi, yi
                n = n_bien // 2
                new_individual = [None] * (2 * n)
                xi_value = new_population[i][j]  # Lấy xi
                new_individual[j + 1] = new_population[i][j + 1]  # Lấy yi

                retries = 0
                found = False
                # Gán giá trị mặc định cho new_xi
                # new_xi = -1
                max_retries = 100

                used_xi = set()  # Tập hợp để theo dõi các giá trị xi đã được tạo
                new_xis = []  # Danh sách tạm thời để lưu các giá trị new_xi

                # Tạo danh sách các giá trị khả thi trước
                possible_xis = list(range(0, 20))  # Giới hạn phạm vi từ 0 đến 19

                for h in range(0, len(iguana_G), 2):
                    # Cập nhật xi và yi dựa trên điều kiện fitness
                    if individual_fitness_G < fitness[i]:
                        while retries < max_retries:
                            # Cập nhật xi và yi khi fitness của iguana_G nhỏ hơn fitness cá thể hiện tại
                            new_xi = xi_value + r * (iguana_G[h] - I * np.ones_like(xi_value))
                            new_xi = int(round(new_xi))

                            if new_xi < 0:
                                new_xi = 0
                            elif new_xi > 19:
                                new_xi = 19

                            # Nếu không trùng, thêm vào danh sách và thoát khỏi vòng lặp
                            if new_xi not in used_xi and new_xi in possible_xis:
                                used_xi.add(new_xi)
                                new_xis.append(new_xi)
                                found = True
                                break
                            retries += 1

                            if not found:  # Nếu không tìm thấy sau max_retries lần thử, gán giá trị mặc định
                                new_xi = random.choice([xi for xi in possible_xis if xi not in used_xi])
                                used_xi.add(new_xi)
                                new_xis.append(new_xi)

                            # Gán giá trị mới vào cá thể
                            for k, new_xi in enumerate(new_xis):
                                new_individual[2 * k] = new_xi

                            # Thêm cá thể mới vào quần thể
                            new_population_coa.append(new_individual)

                    else:
                        while retries < max_retries:
                            # Cập nhật xi và yi khi fitness của iguana_G lớn hơn hoặc bằng fitness cá thể hiện tại
                            new_xi = xi_value + r * (np.ones_like(xi_value) - iguana_G[h])
                            new_xi = int(round(new_xi))

                            if new_xi < 0:
                                new_xi = 0
                            elif new_xi > 19:
                                new_xi = 19

                            # Nếu không trùng, thêm vào danh sách và thoát khỏi vòng lặp
                            if new_xi not in used_xi and new_xi in possible_xis:
                                used_xi.add(new_xi)
                                new_xis.append(new_xi)
                                found = True
                                break
                            retries += 1

                            if not found:  # Nếu không tìm thấy sau max_retries lần thử, gán giá trị mặc định
                                new_xi = random.choice([xi for xi in possible_xis if xi not in used_xi])
                                used_xi.add(new_xi)
                                new_xis.append(new_xi)

                            # Gán giá trị mới vào cá thể
                            for k, new_xi in enumerate(new_xis):
                                new_individual[2 * k] = new_xi

                            # Thêm cá thể mới vào quần thể
                            new_population_coa.append(new_individual)

        else:
            # Coati trên cây đe dọa Iguana
            for j in range(0, len(population[i]), 2):
                n = n_bien // 2
                new_individual = [None] * (2 * n)
                xi_value = population[i][j]  # Lấy xi
                yi_value = population[i][j + 1]  # Lấy yi

                retries = 0
                found = False
                # Gán giá trị mặc định cho new_xi
                # new_xi = -1
                max_retries = 100

                used_xi = set()  # Tập hợp để theo dõi các giá trị xi đã được tạo
                new_xis = []  # Danh sách tạm thời để lưu các giá trị new_xi

                # Tạo danh sách các giá trị khả thi trước
                possible_xis = list(range(0, 20))  # Giới hạn phạm vi từ 0 đến 19

                for h in range(0, len(iguana), 2):
                    while retries < max_retries:
                        new_xi = xi_value + r * (iguana[h] - I * np.ones_like(xi_value))
                        new_xi = int(round(new_xi))

                        if new_xi < 0:
                            new_xi = 0
                        elif new_xi > 19:
                            new_xi = 19

                        # Nếu không trùng, thêm vào danh sách và thoát khỏi vòng lặp
                        if new_xi not in used_xi and new_xi in possible_xis:
                            used_xi.add(new_xi)
                            new_xis.append(new_xi)
                            found = True
                            break
                        retries += 1

                        if not found:  # Nếu không tìm thấy sau max_retries lần thử, gán giá trị mặc định
                            new_xi = random.choice([xi for xi in possible_xis if xi not in used_xi])
                            used_xi.add(new_xi)
                            new_xis.append(new_xi)

                        # Gán giá trị mới vào cá thể
                        for k, new_xi in enumerate(new_xis):
                            new_individual[2 * k] = new_xi

                        # Thêm cá thể mới vào quần thể
                        new_population_coa.append(new_individual)

        return new_population_coa


# Thuật toán lai COA-MVO
def hybrid_coa_mvo(population, fitness, bounds, max_iter, switch_iter, fitness_investment):
    # Khởi tạo dân số và tính fitness ban đầu
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = np.min(fitness)

    WEPmin, WEPmax = 0.2, 0.8
    TDR = 1.0

    for iter in range(max_iter):
        WEP = WEPmin + (iter / max_iter) * (WEPmax - WEPmin)
        TDR = 1 - (iter / max_iter)
        new_population = []

        # Cập nhật dân số theo chiến lược kết hợp
        if iter < switch_iter:
            # Giai đoạn đầu: chủ yếu MVO
            new_population = update_population_mvo(population, fitness, best_solution, WEP, bounds, TDR)
        else:
            # Giai đoạn sau: chuyển sang COA
            new_population = update_population_coa(new_population, fitness, best_solution)

        # Cập nhật cá thể tốt nhất
        fitness_current = calculate_fitness(fitness_investment, new_population)
        current_best_idx = np.argmin(fitness_current)
        current_best_fitness = fitness_current[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            # Ensure population is not empty
            if len(population) > 0:
                best_solution = population[current_best_idx].copy()
            else:
                print("Error: Population is empty")
        return best_fitness, best_solution


best_fitness_investment, best_solution_investment = hybrid_coa_mvo(population, fitness, bounds, max_iter, switch_iter,
                                                                   fitness_investment)
"""
best_idx = -1
print("Quần thể đã tạo:")
for idx, values in enumerate(population):
    print(f"Cá thể {idx + 1}: {values}")
    print("\n")
    if values == best_solution_investment:
        best_idx = idx + 1

print(f"- Cá thể {best_idx} là cá thể tốt nhất: {best_solution_investment}")
"""

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

# Lợi nhuận và rủi ro ước tính cho từng danh mục
criteria_estimated = [
    "5 - 8% (1 năm)",  # Đầu tư vàng
    "8 - 12% (1 năm)",  # Bất động sản
    "1 - 3% (1 năm)",  # Gửi ngân hàng
    "8 - 12% (1 năm)",  # Cổ phiếu Top thị trường
    "5 - 10% (1 năm)",  # Cổ phiếu không thuộc top
    "5 - 7% (1 năm)",  # Trái phiếu doanh nghiệp
    "2 - 4% (1 năm)",  # Trái phiếu chính phủ
    "7 - 10% (1 năm)",  # Quỹ cổ phiếu
    "3 - 5% (1 năm)",  # Quỹ trái phiếu
    "5 - 8% (1 năm)",  # Quỹ hỗn hợp cân bằng
    "10 - 20% (1 năm)",  # Crypto (BTC, ETH)
    "15 - 25% (1 năm)",  # ALT Coin
    "20 - 50% (1 năm)",  # Meme Coin
    "30 - 100% (1 năm)",  # Shit Coin
    "6 - 9% (1 năm)",  # Quỹ đầu tư bất động sản (REITs)
    "1 - 2% (1 năm)",  # Chứng chỉ tiền gửi (CDs)
    "5 - 15% (1 năm)",  # Hợp đồng tương lai
    "10 - 30% (1 năm)",  # Hợp đồng quyền chọn
    "15 - 25% (1 năm)",  # Vốn đầu tư mạo hiểm
    "10 - 20% (1 năm)",  # Đầu tư công nghệ tài chính (Fintech)
]

risk_estimated = [
        "5 - 10% (1 năm)",  # Đầu tư vàng
        "10 - 15% (1 năm)",  # Bất động sản
        "1 - 2% (1 năm)",  # Gửi ngân hàng
        "10 - 15% (1 năm)",  # Cổ phiếu Top thị trường
        "15 - 20% (1 năm)",  # Cổ phiếu không thuộc top
        "8 - 12% (1 năm)",  # Trái phiếu doanh nghiệp
        "3 - 5% (1 năm)",  # Trái phiếu chính phủ
        "10 - 15% (1 năm)",  # Quỹ cổ phiếu
        "5 - 8% (1 năm)",  # Quỹ trái phiếu
        "8 - 12% (1 năm)",  # Quỹ hỗn hợp cân bằng
        "25 - 40% (1 năm)",  # Crypto (BTC, ETH)
        "30 - 50% (1 năm)",  # ALT Coin
        "40 - 60% (1 năm)",  # Meme Coin
        "50 - 80% (1 năm)",  # Shit Coin
        "10 - 15% (1 năm)",  # Quỹ đầu tư bất động sản (REITs)
        "1 - 2% (1 năm)",  # Chứng chỉ tiền gửi (CDs)
        "15 - 30% (1 năm)",  # Hợp đồng tương lai
        "20 - 50% (1 năm)",  # Hợp đồng quyền chọn
        "30 - 60% (1 năm)",  # Vốn đầu tư mạo hiểm
        "15 - 25% (1 năm)",  # Đầu tư công nghệ tài chính (Fintech)
]

def display_list_investments(investment_names, best_solution_investment):
    for i in range(0, len(best_solution_investment), 2):
        selected_investments = defaultdict(list)
        idx_investment = best_solution_investment[i]
        criteria_value_investment = best_solution_investment[i + 1]

        if not (0 <= idx_investment < len(investment_names)):
            idx_investment = random.randint(0, len(investment_names) - 1)

        investment_name = investment_names[idx_investment]

        selected_investments[investment_name].append(criteria_value_investment)

        # Sắp xếp danh sách theo giá trị tiêu chí
        for investment in selected_investments:
            selected_investments[investment].sort(reverse=True)
    table_data = []

    for idx, (investment_name, weights_investment) in enumerate(selected_investments.items()):
        avg_weight = sum(weights_investment) / len(weights_investment) if weights_investment else 0
        table_data.append([
            idx + 1,
            investment_name,
            f"{avg_weight:.2f}%",
            criteria_estimated[idx],
            risk_estimated[idx]
        ])

    # Hiển thị bảng
    print("\nTop danh mục đầu tư theo loại và tỷ lệ phần trăm của từng danh mục:")
    print(tabulate(table_data,
                       headers=["STT", "Danh mục đầu tư", "Trọng số (%)", "Lợi nhuận ước tính", "Rủi ro ước tính"],
                       tablefmt="fancy_grid"))

display_list_investments(investment_names, best_solution_investment)