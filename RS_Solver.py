import numpy as np
import pandas as pd


# Loading Data for calculating hirsh-index

def hirsh(path):
    print("\n=== Индекс Хирша ===\n")
    print("Дано: \n")
    data = pd.read_excel(path, sheet_name=0, dtype=int)
    print(data)
    print("-----\nРешение:")
    h_index(data)


# Sorting table for h-index

def sorted_h_table(data):
    data["Цитирование"] = data.sort_values("Цитирование", ascending=False)["Цитирование"].values
    print(f'\nSorted Table:\n{data}')
    return data


# Calculating h-index from sorted table

def h_index(data):
    sorted_h_table(data)
    for index, row in data.iterrows():
        if row[1] < row[0]:
            print(f'\nh-index = {data.iloc[index - 1, 1]}')
            break

path = input(
    "Введите путь к файлу, либо название файла с расширением (если он в той же папке, что и скрипт): ")
try:
    hirsh(path)
except FileNotFoundError:
    print(f"ОШИБКА: Не найден Excel-файл: '{path}'. Вероятно, путь был указан некорректно. Проверьте, указано ли расширение файла (после названия файла в конце должно стоять .xlsx)")
    k = input("press Enter to exit")
except OSError:
    print(f"ОШИБКА: Путь не читаем: '{path}'. Вероятно это связано с тем, что вы ввели его через кавычки. Если это так, нужно кавычки убрать.")
    k = input("press Enter to exit")


# Непосредственная оценка

def direct_assessment_loading_data(path):
    print("\n=== Непосредственная оценка ===\n")
    print("Дано: \n")
    data2 = pd.read_excel(path, sheet_name=1, index_col=0)
    print(data2)
    return data2


def direct_assessment(data2):
    print("-----\nРешение:")
    data2_sum = data2.sum().sum()
    print(round(data2.sum() / data2_sum, 2))


data2 = direct_assessment_loading_data(path)
direct_assessment(data2)


def calculate_X(expert_matrix, n_experts):
    """
    Calculate Expected Value matrix from expert matrix

    """

    m = n_experts
    mi = 1
    mj = 0

    bool_matrix_mi = expert_matrix == mi
    bool_matrix_mi = bool_matrix_mi.astype(int)
    mi_matrix = sum(bool_matrix_mi)

    bool_matrix_mj = expert_matrix == mj
    bool_matrix_mj = bool_matrix_mj.astype(int)
    mj_matrix = sum(bool_matrix_mj)

    X = 0.5 + (mi_matrix - mj_matrix) / (2 * m)
    print(f'Матрица математических ожиданий оценок (X): {X}\n')

    return X


def calculate_coefficients(X, epsilon):
    """
    Calculate coefficients from Expected Value matrix.

    """

    K = np.transpose(np.ones(X.shape[1]))
    while True:
        Y = np.dot(X, K)
        Lambda = np.dot(np.ones(X.shape[1]), Y)
        k = (1 / Lambda) * Y
        if max(abs(k - K)) > epsilon:
            K = k
            print(f'Iteration: {K}')
            continue
        else:
            print(f'Итоговые значения коэффициента: {np.round(k, 4)}')
            break


print("\n=== Парная оценка ===\n")
print("Дано: \n")

confidence_interval = float(input("Введите доверительный интервал "))
epsilon = 1 - confidence_interval

n_experts = int(input("Введите кол-во экспертов "))
matrix_dictionary = {}
for expert in range(0, n_experts):
    matrix_dictionary["expert{0}".format(expert + 1)] = pd.read_excel(
        path, sheet_name=2, index_col=0, nrows=2, skiprows=expert * 4).values

expert_matrix = np.stack(tuple(matrix_dictionary.values()))
print(f'\nМатрица экспертных оценок:\n {expert_matrix}\n')

print("-----\nРешение:\n")

X = calculate_X(expert_matrix, n_experts)
calculate_coefficients(X, epsilon)

# Анализ иерархий

def solve_hierarchy(criteria_data, object_matrix, n_criteria, n_objects):
    common_sum = np.sum(np.power(np.prod(criteria_data, axis=n_criteria - 1), 1 / n_criteria))
    LK_array = np.power(np.prod(criteria_data, axis=n_criteria - 1), 1 / n_criteria) / common_sum
    print(f'\nЛокальный приоритет критерия: {LK_array}')

    common_sum_objects = np.sum(np.power(np.prod(object_matrix, axis=n_objects - 1), 1 / n_objects), axis=1)
    LK_objects_array = np.power(np.prod(object_matrix, axis=n_objects - 1), 1 / n_objects) / common_sum_objects[:,
                                                                                             np.newaxis]
    print(f'\nЛокальные приоритеты объектов относительно критериев:\n{LK_objects_array}')
    GK = np.dot(LK_array, LK_objects_array)
    print(f'\nГлобальные приоритеты: {GK}')


print("\n=== Анализ иерархий ===\n")
print("Дано: \n")

n_criteria = int(input("Введите кол-во критериев: "))
n_objects = int(input("Введите кол-во объектов: "))

matrix_dictionary = {}

criteria_data = pd.read_excel(
    path, sheet_name=3, index_col=0, nrows=2, usecols=list(range(0, n_criteria + 1))).values
print(f'\nМатрица с оценкой критериев: \n{criteria_data}')

for criteria in range(0, n_criteria):
    matrix_dictionary["criteria{0}".format(criteria + 1)] = pd.read_excel(
        "research_seminar_problems.xlsx", sheet_name=3, index_col=0, nrows=n_objects,
        skiprows=n_criteria + 2 + criteria * (n_objects + 2)).values

object_matrix = np.stack(tuple(matrix_dictionary.values()))
print(f'\nОценка объектов по критериям:\n {object_matrix}\n')

print("-----\nРешение:")

solve_hierarchy(criteria_data, object_matrix, n_criteria, n_objects)

# Комплексная оценка

def calculate_weighted_scores(matrix, length, V):
    mean_criteria = np.mean(matrix, axis=1)
    mean_criteria_matrix = np.transpose(
        np.vstack([mean_criteria]*length))
    abs_diff = np.sum(np.absolute(matrix - mean_criteria_matrix),
                      axis=1)/(length*mean_criteria)
    print(f"\nМатрица R: {abs_diff}")
    R_sum = np.sum(abs_diff)
    print(f"\nСумма R: {R_sum}")
    Z = abs_diff/R_sum
    print(f"\nМатрица Z: {Z}")
    W = (V+Z)/2
    print(f"\nМатрица W (обобщенные веса критериев): {W}")
    W_matrix = np.transpose(np.vstack([W]*length))
    answer = np.sum(W_matrix*matrix,axis=0)
    print(f'\nОтвет: {answer}')


print("\n=== Метод комплексной оценки ===\n")
print("Дано: \n")

df = pd.read_excel(path, sheet_name=4, index_col=0)
df1 = df.copy()

print(f"\nТаблица: \n{df1}\n")

print(f'Список критериев: {df1.index.to_list()}')

minimize_crit = input(
    "Какие критерии минимизируем? Введите полное название критериев через запятую без пробела и кавычек: ").rsplit(",")

maximize_crit = input(
    "Какие критерии максимизируем? Введите полное название критериев через запятую без пробела и кавычек: ").rsplit(",")

V = input("Введите экспертные коэффициенты (веса) для каждого критерия по порядку (сверху-вниз) через запятую без пробела и кавычек: ").rsplit(",")
V = np.array(V)

try:
    V = V.astype(float)

    for crit in minimize_crit:
        df1.loc[crit] = min(df1.loc[crit])/df1.loc[crit]

    for crit in maximize_crit:
        df1.loc[crit] = df1.loc[crit] / max(df1.loc[crit])

    matrix = df1.values
    length = len(df1.columns)

    print("-----\nРешение:")

    calculate_weighted_scores(matrix, length, V)
except KeyError:
    print(f"ОШИБКА: Некорректно введены критерии. Проверьте, что название указанных критериев в точности совпадает с названиями в списке критериев. Также, проверьте введены ли критерии через запятую без пробела и кавычек.")
    k = input("press Enter to exit")
except ValueError:
    print(f"ОШИБКА: Некорректно введены экспертные коэффициенты.")
    k = input("press Enter to exit")

k=input("press Enter to exit")