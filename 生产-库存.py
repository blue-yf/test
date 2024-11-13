import numpy as np
import matplotlib.pyplot as plt
import random
import time

# 生产模型类
class ProductionModel:
    def __init__(self, n, f, k, P, D, H_W, H_N, H_A, molds, X_alpha, d):
        self.n = n       # 构件数量
        self.f = f       # 流水线数量
        self.k = k       # 工序数量
        self.P = P       # 生产时间矩阵
        self.D = D       # 交付时间矩阵
        self.H_W = H_W   # 正常工作时间
        self.H_N = H_N   # 非正常工作时间
        self.H_A = H_A   # 允许加班时间
        self.molds = molds  # 每种模具对应的构件
        self.X_alpha = X_alpha # 每种模具的数量
        self.d = d  # 栈堆的最大深度

    # 目标函数1: 计算最大完工时间 (C_makespan)
    def calculate_makespan(self, C):
        return np.max(C[:, :, -1])  # 取所有构件的第6道工序结束时间的最大值

    # 目标函数2: 计算总库存时间 (C_inventory)
    def calculate_inventory(self, C, production_schedule):
        C_inventory = 0
        for line in production_schedule:
            line_idx = line - 1  # 将流水线编号转换为数组索引（从0开始）
            for component in production_schedule[line]:
                component_idx = component - 1  # 将构件编号转换为数组索引（从0开始）
                # 获取该构件在其分配产线上的完工时间
                completion_time = C[line_idx, component_idx, -1]
                # 计算库存时间：交货时间减去完工时间（如果完工时间晚于交货时间，则库存时间为0）
                inventory_time = max(0, self.D[component_idx] - completion_time)
                C_inventory += inventory_time
        return C_inventory

    def simulate_stacking(self, C):
        completion_times = C[:, :, -1].flatten()[:self.n]  # 提取前n个构件的完工时间
        component_indices = np.arange(1, self.n + 1)  # 生成从1到n的构件编号

        # 将完工时间和构件编号配对并排序
        all_components = sorted(zip(component_indices, completion_times), key=lambda x: x[1])

        # 按深度d对构件分组入栈
        stacks = []
        current_stack = []
        for idx, (j, t) in enumerate(all_components):
            current_stack.append(j)  # 存储构件编号（从1开始）

            # 如果栈堆已满，开始新栈堆
            if len(current_stack) == self.d:
                stacks.append(current_stack)
                current_stack = []

        # 如果还有剩余的构件（不满d个），它们应该作为最后一个栈堆
        if current_stack:
            stacks.append(current_stack)

        return stacks

    def calculate_movecount(self, stacks, D):
        C_movecount = 0

        # 确保交货期数组D的形状为(n,)
        if D.ndim == 2:
            D = D.flatten()  # 如果D是二维的，则将其展平为一维

        # 对所有构件按交货期进行排序
        sorted_components = sorted(range(1, self.n + 1), key=lambda j: D[j - 1])

        # 出库并计算移动次数
        for j in sorted_components:
            # 找到构件所在的栈堆
            for stack_idx, stack in enumerate(stacks):
                if j in stack:
                    break

            # 获取构件出库时的栈堆构件数 N_d
            N_d = len(stacks[stack_idx])

            # 计算该构件的出库移动次数 F(i,j,d) - 1
            position_in_stack = stacks[stack_idx].index(j)  # 找到该构件在栈中的位置
            F_ijd = N_d - position_in_stack  # 位置从下到上计算
            C_movecount += F_ijd - 1  # 移动次数为 F(i,j,d) - 1

            # 移除出库构件，并更新栈堆
            stacks[stack_idx].remove(j)

            # 重新更新栈堆中其它构件的移动次数
            # 注意：这只是针对当前栈堆，栈堆中其它构件的移动次数需要根据栈堆新深度重新计算
            for i, component in enumerate(stacks[stack_idx]):
                N_d = len(stacks[stack_idx])  # 当前栈堆的新深度
                F_ijd = N_d - i  # 从下到上重新赋值
                C_movecount += F_ijd - 1  # 加上新的移动次数

        return C_movecount

    # 检查所有约束条件
    def check_constraints(self, S, C):
        # 约束1: 工序顺序
        for i in range(self.f):
            for j in range(self.n):
                for k in range(1, self.k):
                    if S[i, j, k] < C[i, j, k - 1]:
                        return False

        # 约束2: 流水线构件顺序
        for i in range(self.f):
            for j in range(1, self.n):
                for k in range(self.k):
                    if S[i, j, k] < C[i, j - 1, k]:
                        return False

        # 约束3: 第3道工序完工时间
        for i in range(self.f):
            for j in range(self.n):
                T = max(C[i, j - 1, 3], C[i, j, 2]) + self.P[i, j, 3]
                day = int(T // 24)
                if T <= 24 * day + self.H_W + self.H_A:
                    if C[i, j, 3] < T:
                        return False
                else:
                    if C[i, j, 3] < 24 * (day + 1) + self.P[i, j, 3]:
                        return False

        # 约束4: 第4道工序完工时间
        for i in range(self.f):
            for j in range(self.n):
                T = max(C[i, j - 1, 4], C[i, j, 3]) + self.P[i, j, 4]
                day = int(T // 24)
                if T <= 24 * day + self.H_W:
                    if C[i, j, 4] < T:
                        return False
                elif T < 24 * (day + 1):
                    if C[i, j, 4] < 24 * (day + 1):
                        return False
                else:
                    if C[i, j, 4] < T:
                        return False

        # 约束5: 模具释放等待
        for mold_type, components in self.molds.items():
            for j in range(len(components)):
                if j >= self.X_alpha[mold_type]:
                    S_wait = S[components[j], 0, 0]
                    C_prev_molds = [C[components[j - x], 0, -1] for x in range(1, self.X_alpha[mold_type] + 1)]
                    if S_wait < min(C_prev_molds):
                        return False
        return True


# 染色体类
class Chromosome:
    def __init__(self, n, f, molds):
        self.n = n
        self.f = f
        self.molds = molds

    def initialize_chromosomes(self):
        priority_chromosome = np.random.permutation(np.arange(1, self.n + 1))
        line_chromosome = np.random.randint(1, self.f + 1, self.n)
        return priority_chromosome, line_chromosome

    # 根据优先级和流水线分配构件生产顺序
    def decode_schedule(self, priority_chromosome, line_chromosome):
        line_assignments = {i: [] for i in range(1, self.f + 1)}
        for component_id in range(1, self.n + 1):
            assigned_line = line_chromosome[component_id - 1]
            line_assignments[assigned_line].append((priority_chromosome[component_id - 1], component_id))

        production_schedule = {}
        for line, components in line_assignments.items():
            components_sorted = sorted(components, key=lambda x: x[0])
            production_schedule[line] = [comp[1] for comp in components_sorted]

        return production_schedule

    # 根据优先级和模具分配构件生产顺序
    def get_mold_priority_order(self, priority_chromosome, molds):
        mold_priority_order = {}
        for mold_type, components in molds.items():
            sorted_components = sorted(components, key=lambda x: priority_chromosome[x - 1])
            mold_priority_order[mold_type] = sorted_components
        return mold_priority_order


# NSGA-II类实现多目标遗传算法
class NSGAII:
    def __init__(self, population_size, max_generations, crossover_rate, mutation_rate, n, f, k, model):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n = n
        self.f = f
        self.k = k
        self.model = model
        self.population = [self.initialize_chromosomes() for _ in range(population_size)]
        self.best_schedules = []  # 新增列表用于记录每代的最佳调度方案
        self.best_stacks = []  # 新增列表用于记录每代的最佳栈堆划分

    def initialize_chromosomes(self):
        priority_chromosome = np.random.permutation(np.arange(1, self.n + 1))
        line_chromosome = np.random.randint(1, self.f + 1, self.n)
        return priority_chromosome, line_chromosome

    def decode_schedule(self, priority_chromosome, line_chromosome):
        line_assignments = {i: [] for i in range(1, self.f + 1)}
        for component_id in range(1, self.n + 1):
            assigned_line = line_chromosome[component_id - 1]
            line_assignments[assigned_line].append((priority_chromosome[component_id - 1], component_id))

        production_schedule = {}
        for line, components in line_assignments.items():
            components_sorted = sorted(components, key=lambda x: x[0])
            production_schedule[line] = [comp[1] for comp in components_sorted]

        return production_schedule

    def calculate_start_end_times(self, S, production_schedule, P):
        C = np.zeros_like(S)
        for line, components in production_schedule.items():
            line_idx = line - 1
            for idx, component in enumerate(components):
                component_idx = component - 1
                for k_idx in range(self.k):
                    if k_idx == 0:
                        if idx == 0:
                            S[line_idx, component_idx, k_idx] = 0
                            C[line_idx, component_idx, k_idx] = S[line_idx, component_idx, k_idx] + P[
                                component_idx, k_idx]
                        else:
                            prev_component_idx = components[idx - 1] - 1
                            S[line_idx, component_idx, k_idx] = C[line_idx, prev_component_idx, -1]
                            C[line_idx, component_idx, k_idx] = S[line_idx, component_idx, k_idx] + P[
                                component_idx, k_idx]
                    else:
                        S[line_idx, component_idx, k_idx] = C[line_idx, component_idx, k_idx - 1]
                        C[line_idx, component_idx, k_idx] = S[line_idx, component_idx, k_idx] + P[
                            component_idx, k_idx]
        return S, C

    def fitness_function(self, priority_chromosome, line_chromosome):
        production_schedule = self.decode_schedule(priority_chromosome, line_chromosome)
        S = np.zeros((self.f, self.n, self.k))
        S, C = self.calculate_start_end_times(S, production_schedule, self.model.P)
        stacks = self.model.simulate_stacking(C)  # 使用模拟入库过程
        makespan = self.model.calculate_makespan(C)
        inventory = self.model.calculate_inventory(C, production_schedule)  # 传递production_schedule参数
        movecount = self.model.calculate_movecount(stacks, self.model.D)  # 使用计算出库移动次数的方法
        return makespan, inventory, movecount

    # 非支配排序
    def non_dominated_sort(self, population):
        fronts = [[]]
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        ranks = [0] * len(population)

        for p in range(len(population)):
            for q in range(len(population)):
                if self.dominates(population[p][1], population[q][1]):
                    dominated_solutions[p].append(q)
                elif self.dominates(population[q][1], population[p][1]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        fronts.pop(-1)
        return fronts

    # 支配关系
    def dominates(self, obj1, obj2):
        return all(x <= y for x, y in zip(obj1, obj2)) and any(x < y for x, y in zip(obj1, obj2))

    # 拥挤度计算
    def calculate_crowding_distance(self, front, objectives):
        distances = [0] * len(front)
        for i in range(len(objectives[0])):  # 确保按目标值的数量进行迭代
            front_objectives = [objectives[idx][i] for idx in front]  # 获取当前目标i的所有值
            min_val = min(front_objectives)
            max_val = max(front_objectives)
            if max_val == min_val:
                continue
            for j in range(len(front)):
                distances[j] += (front_objectives[j] - min_val) / (max_val - min_val)
        return distances

    # 选择操作
    def selection(self, population, objectives):
        fronts = self.non_dominated_sort(population)
        new_population = []

        # 遍历所有前沿，从第一前沿开始选择个体
        for front in fronts:
            # 如果新种群的大小加上当前前沿的大小超过了种群大小，则根据拥挤度选择部分个体
            if len(new_population) + len(front) > self.population_size:
                distances = self.calculate_crowding_distance(front, objectives)
                # 注意：这里我们不需要对距离取反，因为拥挤度计算已经隐含地处理了最小化目标的情况
                # 我们只需要选择拥挤度较高的个体（即在同一前沿内分布更均匀的个体）
                front = [population[i] for _, i in sorted(zip(distances, front), reverse=True)]
                # 只选择足够数量的个体以填满种群
                new_population.extend(front[:self.population_size - len(new_population)])
                break
            else:
                # 如果当前前沿可以完全放入新种群中，则直接添加所有个体
                new_population.extend(population[i] for i in front)

        # 如果新种群的大小仍然小于种群大小，则随机填充剩余个体（这通常不会发生，除非有约束导致某些个体被排除）
        if len(new_population) < self.population_size:
            random_indices = random.sample(range(len(population)), self.population_size - len(new_population))
            new_population.extend(population[i] for i in random_indices)

        return new_population
    # 交叉操作
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.n - 1)
            child1 = (
                np.concatenate([parent1[0][:crossover_point], parent2[0][crossover_point:]]),
                np.concatenate([parent1[1][:crossover_point], parent2[1][crossover_point:]]),
            )
            child2 = (
                np.concatenate([parent2[0][:crossover_point], parent1[0][crossover_point:]]),
                np.concatenate([parent2[1][:crossover_point], parent1[1][crossover_point:]]),
            )
            return child1, child2
        else:
            return parent1, parent2

    # 变异操作
    def mutate(self, chromosome):
        priority_chromosome, line_chromosome = chromosome
        for i in range(self.n):
            if random.random() < self.mutation_rate:
                swap_idx = random.randint(0, self.n - 1)
                priority_chromosome[i], priority_chromosome[swap_idx] = priority_chromosome[swap_idx], \
                priority_chromosome[i]
            if random.random() < self.mutation_rate:
                line_chromosome[i] = random.randint(1, self.f)
        return priority_chromosome, line_chromosome

    def run(self):
        best_objectives_per_generation = []  # 新增列表用于记录每一代的最佳目标函数值

        for generation in range(self.max_generations):
            # 计算适应度
            objectives = [self.fitness_function(ind[0], ind[1]) for ind in self.population]
            combined_population = self.population + self.population
            combined_objectives = objectives + objectives

            # 非支配排序和选择
            self.population = self.selection(combined_population, combined_objectives)

            # 找到当前代的最佳目标函数值及其对应的个体
            best_objective = min(objectives, key=lambda x: (x[0], x[1], x[2]))
            best_index = objectives.index(best_objective)
            best_individual = self.population[best_index]

            # 记录当前代的最佳目标函数值
            best_objectives_per_generation.append(best_objective)

            # 记录最佳调度方案和栈堆划分（仅记录最后一代）
            if generation == self.max_generations - 1:
                best_schedule = self.decode_schedule(best_individual[0], best_individual[1])
                best_stack = self.model.simulate_stacking(
                    self.calculate_start_end_times(np.zeros((self.f, self.n, self.k)), best_schedule, self.model.P)[1])
                self.best_schedules = [best_schedule]  # 仅存储最后一代的最佳调度方案
                self.best_stacks = [best_stack]  # 仅存储最后一代的最佳栈堆划分
                best_objectives_per_generation.append(best_objective)

            # 生成下一代
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(self.population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))

            self.population = new_population[:self.population_size]

        # 返回每一代的最佳目标函数值（实际上只返回了最后一代的）
        return best_objectives_per_generation

    # 输出调度方案和栈堆划分
    def output_best_schedules_and_stacks(self):
        if self.best_schedules and self.best_stacks:
            print("Best Schedule and Stacks:")
            for line, components in self.best_schedules[0].items():
                print(f"Line {line}: {components}")
            print("Stacks:")
            for idx, s in enumerate(self.best_stacks[0], start=1):
                print(f"Stack {idx}: {s}")
            print()

# 初始化和运行代码
n, f, k, d = 10, 2, 6, 2
H_W, H_N, H_A = 10, 14, 4
molds = {1: [1, 3, 4, 6, 9], 2: [2, 8], 3: [5, 7, 10]}
X_alpha = {1: 3, 2: 2, 3: 2}
# 手动输入每个构件的6道工序的生产时间
P = np.array([
    [2.0, 1.6, 2.4, 12, 2.5, 1.0],   # 构件1的生产时间
    [3.4, 4.0, 4.0, 12, 2.4, 2.5],   # 构件2的生产时间
    [0.8, 1.0, 1.2, 12, 0.8, 1.7],   # 构件3的生产时间
    [0.6, 0.8, 1.0, 12, 0.6, 2.0],   # 构件4的生产时间
    [3.0, 3.6, 2.4, 12, 2.4, 3.0],   # 构件5的生产时间
    [3.0, 3.2, 3.0, 12, 3.0, 1.6],   # 构件6的生产时间
    [1.3, 0.9, 2.4, 12, 1.9, 1.8],   # 构件7的生产时间
    [1.7, 1.4, 1.1, 12, 0.9, 0.7],   # 构件8的生产时间
    [2.2, 1.8, 1.2, 12, 2.3, 0.7],   # 构件9的生产时间
    [1.6, 3.2, 2.3, 12, 2.1, 2.7]    # 构件10的生产时间
])

D = np.array([
    68,   # 构件1的交货时间
    64,   # 构件2的交货时间
    72,   # 构件3的交货时间
    68,   # 构件4的交货时间
    118,  # 构件5的交货时间
    68,   # 构件6的交货时间
    54,   # 构件7的交货时间
    92,   # 构件8的交货时间
    105,  # 构件9的交货时间
    68    # 构件10的交货时间
])


# 创建NSGA-II实例
model = ProductionModel(n, f, k, P, D, H_W, H_N, H_A, molds, X_alpha, d)
nsga2 = NSGAII(population_size=100, max_generations=50, crossover_rate=0.8, mutation_rate=0.2, n=n, f=f, k=k,
               model=model)

# 记录开始时间
start_time = time.time()

# 运行NSGA-II算法并记录结果
best_objectives_per_generation = nsga2.run()

# 输出调度方案和栈堆划分
nsga2.output_best_schedules_and_stacks()

# 记录结束时间
end_time = time.time()
execution_time = end_time - start_time

# 打印算法运行时间
print(f"Algorithm Execution Time: {execution_time:.2f} seconds")

# 绘制迭代图
generations = range(1, len(best_objectives_per_generation) + 1)
best_makespans = [obj[0] for obj in best_objectives_per_generation]
best_inventories = [obj[1] for obj in best_objectives_per_generation]
best_movecounts = [obj[2] for obj in best_objectives_per_generation]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(generations, best_makespans, label='Max Makespan')
plt.xlabel('Generation')
plt.ylabel('Max Makespan')
plt.title('Best Makespan per Generation')


plt.subplot(1, 3, 2)
plt.plot(generations, best_inventories, label='Total Inventory Time')
plt.xlabel('Generation')
plt.ylabel('Total Inventory Time')
plt.title('Best Inventory per Generation')


plt.subplot(1, 3, 3)
plt.plot(generations, best_movecounts, label='Total Move Count')
plt.xlabel('Generation')
plt.ylabel('Total Move Count')
plt.title('Best Move Count per Generation')


plt.tight_layout()
plt.show()