import numpy as np
import matplotlib.pyplot as plt
import random
import time

class Chromosome:
    def __init__(self, n, f):
        self.n = n
        self.f = f
        self.priority_chromosome = np.zeros(n, dtype=int)
        self.line_chromosome = np.zeros(n, dtype=int)

    def initialize_chromosomes(self):
        self.priority_chromosome = np.random.permutation(np.arange(1, self.n + 1))
        self.line_chromosome = np.random.randint(1, self.f + 1, self.n)


class ProductionScheduler:
    def __init__(self, n, f, k, P, D_order, H_W, H_N, H_A, molds, X_alpha, d, chromosome):
        self.n = n  # 构件数量
        self.f = f  # 流水线数量
        self.k = k  # 工序数量
        self.P = P  # 生产时间矩阵
        self.D_order = D_order  # 交货顺序
        self.H_W = H_W  # 正常工作时间
        self.H_N = H_N  # 非正常工作时间
        self.H_A = H_A  # 允许加班时间
        self.molds = molds  # 每种模具对应的构件
        self.X_alpha = X_alpha  # 每种模具的数量
        self.d = d  # 栈堆的最大深度
        self.chromosome = chromosome  # 个体的染色体

        # 解码染色体得到生产计划和模具类型计划
        self.production_schedule, self.global_mold_type_schedule = self.decode_chromosome()

        # 初始化开始时间、完成时间和等待时间矩阵
        self.S = np.zeros((f, n, k))  # 开始时间
        self.C = np.zeros((f, n, k))  # 完成时间
        self.W = np.zeros((f, n, k))  # 等待时间

        # 初始化每条流水线的当前时间
        self.current_time = np.zeros(f)

        # 初始化模具使用状态
        self.mold_available = {mold: [] for mold in self.global_mold_type_schedule.keys()}

    def decode_chromosome(self):
        # 初始化生产线分配字典
        line_assignments = {i: [] for i in range(1, self.f + 1)}

        # 遍历每个构件位置
        for position in range(self.n):
            component_id = position + 1
            assigned_line = self.chromosome.line_chromosome[position]
            priority = self.chromosome.priority_chromosome[position]

            # 获取模具类型（假设 molds 字典已正确初始化）
            mold_type = next((mt for mt, cids in self.molds.items() if component_id in cids), None)
            if mold_type is None:
                raise ValueError(f"Component ID {component_id} not found in molds dictionary")

            # 将构件信息添加到对应生产线的列表中
            line_assignments[assigned_line].append((priority, component_id, mold_type))

        # 初始化生产计划字典
        production_schedule = {i: [] for i in range(1, self.f + 1)}

        # 初始化全局模具类型字典
        global_mold_type_schedule = {mt: [] for mt in self.molds.keys()}

        # 遍历每条生产线，将构件按优先级添加到生产计划中，并同时填充全局模具类型字典
        for line, components in line_assignments.items():
            # 按优先级排序构件
            sorted_components = sorted(components, key=lambda x: x[0])

            # 填充生产计划
            production_schedule[line] = [comp[1] for comp in sorted_components]

            # 填充全局模具类型字典，并按优先级排序
            for priority, component_id, mold_type in sorted_components:
                global_mold_type_schedule[mold_type].append(component_id)

        # 注意：此时全局模具类型字典中的构件ID已经按优先级排序了

        # 返回生产计划和全局模具类型计划
        return production_schedule, global_mold_type_schedule


    def check_constraints(self, i, j, h, T):
        # 检查约束条件1和2
        if h > 0 and T < self.C[i, j, h - 1]:
            return False
        if j > 0 and T < self.C[i, j - 1, h]:
            return False

        # 检查约束条件3（第3道工序）
        if h == 3:
            D = int(T / 24)
            if T <= 24 * D + self.H_W + self.H_A:
                pass  # 满足条件
            elif T > 24 * D + self.H_W + self.H_A:
                return T >= 24 * (D + 1)
            else:
                return False

        # 检查约束条件4（第4道工序）
        if h == 4:
            D = int(T / 24)
            if T <= 24 * D + self.H_W:
                pass  # 满足条件
            elif 24 * D + self.H_W < T < 24 * (D + 1):
                return T >= 24 * (D + 1)
            elif T > 24 * (D + 1):
                pass  # 满足条件
            else:
                return False

        # 检查约束条件5（除第3、4道工序外的其他工序）
        if h not in [3, 4]:
            D = int(T / 24)
            if T <= 24 * D + self.H_W:
                pass  # 满足条件
            elif T > 24 * D + self.H_W:
                return T + self.H_N >= 24 * (D + 1)
            else:
                return False

        # 检查约束条件6（模具约束）
        component_id = self.production_schedule[i + 1][j]
        mold_type = next((mt for mt, cids in self.global_mold_type_schedule.items() if component_id in cids),
                         None)
        if mold_type is not None and mold_type in self.mold_available:
            last_release_time = max(self.mold_available[mold_type]) if self.mold_available[mold_type] else 0
            if T < last_release_time:
                # 模具不可用，增加等待时间
                return False

        # 所有约束条件都满足
        return True

    def schedule_production(self):
        """
        计算每个构件每道工序的开始时间和完成时间。
        """
        self.current_time = np.zeros(self.f)  # 每条流水线的当前时间初始化为0

        # 遍历每条生产线
        for i in range(self.f):
            for j, component_id in enumerate(self.production_schedule[i + 1]):
                for h in range(self.k):
                    # 计算满足所有约束条件的最早开始时间
                    T = self.estimate_earliest_start_time(i, j, h)

                    # 更新开始时间和完成时间
                    self.S[i, j, h] = T
                    self.C[i, j, h] = T + self.P[j, h]

                    # 更新模具的释放时间（仅在最后一个工序时更新）
                    if h == self.k - 1:
                        mold_type = next(
                            (mt for mt, cids in self.global_mold_type_schedule.items() if component_id in cids), None)
                        if mold_type is not None:
                            self.mold_available[mold_type].append(self.C[i, j, h])

        # 更新最大完工时间
        self.C_makespan = np.max(self.C[:, :, -1])

    def estimate_earliest_start_time(self, i, j, h):
        """
        估算满足所有约束条件的最早开始时间。
        """
        # 初始化为流水线的当前时间
        T = self.current_time[i]

        # 工序约束（前一工序的完成时间）
        if h > 0:
            T = max(T, self.C[i, j, h - 1])

        # 构件顺序约束（前一个构件同一工序的完成时间）
        if j > 0:
            T = max(T, self.C[i, j - 1, h])

        # 模具约束
        component_id = self.production_schedule[i + 1][j]
        mold_type = next((mt for mt, cids in self.global_mold_type_schedule.items() if component_id in cids), None)
        if mold_type is not None and self.mold_available[mold_type]:
            last_release_time = max(self.mold_available[mold_type])
            T = max(T, last_release_time)

        # 加班时间和工作时间约束
        T = self.enforce_time_constraints(T, h)

        return T

    def enforce_time_constraints(self, T, h):
        """
        根据工序时间限制（正常工作时间、加班时间等）调整开始时间。
        """
        D = int(T // 24)  # 当前天数
        if h == 3:  # 第3道工序的特殊约束
            if T <= 24 * D + self.H_W + self.H_A:
                return T
            else:
                return 24 * (D + 1)
        elif h == 4:  # 第4道工序的特殊约束
            if T <= 24 * D + self.H_W:
                return T
            elif T < 24 * (D + 1):
                return 24 * (D + 1)
            else:
                return T
        else:  # 其他工序的通用约束
            if T <= 24 * D + self.H_W:
                return T
            else:
                return max(T, 24 * (D + 1) - self.H_N)

    def simulate_stacking(self):
        """
        模拟堆叠完成后返回堆栈的构件组。
        """
        # 使用 self.C 的最后一维（工序完成时间）计算
        completion_times = self.C[:, :, -1].flatten()[:self.n]
        component_indices = np.arange(1, self.n + 1)
        all_components = sorted(zip(component_indices, completion_times), key=lambda x: x[1])

        stacks = []
        current_stack = []
        for idx, (j, t) in enumerate(all_components):
            current_stack.append(j)
            if len(current_stack) == self.d:
                stacks.append(current_stack)
                current_stack = []

        if current_stack:
            stacks.append(current_stack)

        return stacks

    def calculate_movecount(self, stacks, D_order):
        C_movecount = 0
        D_order_indices = {j: i for i, j in enumerate(self.D_order)}
        sorted_components = sorted(range(1, self.n + 1), key=lambda j: D_order_indices[j])

        for j in sorted_components:
            for stack_idx, stack in enumerate(stacks):
                if j in stack:
                    break

            N_d = len(stacks[stack_idx])
            position_in_stack = stacks[stack_idx].index(j)
            F_ijd = N_d - position_in_stack
            C_movecount += F_ijd - 1
            stacks[stack_idx].remove(j)

            for i, component in enumerate(stacks[stack_idx]):
                N_d = len(stacks[stack_idx])
                F_ijd = N_d - i
                C_movecount += F_ijd - 1

        return C_movecount

    def calculate_objectives(self):
        # 首先调用schedule_production方法来安排生产并计算最大完工时间
        self.schedule_production()
        # 从schedule_production方法中，我们已经得到了self.C_makespan作为最大完工时间
        max_makespan = self.C_makespan

        # 调用simulate_stacking方法来获取堆叠的构件组
        stacks = self.simulate_stacking()  # 不再传递 self.C

        # 最后，调用calculate_movecount方法来计算移动次数
        move_count = self.calculate_movecount(stacks, self.D_order)

        # 返回最大完工时间和移动次数作为目标函数值
        return max_makespan, move_count


class NSGAII:
    def __init__(self, population_size, max_generations, crossover_rate, mutation_rate, n, f, k, P, D_order, H_W, H_N,
                 H_A, molds, X_alpha, d):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n = n
        self.f = f
        self.k = k
        self.P = P
        self.D_order = D_order
        self.H_W = H_W
        self.H_N = H_N
        self.H_A = H_A
        self.molds = molds
        self.X_alpha = X_alpha
        self.d = d

        # 初始化最佳调度方案和栈堆划分列表
        self.best_schedules = []
        self.best_stacks = []

        # 初始化种群，并对每个个体进行解码和计算目标函数值
        self.population = []
        for _ in range(population_size):
            chromosome = Chromosome(n, f)
            chromosome.initialize_chromosomes()
            individual = ProductionScheduler(n, f, k, P, D_order, H_W, H_N, H_A, self.molds, X_alpha, d, chromosome)
            individual.calculate_objectives()  # 确保在创建个体后立即计算目标函数值
            self.population.append(individual)

    def evaluate_population(self):
        """
        计算当前种群中每个个体的目标函数值。
        返回一个目标值列表，形如 [(目标值1, 目标值2), ...]
        """
        objectives = []
        for individual in self.population:
            objectives.append(individual.calculate_objectives())
        return objectives


    def non_dominated_sort(self, population, objectives):
        fronts = [[]]
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        ranks = [0] * len(population)

        for p in range(len(population)):
            for q in range(len(population)):
                if self.dominates(objectives[p], objectives[q]):
                    dominated_solutions[p].append(q)
                elif self.dominates(objectives[q], objectives[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)

        return fronts, ranks

    def dominates(self, obj1, obj2):
        return all(x <= y for x, y in zip(obj1, obj2)) and any(x < y for x, y in zip(obj1, obj2))

    def calculate_crowding_distance(self, front, objectives):
        """
        计算种群中某个前沿（Pareto front）的拥挤距离。
        如果 front 为空，直接返回空列表。
        """
        if not front:
            return []

        distances = [0] * len(front)
        n_obj = len(objectives[0])

        for i in range(n_obj):
            front_objectives = np.array([objectives[idx][i] for idx in front])

            # 如果目标值全相等，则跳过该目标的拥挤距离计算
            if front_objectives[-1] == front_objectives[0]:
                continue

            sorted_indices = np.argsort(front_objectives)
            distances[sorted_indices[0]] = float('inf')  # 边界点的距离设为无穷大
            distances[sorted_indices[-1]] = float('inf')

            for j in range(1, len(front) - 1):
                distances[sorted_indices[j]] += (
                                                        front_objectives[sorted_indices[j + 1]] - front_objectives[
                                                    sorted_indices[j - 1]]
                                                ) / (front_objectives[-1] - front_objectives[0])

        return distances

    def selection_tournament(self, population, fronts, objectives):
        front1_idx, front2_idx = random.sample(range(len(fronts)), 2)
        front1, front2 = fronts[front1_idx], fronts[front2_idx]

        if not front1:
            idx1 = random.choice(range(len(population)))
        else:
            idx1 = random.choice(front1)

        if not front2:
            idx2 = random.choice(range(len(population)))
        else:
            idx2 = random.choice(front2)

        obj1, obj2 = objectives[idx1], objectives[idx2]
        if self.dominates(obj1, obj2) or (self.crowding_distances[front1_idx][idx1] > self.crowding_distances[front2_idx][idx2] if front1_idx == front2_idx else True):
            return population[idx1]
        else:
            return population[idx2]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            start_point = random.randint(1, self.n - 2)  # 确保起止点之间至少有一个基因
            end_point = random.randint(start_point + 1, self.n)

            # 获取父代染色体
            parent1_priority = parent1.chromosome.priority_chromosome
            parent2_priority = parent2.chromosome.priority_chromosome
            parent1_line = parent1.chromosome.line_chromosome
            parent2_line = parent2.chromosome.line_chromosome

            # 交换优先级染色体在交叉区域内的基因
            child1_priority = np.copy(parent1_priority)
            child2_priority = np.copy(parent2_priority)
            child1_priority[start_point:end_point], child2_priority[start_point:end_point] = \
                child2_priority[start_point:end_point], child1_priority[start_point:end_point]

            # 保持生产线染色体不变
            child1_line = np.copy(parent1_line)
            child2_line = np.copy(parent2_line)

            # 修复重复基因问题
            child1_priority = self.repair_chromosome_with_mapping(child1_priority, parent1_priority, parent2_priority,
                                                                  start_point, end_point)
            child2_priority = self.repair_chromosome_with_mapping(child2_priority, parent2_priority, parent1_priority,
                                                                  start_point, end_point)

            # 创建子代个体
            child1 = ProductionScheduler(self.n, self.f, self.k, self.P, self.D_order, self.H_W, self.H_N, self.H_A,
                                         self.molds, self.X_alpha, self.d, Chromosome(self.n, self.f))
            child1.chromosome.priority_chromosome = child1_priority
            child1.chromosome.line_chromosome = child1_line

            child2 = ProductionScheduler(self.n, self.f, self.k, self.P, self.D_order, self.H_W, self.H_N, self.H_A,
                                         self.molds, self.X_alpha, self.d, Chromosome(self.n, self.f))
            child2.chromosome.priority_chromosome = child2_priority
            child2.chromosome.line_chromosome = child2_line

            return child1, child2
        else:
            return parent1, parent2



    def repair_chromosome_with_mapping(self, chromosome, parent1, parent2, start_point, end_point):
        # 建立映射关系
        mapping = {parent2[i]: parent1[i] for i in range(start_point, end_point)}

        # 初始化一个集合来跟踪已使用的基因
        used_genes = set(chromosome)

        # 初始化结果染色体
        repaired_chromosome = np.copy(chromosome)

        # 遍历染色体中的每个基因，检查是否需要根据映射关系进行替换
        for i in range(len(chromosome)):
            if chromosome[i] in mapping:
                # 如果当前基因在映射中，获取其映射后的基因
                mapped_gene = mapping[chromosome[i]]

                # 如果映射后的基因已在结果染色体中使用，则需要替换
                if mapped_gene in used_genes:
                    # 找到一个未使用的基因进行替换
                    for j in range(1, self.n + 1):
                        if j not in used_genes:
                            used_genes.add(j)
                            repaired_chromosome[i] = j
                            break
            else:
                # 如果当前基因不在映射中，直接添加到已使用基因的集合中
                used_genes.add(chromosome[i])

        # 返回修复后的染色体
        return repaired_chromosome

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.n), 2)
            individual.chromosome.priority_chromosome[idx1], individual.chromosome.priority_chromosome[idx2] = \
                individual.chromosome.priority_chromosome[idx2], individual.chromosome.priority_chromosome[idx1]
        return individual

    def run(self):
        best_objectives_per_generation = []

        for generation in range(self.max_generations):
            objectives = self.evaluate_population()
            self.fronts, self.ranks = self.non_dominated_sort(self.population, objectives)
            self.crowding_distances = [self.calculate_crowding_distance(front, objectives) for front in self.fronts]

            best_objective = min(objectives, key=lambda x: (x[0], x[1]))
            best_objectives_per_generation.append(best_objective)

            offspring_population = []
            while len(offspring_population) < self.population_size:
                parent1 = self.selection_tournament(self.population, self.fronts, objectives)
                parent2 = self.selection_tournament(self.population, self.fronts, objectives)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring_population.extend([child1, child2])

            combined_population = self.population + offspring_population
            combined_objectives = objectives + [objectives[self.population.index(ind)] for ind in offspring_population]

            self.fronts, self.ranks = self.non_dominated_sort(combined_population, combined_objectives)
            self.crowding_distances = [self.calculate_crowding_distance(front, combined_objectives) for front in
                                       self.fronts]

            new_population = []
            i = 0
            while len(new_population) < self.population_size:
                front = self.fronts[i]
                to_select = min(len(front), self.population_size - len(new_population))
                sorted_front = sorted(front, key=lambda x: self.crowding_distances[self.ranks[x]][x], reverse=True)
                new_population.extend(combined_population[idx] for idx in sorted_front[:to_select])
                i += 1

            self.population = new_population

        # 记录最后一代的最佳调度方案和栈堆划分
        best_individual = self.population[objectives.index(best_objective)]
        best_schedule = best_individual.production_schedule
        best_stack = best_individual.simulate_stacking(best_individual.C)
        self.best_schedules.append(best_schedule)
        self.best_stacks.append(best_stack)

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
X_alpha = {1: 2, 2: 2, 3: 2}
molds = {1: [1, 3, 4, 6, 9], 2: [2, 8], 3: [5, 7, 10]}
# 手动输入每个构件的6道工序的生产时间
P = np.array([
    [2.0, 1.6, 2.4, 12, 2.5, 1.0],  # 构件1的生产时间
    [3.4, 4.0, 4.0, 12, 2.4, 2.5],  # 构件2的生产时间
    [0.8, 1.0, 1.2, 12, 0.8, 1.7],  # 构件3的生产时间
    [0.6, 0.8, 1.0, 12, 0.6, 2.0],  # 构件4的生产时间
    [3.0, 3.6, 2.4, 12, 2.4, 3.0],  # 构件5的生产时间
    [3.0, 3.2, 3.0, 12, 3.0, 1.6],  # 构件6的生产时间
    [1.3, 0.9, 2.4, 12, 1.9, 1.8],  # 构件7的生产时间
    [1.7, 1.4, 1.1, 12, 0.9, 0.7],  # 构件8的生产时间
    [2.2, 1.8, 1.2, 12, 2.3, 0.7],  # 构件9的生产时间
    [1.6, 3.2, 2.3, 12, 2.1, 2.7]   # 构件10的生产时间
])
# 构件的交货顺序
D_order = [1, 3, 4, 6, 9, 2, 8, 5, 7, 10]

# 创建NSGA-II实例
nsga2 = NSGAII(population_size=100, max_generations=50, crossover_rate=0.8, mutation_rate=0.01, n=n, f=f, k=k,
               P=P, D_order=D_order, H_W=H_W, H_N=H_N, H_A=H_A, molds = molds, X_alpha=X_alpha, d=d)

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

# 绘制多目标优化结果图
plt.figure(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(best_objectives_per_generation)))

for generation, (makespan, movecount) in enumerate(best_objectives_per_generation):
    plt.scatter(makespan, movecount, color=colors[generation], label=f'Generation {generation + 1}')

plt.xlabel('Max Makespan')
plt.ylabel('Total Move Count')
plt.title('Multi-Objective Optimization Results')
plt.legend(title='Generation')
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制迭代图
generations = range(1, len(best_objectives_per_generation) + 1)
best_makespans = [obj[0] for obj in best_objectives_per_generation]
best_movecounts = [obj[1] for obj in best_objectives_per_generation]

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(generations, best_makespans, label='Max Makespan')
plt.xlabel('Generation')
plt.ylabel('Max Makespan')
plt.title('Best Makespan per Generation')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(generations, best_movecounts, label='Total Move Count')
plt.xlabel('Generation')
plt.ylabel('Total Move Count')
plt.title('Best Move Count per Generation')
plt.legend()

plt.tight_layout()
plt.show()