import streamlit as st
import random
import operator
import math
import matplotlib.pyplot as plt

# ==========================
# DATA
# ==========================
jobs = ["J1", "J2", "J3", "J4", "J5"]

processing_time = {
    "J1": [1,12,19,11,15,14,17,1,12,13],
    "J2": [2,19,11,5,12,17,11,1,19,14],
    "J3": [16,16,6,19,7,2,11,15,14,5],
    "J4": [6,15,15,13,19,9,9,10,12,10],
    "J5": [3,13,10,15,9,9,10,17,6,14]
}

# ==========================
# JOB FEATURES
# ==========================
job_features = {}
for j in jobs:
    times = processing_time[j]
    job_features[j] = {
        "total": sum(times),
        "max": max(times),
        "min": min(times)
    }

# ==========================
# MAKESPAN
# ==========================
def makespan(sequence):
    machines = len(processing_time["J1"])
    machine_time = [0]*machines

    for job in sequence:
        for m in range(machines):
            machine_time[m] = max(
                machine_time[m],
                machine_time[m-1] if m > 0 else 0
            ) + processing_time[job][m]
    return machine_time[-1]

# ==========================
# GP FUNCTION SET
# ==========================
def protected_div(a, b):
    return a / b if abs(b) > 0.001 else a

FUNCTIONS = [
    ("+", operator.add),
    ("-", operator.sub),
    ("*", operator.mul),
    ("/", protected_div)
]

TERMINALS = ["total", "max", "min"]

# ==========================
# RANDOM EXPRESSION TREE
# ==========================
def random_tree(depth):
    if depth == 0 or random.random() < 0.3:
        if random.random() < 0.7:
            return random.choice(TERMINALS)
        else:
            return random.uniform(0.1, 10)
    func = random.choice(FUNCTIONS)
    return (func, random_tree(depth-1), random_tree(depth-1))

# ==========================
# TREE EVALUATION
# ==========================
def eval_tree(tree, features):
    if isinstance(tree, (int, float)):
        return tree
    if isinstance(tree, str):
        return features[tree]
    func, left, right = tree
    return func[1](eval_tree(left, features), eval_tree(right, features))

# ==========================
# FITNESS FUNCTION
# ==========================
def fitness(tree):
    priorities = {}
    for j in jobs:
        priorities[j] = eval_tree(tree, job_features[j])
    sequence = sorted(jobs, key=lambda x: priorities[x])
    return makespan(sequence)

# ==========================
# GP OPERATORS
# ==========================
def tournament(pop, k=3):
    return min(random.sample(pop, k), key=fitness)

def crossover(t1, t2):
    if random.random() < 0.5:
        return t1
    if isinstance(t1, tuple) and isinstance(t2, tuple):
        return (t1[0], crossover(t1[1], t2[1]), crossover(t1[2], t2[2]))
    return t2

def mutation(tree, depth):
    if random.random() < 0.1:
        return random_tree(depth)
    if isinstance(tree, tuple):
        return (tree[0], mutation(tree[1], depth-1), mutation(tree[2], depth-1))
    return tree

# ==========================
# GP MAIN
# ==========================
def genetic_programming(pop_size, generations, depth):
    population = [random_tree(depth) for _ in range(pop_size)]
    history = []

    best = min(population, key=fitness)

    for _ in range(generations):
        new_pop = []
        for _ in range(pop_size):
            p1 = tournament(population)
            p2 = tournament(population)
            child = crossover(p1, p2)
            child = mutation(child, depth)
            new_pop.append(child)

        population = new_pop
        current_best = min(population, key=fitness)
        if fitness(current_best) < fitness(best):
            best = current_best

        history.append(fitness(best))

    return best, fitness(best), history

# ==========================
# STREAMLIT UI
# ==========================
st.title("Genetic Programming for Job Scheduling")

pop = st.sidebar.slider("Population Size", 10, 100, 30)
gen = st.sidebar.slider("Generations", 10, 200, 80)
depth = st.sidebar.slider("Tree Depth", 2, 6, 4)

if st.button("Run GP"):
    best_tree, best_fit, history = genetic_programming(pop, gen, depth)

    st.subheader("Best Evolved Program (Tree)")
    st.write(best_tree)

    st.subheader("Best Makespan")
    st.success(best_fit)

    st.subheader("Convergence Curve")
    plt.figure()
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best Makespan")
    st.pyplot(plt)
