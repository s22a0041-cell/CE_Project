import streamlit as st
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="GP Job Shop Scheduling",
    layout="wide"
)

st.title("Job Shop Scheduling using Genetic Programming")
st.write("CSV format: Job × Machine processing time matrix")

# -------------------------------------------------
# SIDEBAR PARAMETERS
# -------------------------------------------------
st.sidebar.header("Genetic Programming Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 50)
GENERATIONS = st.sidebar.slider("Generations", 10, 100, 30)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 0.5, 0.2)

OBJECTIVE_TYPE = st.sidebar.radio(
    "Objective Type",
    options=["Single-Objective", "Multi-Objective"]
)

if OBJECTIVE_TYPE == "Multi-Objective":
    st.sidebar.write("Set weights for multi-objective fitness (sum <= 1)")
    w_makespan = st.sidebar.slider("Weight: Makespan", 0.0, 1.0, 0.4)
    w_idle = st.sidebar.slider("Weight: Machine Idle Time", 0.0, 1.0, 0.3)
    w_wait = st.sidebar.slider("Weight: Job Waiting Time", 0.0, 1.0, 0.3)
else:
    w_makespan = 1.0
    w_idle = 0.1
    w_wait = 0.0  # hanya untuk single-objective

# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------
st.subheader("Upload Job Shop CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload your jobshop CSV file.")
    st.stop()

df = pd.read_csv(uploaded_file)
job_col = df.columns[0]
machine_cols = df.columns[1:]

if not all(col.startswith("M") for col in machine_cols):
    st.error("Machine columns must be named M1, M2, ..., Mn")
    st.stop()

if not np.all([pd.api.types.is_numeric_dtype(df[col]) for col in machine_cols]):
    st.error("Processing times must be numeric")
    st.stop()

st.subheader("Uploaded Dataset")
st.dataframe(df, use_container_width=True)

jobs = df[job_col].astype(str).tolist()
processing = df[machine_cols].to_numpy()

NUM_JOBS = len(jobs)
NUM_MACHINES = len(machine_cols)

# -------------------------------------------------
# GP REPRESENTATION
# -------------------------------------------------
OPERATORS = ["+", "-", "*"]
TERMINALS = ["p", "r", "1", "2"]

def random_expression(depth=2):
    if depth == 0:
        return random.choice(TERMINALS)
    return f"({random_expression(depth-1)} {random.choice(OPERATORS)} {random_expression(depth-1)})"

def eval_rule(expr, p, r):
    try:
        value = eval(expr, {"p": p, "r": r})
        if isinstance(value, complex) or not np.isfinite(value):
            return 1e9
        return float(value)
    except:
        return 1e9

# -------------------------------------------------
# JOB SHOP SIMULATION (tambah job waiting time)
# -------------------------------------------------
def simulate(rule_expr):
    machine_time = [0] * NUM_MACHINES
    job_time = [0] * NUM_JOBS
    completed = [False] * NUM_JOBS
    remaining_ops = [NUM_MACHINES] * NUM_JOBS
    completion_times = []
    waiting_time = [0] * NUM_JOBS

    while not all(completed):
        available_jobs = [j for j in range(NUM_JOBS) if not completed[j]]
        scores = []
        for j in available_jobs:
            total_p = processing[j].sum()
            score = eval_rule(rule_expr, total_p, remaining_ops[j])
            scores.append((score, j))

        scores = [(s, j) for s, j in scores if np.isfinite(s)]
        scores.sort(key=lambda x: x[0])
        job = scores[0][1]

        for m in range(NUM_MACHINES):
            start = max(machine_time[m], job_time[job])
            waiting_time[job] += start - job_time[job]  # job menunggu di mesin
            finish = start + processing[job][m]
            machine_time[m] = finish
            job_time[job] = finish
            remaining_ops[job] -= 1

        completed[job] = True
        completion_times.append(job_time[job])

    makespan = max(completion_times)
    idle_time = sum(machine_time) - makespan
    total_waiting_time = sum(waiting_time)

    # fitness weighted sum untuk multi-objective
    fitness_value = w_makespan * makespan + w_idle * idle_time + w_wait * total_waiting_time
    return fitness_value

# -------------------------------------------------
# GENETIC OPERATORS
# -------------------------------------------------
def crossover(p1, p2):
    c1 = random.randint(1, len(p1)-2)
    c2 = random.randint(1, len(p2)-2)
    return p1[:c1] + p2[c2:]

def mutate(expr):
    if random.random() < MUTATION_RATE:
        return random_expression(2)
    return expr

def tournament(pop, fitness, k=3):
    chosen = random.sample(list(zip(pop, fitness)), k)
    chosen.sort(key=lambda x: x[1])
    return chosen[0][0]

# -------------------------------------------------
# FUNCTION TO VISUALIZE TREE
# -------------------------------------------------
def expr_to_tree(expr):
    count = 0
    g = Digraph(format='png')
    def add_node(e):
        nonlocal count
        node_id = f"n{count}"
        count += 1

        e = e.strip()
        if e.startswith("(") and e.endswith(")"):
            level = 0
            for i, c in enumerate(e[1:-1], start=1):
                if c == "(":
                    level += 1
                elif c == ")":
                    level -= 1
                elif c in "+-*":
                    if level == 0:
                        op = c
                        left = e[1:i]
                        right = e[i+1:-1]
                        g.node(node_id, op)
                        left_id = add_node(left)
                        right_id = add_node(right)
                        g.edge(node_id, left_id)
                        g.edge(node_id, right_id)
                        return node_id
        g.node(node_id, e)
        return node_id
    add_node(expr)
    return g

# -------------------------------------------------
# RUN GP
# -------------------------------------------------
if st.button("Run Genetic Programming"):
    population = [random_expression(2) for _ in range(POP_SIZE)]
    best_history = []
    best_rule = None

    for g in range(GENERATIONS):
        fitness = [simulate(ind) for ind in population]
        best_idx = int(np.argmin(fitness))
        best_history.append(fitness[best_idx])
        best_rule = population[best_idx]

        new_population = []
        for _ in range(POP_SIZE):
            p1 = tournament(population, fitness)
            p2 = tournament(population, fitness)
            if random.random() < CROSSOVER_RATE:
                child = crossover(p1, p2)
            else:
                child = p1
            child = mutate(child)
            new_population.append(child)
        population = new_population

    # -------------------------------------------------
    # PLOT FITNESS CONVERGENCE
    # -------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(range(1, GENERATIONS+1), best_history, marker='o')
    ax.set_title(f"Fitness Convergence ({OBJECTIVE_TYPE})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.grid(True)
    st.pyplot(fig)

    # -------------------------------------------------
    # DISPLAY BEST RULE
    # -------------------------------------------------
    st.subheader("Best Evolved Priority Rule")
    st.code(best_rule, language="text")

    # -------------------------------------------------
    # DISPLAY TREE VISUALIZATION
    # -------------------------------------------------
    st.subheader("Best Priority Rule Tree")
    tree_graph = expr_to_tree(best_rule)
    st.graphviz_chart(tree_graph)

    # -------------------------------------------------
    # DISPLAY FINAL FITNESS
    # -------------------------------------------------
    st.subheader("Final Fitness Value (Weighted Multi-Objective)")
    st.write(best_history[-1])
