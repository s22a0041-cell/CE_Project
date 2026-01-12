import streamlit as st
import random
import numpy as np
import pandas as pd

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
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2)

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

        if isinstance(value, complex):
            return 1e9

        if not np.isfinite(value):
            return 1e9

        return float(value)

    except:
        return 1e9

# -------------------------------------------------
# JOB SHOP SIMULATION
# -------------------------------------------------
def simulate(rule_expr):
    machine_time = [0] * NUM_MACHINES
    job_time = [0] * NUM_JOBS
    completed = [False] * NUM_JOBS
    remaining_ops = [NUM_MACHINES] * NUM_JOBS
    completion_times = []

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
            finish = start + processing[job][m]
            machine_time[m] = finish
            job_time[job] = finish
            remaining_ops[job] -= 1

        completed[job] = True
        completion_times.append(job_time[job])

    makespan = max(completion_times)
    idle_time = sum(machine_time) - makespan

    return makespan + 0.1 * idle_time

# -------------------------------------------------
# GENETIC OPERATORS
# -------------------------------------------------
def crossover(p1, p2):
    c1 = random.randint(1, len(p1) - 2)
    c2 = random.randint(1, len(p2) - 2)
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

    st.subheader("Fitness Convergence")
    st.line_chart(
        pd.DataFrame({"Best Fitness": best_history}),
        use_container_width=True
    )

    st.subheader("Best Evolved Priority Rule")
    st.code(best_rule, language="text")

    st.subheader("Final Fitness Value (Makespan-based)")
    st.write(best_history[-1])
