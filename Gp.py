import streamlit as st
import random
import numpy as np
import pandas as pd

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="GP Job Scheduling Optimization",
    layout="wide"
)

st.title("Job Scheduling using Genetic Programming")
st.write("Genetic Programming for Job × Machine Scheduling (CSV Flexible Format)")

# -------------------------------------------------
# SIDEBAR PARAMETERS
# -------------------------------------------------
st.sidebar.header("Simulation Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 50)
GENERATIONS = st.sidebar.slider("Generations", 10, 100, 30)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2)

# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------
st.subheader("Upload Job × Machine CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file.")
    st.stop()

df = pd.read_csv(uploaded_file)

if df.shape[1] < 2:
    st.error("CSV must contain at least one job column and one machine column")
    st.stop()

# Assume first column = Job ID, rest = machines
job_col = df.columns[0]
machine_cols = df.columns[1:]

# Validate machine columns are numeric
if not np.all([pd.api.types.is_numeric_dtype(df[col]) for col in machine_cols]):
    st.error("All machine columns must contain numeric processing times")
    st.stop()

st.subheader("Uploaded Dataset")
st.dataframe(df, use_container_width=True)

jobs = df[job_col].astype(str).tolist()
processing_matrix = df[machine_cols].to_numpy()

NUM_JOBS = len(jobs)
NUM_MACHINES = len(machine_cols)

# -------------------------------------------------
# GP REPRESENTATION
# p = total processing time
# r = remaining operations
# -------------------------------------------------
OPERATORS = ["+", "-", "*"]
TERMINALS = ["p", "r", "1", "2"]

def random_expression(depth=2):
    if depth == 0:
        return random.choice(TERMINALS)
    return f"({random_expression(depth-1)} {random.choice(OPERATORS)} {random_expression(depth-1)})"

def eval_expr(expr, p, r):
    try:
        return eval(expr, {"p": p, "r": r})
    except:
        return float("inf")

# -------------------------------------------------
# SCHEDULING SIMULATION (FLOW SHOP)
# -------------------------------------------------
def simulate(rule_expr):
    machine_time = [0] * NUM_MACHINES
    job_time = [0] * NUM_JOBS
    completed = [0] * NUM_JOBS
    remaining = [NUM_MACHINES] * NUM_JOBS

    completion_times = []

    while sum(completed) < NUM_JOBS:
        available_jobs = [j for j in range(NUM_JOBS) if completed[j] == 0]

        scores = []
        for j in available_jobs:
            total_p = sum(processing_matrix[j])
            score = eval_expr(rule_expr, total_p, remaining[j])
            scores.append((score, j))

        scores.sort(key=lambda x: x[0])
        job = scores[0][1]

        for m in range(NUM_MACHINES):
            start = max(machine_time[m], job_time[job])
            finish = start + processing_matrix[job][m]
            machine_time[m] = finish
            job_time[job] = finish
            remaining[job] -= 1

        completed[job] = 1
        completion_times.append(job_time[job])

    makespan = max(completion_times)
    idle_time = sum(machine_time) - makespan

    fitness = makespan + 0.1 * idle_time
    return fitness

# -------------------------------------------------
# GENETIC OPERATORS
# -------------------------------------------------
def crossover(p1, p2):
    cut1 = random.randint(1, len(p1) - 2)
    cut2 = random.randint(1, len(p2) - 2)
    return p1[:cut1] + p2[cut2:]

def mutate(expr):
    if random.random() < MUTATION_RATE:
        return random_expression(2)
    return expr

def tournament_selection(pop, fitness, k=3):
    candidates = random.sample(list(zip(pop, fitness)), k)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]

# -------------------------------------------------
# RUN GP
# -------------------------------------------------
if st.button("Run Genetic Programming"):
    population = [random_expression(2) for _ in range(POP_SIZE)]
    best_history = []
    best_rule = None

    for g in range(GENERATIONS):
        fitness_values = [simulate(ind) for ind in population]
        best_idx = int(np.argmin(fitness_values))

        best_history.append(fitness_values[best_idx])
        best_rule = population[best_idx]

        new_population = []
        for _ in range(POP_SIZE):
            p1 = tournament_selection(population, fitness_values)
            p2 = tournament_selection(population, fitness_values)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    st.subheader("Fitness Convergence")
    st.line_chart(
        pd.DataFrame({"Best Fitness": best_history}),
        use_container_width=True
    )

    st.subheader("Best Evolved Scheduling Rule")
    st.code(best_rule, language="text")

    st.subheader("Final Fitness Value")
    st.write(best_history[-1])
