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
st.write("Genetic Programming for Job × Machine Scheduling using CSV upload")

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
    st.info("Please upload a Job × Machine CSV file.")
    st.stop()

df = pd.read_csv(uploaded_file)

if "Job" not in df.columns:
    st.error("CSV must contain 'Job' as the first column")
    st.stop()

machine_cols = df.columns[1:]

st.subheader("Uploaded Dataset")
st.dataframe(df, use_container_width=True)

jobs = df["Job"].tolist()
processing_matrix = df[machine_cols].values

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

    total_completion = []

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
        total_completion.append(job_time[job])

    makespan = max(total_completion)
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

        new_pop = []
        for _ in range(POP_SIZE):
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        population = new_pop

    st.subheader("Fitness Convergence")
    st.line_chart(
        pd.DataFrame({"Best Fitness": best_history}),
        use_container_width=True
    )

    st.subheader("Best Scheduling Rule")
    st.code(best_rule, language="text")

    st.subheader("Final Makespan-based Fitness")
    st.write(best_history[-1])
