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
st.write("Genetic Programming to evolve scheduling rules on limited machines")

# -------------------------------------------------
# SIDEBAR PARAMETERS
# -------------------------------------------------
st.sidebar.header("Simulation Parameters")

NUM_JOBS = st.sidebar.slider("Number of Jobs", 3, 10, 5)
NUM_MACHINES = st.sidebar.slider("Number of Machines", 1, 3, 2)
POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 50)
GENERATIONS = st.sidebar.slider("Generations", 10, 100, 30)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2)

random.seed(42)

# -------------------------------------------------
# JOB DATA
# -------------------------------------------------
jobs = []
for i in range(NUM_JOBS):
    jobs.append({
        "job": f"J{i+1}",
        "processing": random.randint(1, 10),
        "release": random.randint(0, 5)
    })

jobs_df = pd.DataFrame(jobs)

st.subheader("Job Data")
st.dataframe(jobs_df, use_container_width=True)

# -------------------------------------------------
# GP REPRESENTATION (RULE TREE AS EXPRESSION STRING)
# -------------------------------------------------
OPERATORS = ["+", "-", "*"]
TERMINALS = ["p", "w", "1", "2"]

def random_expression(depth=2):
    if depth == 0:
        return random.choice(TERMINALS)
    return f"({random_expression(depth-1)} {random.choice(OPERATORS)} {random_expression(depth-1)})"

def evaluate_expression(expr, p, w):
    try:
        return eval(expr, {"p": p, "w": w})
    except:
        return float("inf")

# -------------------------------------------------
# SCHEDULING SIMULATION
# -------------------------------------------------
def simulate(rule_expr):
    time = 0
    machines = [0] * NUM_MACHINES
    completion_times = []
    job_queue = jobs.copy()

    while job_queue:
        available = [j for j in job_queue if j["release"] <= time]

        if not available:
            time += 1
            continue

        scores = []
        for j in available:
            wait = time - j["release"]
            score = evaluate_expression(rule_expr, j["processing"], wait)
            scores.append((score, j))

        scores.sort(key=lambda x: x[0])
        selected = scores[0][1]

        m = machines.index(min(machines))
        start = max(time, machines[m])
        finish = start + selected["processing"]

        machines[m] = finish
        completion_times.append(finish)
        job_queue.remove(selected)
        time += 1

    makespan = max(completion_times)
    total_completion = sum(completion_times)
    idle_time = sum(machines) - makespan

    fitness = makespan + 0.1 * total_completion + 0.1 * idle_time
    return fitness

# -------------------------------------------------
# GENETIC OPERATORS
# -------------------------------------------------
def crossover(p1, p2):
    cut1 = random.randint(1, len(p1)-2)
    cut2 = random.randint(1, len(p2)-2)
    return p1[:cut1] + p2[cut2:]

def mutate(expr):
    if random.random() < MUTATION_RATE:
        return random_expression(2)
    return expr

def tournament_selection(pop, fitness, k=3):
    selected = random.sample(list(zip(pop, fitness)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]

# -------------------------------------------------
# RUN GP
# -------------------------------------------------
if st.button("Run Genetic Programming"):
    population = [random_expression(2) for _ in range(POP_SIZE)]
    best_fitness = []
    best_rule = None

    for gen in range(GENERATIONS):
        fitness_values = [simulate(ind) for ind in population]
        best_idx = np.argmin(fitness_values)

        best_fitness.append(fitness_values[best_idx])
        best_rule = population[best_idx]

        new_population = []
        for _ in range(POP_SIZE):
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    st.subheader("Fitness Convergence")
    st.line_chart(
        pd.DataFrame({"Fitness": best_fitness}),
        use_container_width=True
    )

    st.subheader("Best Scheduling Rule (Evolved)")
    st.code(best_rule, language="text")

    st.subheader("Best Fitness Value")
    st.write(best_fitness[-1])
