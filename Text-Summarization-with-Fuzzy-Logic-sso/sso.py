from utils import sentences_from_document, generate_vocab, get_TF, fitness, fitness_gradient, clean
import random
import numpy as np


#Get reference summary
ref_file = open('./5.reference/001.txt')
ref = ref_file.read()
ref_file.close()

#Clean the Reference Summary
ref = clean(ref)

ref_T, ref_label, ref_sents = sentences_from_document(ref)
ref_T = ref_T[2:]

reference = []
reference.append(ref_T)
for i in range(0, len(ref_sents)):
    reference.append(ref_sents[i])


#Pre-Processing - Get the sentence scores
feature_file = open('./2.feature/001.txt')
feature = feature_file.read()

feature_matrix = []

for i, line in enumerate(feature.splitlines()):
    ele = ' '.join(line.split())
    feature_matrix.append([float(i) for i in (ele.split(' ')[1:])])

feature_file.close()


doc_file = open('./1.clean/001.txt')
doc = doc_file.read()
doc_file.close()

doc_T, doc_label, doc_sents = sentences_from_document(doc)
doc_T = doc_T[2:]

document = {}
for i in range(0, len(doc_sents)):
    document[i+1] = doc_sents[i]

vocab = generate_vocab(doc_sents)

ref_tf = get_TF(reference, vocab)


#STEP 1 - Initialize Set of Solutions A as weight of text features,
#Set of Velocities V, stage counter k = 1 and set the different
#parameters of SSO
random.seed()

best_solution = []

population_size = 100
no_of_features = 8
iterations = 100
sentences_to_extract = 4
velocity_multiplier = 0.75
inertia = 0.56
velocity_limiter_ratio = 0.37
time_delta = 1
no_of_rotational_positions = 10

weight_matrix = []
velocity_matrix = [[0.1 for i in range(0, no_of_features)] for j in range(0, population_size)]

for i in range(0, population_size):
    weights = []
    for j in range(0, no_of_features):
        weights.append(random.random())
    weight_matrix.append(weights)

def score_single_solution(a):
    single_solution_score = {}

    for i in range(0, len(feature_matrix)):
        sentence_score = 0.0
        for j in range(0, no_of_features):
            sentence_score += (a[j] * feature_matrix[i][j])
        single_solution_score[i+1] = sentence_score

    return single_solution_score

def extract_best_sentences_from_single_solution(sc):
    single_solution_best_sentences = []
    best_values = sorted(list(sc.values()))
    best_values.reverse()
    best_values = best_values[0:sentences_to_extract]

    for val in best_values:
        single_solution_best_sentences.append(list(sc.keys())[list(sc.values()).index(val)])

    return sorted(single_solution_best_sentences)

def generate_summary_for_single_solution(best_sents):
    single_summary = []
    single_summary.append(doc_T)
    for i in best_sents:
        single_summary.append(document[i])

    return single_summary

def get_fitness_for_single_solution(summ):
    sys_tf = get_TF(summ, vocab)
    #ref_tf is globally defined

    fit = fitness(sys_tf, ref_tf)

    return fit

def normalize_weight_matrix(weights):
    temp_weights = []
    for j in range(0, len(weights)):
        temp_weights.append(weights[j] / max(weights))

    return temp_weights


for k in range(0, iterations):
    # STEP 2 - Score the sentences for each solution with their text feature scores
    scores = []
    for iter in range(0, population_size):
        scores.append(score_single_solution(weight_matrix[iter]))

    # STEP 3 - Extract high scored l number of sentences from document for each solution
    best_sentences = []
    for iter in range(0, population_size):
        best_sentences.append(extract_best_sentences_from_single_solution(scores[iter]))

    # STEP 4 - Calculate the fitness value for the sentences extracted for each solution
    summaries = []
    fitness_gradients = []
    fitness_values = []
    for iter in range(0, population_size):
        summaries.append(generate_summary_for_single_solution(best_sentences[iter]))

    for iter in range(0, population_size):
        fitness_values.append(get_fitness_for_single_solution(summaries[iter]))


    def get_fitness_gradient_for_single_solution(summ):
        sys_tf = get_TF(summ, vocab)
        # ref_tf is globally defined

        fit_grad = fitness_gradient(sys_tf, ref_tf)

        return fit_grad


    for iter in range(0, population_size):
        fitness_gradients.append(get_fitness_gradient_for_single_solution(summaries[iter]))

    # STEP 5 - Set the global best solution among all according to highest fitness value
    best_solution = weight_matrix[fitness_values.index(max(fitness_values))]
    best_solution = normalize_weight_matrix(best_solution)

    print("Iteration: " + str(k) + ":  " + str(best_solution))

    # STEP 6 - Update the velocity vector
    for q in range(0, population_size):
        for x in range(0, no_of_features):
            R1 = 0.6
            R2 = 0.7

            max_attainable_velocity = velocity_limiter_ratio * velocity_matrix[q][x]
            inertial_velocity = inertia * R2 * velocity_matrix[q][x]
            changed_velocity = velocity_multiplier * R1 * fitness_gradients[q]

            updated_velocity = min((changed_velocity + inertial_velocity), max_attainable_velocity)
            velocity_matrix[q][x] = updated_velocity

    # STEP 7 - Find new weights according to forward and rotational movements
    new_positions = []
    forward_position = []
    rotational_position = []

    # Forward Movement
    for q in range(0, population_size):
        new_pos = (np.array(weight_matrix[q]) + (time_delta * np.array(velocity_matrix[q]))).tolist()
        forward_position.append(new_pos)

    # Rotational Movement
    # random number in range [-1, 1]
    R3 = (random.randint(-1000000, 1000000)) / 1000000
    for m in range(0, no_of_rotational_positions):
        rot_pos = []
        for q in range(0, population_size):
            new_pos = (np.array(forward_position[q]) + (R3 * np.array(forward_position[q]))).tolist()
            rot_pos.append(new_pos)
        rotational_position.append(rot_pos)

    new_positions.append(forward_position)
    new_positions.extend(rotational_position)

    # caluclate fitness values using new positions
    new_fitness = []
    new_best = []

    for i in range(0, len(new_positions)):
        new_fit = []
        for j in range(0, population_size):
            new_score = score_single_solution(new_positions[i][j])
            new_best_sentences = extract_best_sentences_from_single_solution(new_score)
            new_summary = generate_summary_for_single_solution(new_best_sentences)
            new_fit.append(get_fitness_for_single_solution(new_summary))

        new_fitness.append(new_fit)

    for i in range(0, population_size):
        temp = []
        for j in range(0, len(new_fitness)):
            temp.append(new_fitness[j][i])
        new_best.append(temp)

    matrix = []
    for i in range(0, population_size):
        matrix.append(new_positions[new_best[i].index(max(new_best[i]))][i])
    weight_matrix = matrix

#Save the results
file = open('./weights.txt', 'w')
for val in best_solution:
    file.write(str(val) + "\n")
file.close()









