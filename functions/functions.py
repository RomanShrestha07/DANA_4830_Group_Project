from random import randint
from statistics import median

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold, ShuffleSplit


def population_init(size, n_feat):
    """
    Initializes the population for the genetic algorithm.

    Parameters:
    size (int): Number of chromosomes in the population.
    n_feat (int): Number of features (length of each chromosome).

    Returns:
    list: A list of numpy arrays representing the population.
    """
    population = []

    for i in range(size):
        chromosome = np.ones(n_feat, dtype='bool')
        chromosome[:int(0.3 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)

    return population


def fitness_score(population, model, X, y):
    """
    Evaluates the fitness score of each chromosome in the population.

    Parameters:
    population (list): List of chromosomes.
    model (object): Machine learning model to evaluate.
    X (DataFrame): Features dataset.
    y (Series): Labels dataset.

    Returns:
    tuple: Sorted list of fitness scores and corresponding sorted population.
    """
    scores = []

    for chromosome in population:
        cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        accuracy_scores = cross_val_score(model, X.iloc[:, chromosome], y, scoring='accuracy', cv=cv)
        median_score = median(accuracy_scores)
        scores.append(median_score)

    scores, population = np.array(scores), np.array(population)
    indices = np.argsort(scores)

    return list(scores[indices][::-1]), list(population[indices, :][::-1])


def selection(pop_after_fit, selection_prop=0.5):
    """
    Selects the top-performing chromosomes from the population.

    Parameters:
    pop_after_fit (list): List of chromosomes after fitness evaluation.
    selection_prop (float): Proportion of the population to select.

    Returns:
    list: List of selected chromosomes.
    """
    n_select = int(len(pop_after_fit) * selection_prop)
    population_nextgen = pop_after_fit[:n_select]

    return population_nextgen


def crossover(pop_after_sel, size, crossover_prop=0.5):
    """
    Performs crossover between selected chromosomes to create new ones.

    Parameters:
    pop_after_sel (list): List of selected chromosomes.
    size (int): Desired size of the new population.
    crossover_prop (float): Proportion of the chromosome length to use for crossover.

    Returns:
    list: List of new chromosomes created by crossover.
    """
    pop_nextgen = []

    while len(pop_nextgen) < size - 2:
        parent1 = pop_after_sel[randint(0, len(pop_after_sel) - 1)]
        parent2 = pop_after_sel[randint(0, len(pop_after_sel) - 1)]

        split_index = int(len(parent1) * crossover_prop)

        child1 = np.concatenate((parent1[:split_index], parent2[split_index:]))
        child2 = np.concatenate((parent2[:split_index], parent1[split_index:]))

        pop_nextgen.append(child1)
        pop_nextgen.append(child2)

    return pop_nextgen[:size - 2]


def mutation(pop_after_cross, n_feat, mutation_rate=0.3):
    """
    Introduces mutations into the population.

    Parameters:
    pop_after_cross (list): List of chromosomes after crossover.
    n_feat (int): Number of features (length of each chromosome).
    mutation_rate (float): Proportion of the chromosome to mutate.

    Returns:
    list: List of mutated chromosomes.
    """
    mutation_range = int(mutation_rate * n_feat)
    pop_next_gen = []

    for n in range(0, len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_position = []

        for i in range(0, mutation_range):
            pos = randint(0, n_feat - 1)
            rand_position.append(pos)

        for j in rand_position:
            chromo[j] = not chromo[j]

        pop_next_gen.append(chromo)

    return pop_next_gen


def evaluate_model(model, X, y, k_folds):
    """
    Evaluates a machine learning model with accuracy, sensitivity and specificity with different k-folds.

    Parameters:
    model (object): Machine learning model to evaluate.
    X (DataFrame): Features dataset.
    y (Series): Labels dataset.
    k_folds (list): List of k-fold values for cross-validation.

    Returns:
    dict: Dictionary containing the evaluation metrics for each k-fold value.
    """
    result_dict = {'accuracy': [], 'sensitivity': [], 'specificity': []}

    for k in k_folds:
        if k == 0:
            model.fit(X, y)
            y_pred = model.predict(X)
        else:
            cv = KFold(n_splits=k, shuffle=True, random_state=42)
            y_pred = cross_val_predict(model, X, y, cv=cv)

        accuracy = accuracy_score(y, y_pred)
        sensitivity = recall_score(y, y_pred)

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp)

        result_dict['accuracy'].append(accuracy)
        result_dict['sensitivity'].append(sensitivity)
        result_dict['specificity'].append(specificity)

    return result_dict
