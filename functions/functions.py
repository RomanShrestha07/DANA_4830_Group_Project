from random import randint
from statistics import median

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold, ShuffleSplit, StratifiedKFold, \
    train_test_split


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

    for _ in range(size):
        # Creating ones array with True values
        chromosome = np.ones(n_feat, dtype='bool')

        # Converting some of them to False values
        chromosome[:int(0.3 * n_feat)] = False

        # Randomly shuffling the true false array
        np.random.shuffle(chromosome)

        # Appending them to the population, this is done according to the size provided
        population.append(chromosome)

    # Return the array of chromosomes
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
        # ShuffleSplit for cross-validation with 10 splits and 30% of data as test set (mentioned in the article)
        cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

        # Evaluate the model using cross-validation on the features selected by the chromosome
        accuracy_scores = cross_val_score(model, X.iloc[:, chromosome], y, scoring='accuracy', cv=cv, n_jobs=-1)

        # Compute the median of the accuracy scores from cross-validation (mentioned in the article)
        median_score = median(accuracy_scores)

        # Append the median score to the scores list
        scores.append(median_score)

    # Convert scores and population lists to numpy arrays for sorting
    scores, population = np.array(scores), np.array(population)

    # Get the indices that would sort the scores in ascending order
    indices = np.argsort(scores)

    # Return the sorted scores and population in descending order of fitness scores
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
    # Calculate the number of chromosomes to select based on the proportion
    n_select = int(len(pop_after_fit) * selection_prop)

    # Select the top-performing chromosomes from the evaluated population
    population_nextgen = pop_after_fit[:n_select]

    # Return the selected chromosomes for the next generation
    return population_nextgen


def crossover(pop_after_sel, size, crossover_prop=0.5, n_elites=2):
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

    while len(pop_nextgen) < size - n_elites:
        # Randomly select two parents from the selected chromosomes
        parent1 = pop_after_sel[randint(0, len(pop_after_sel) - 1)]
        parent2 = pop_after_sel[randint(0, len(pop_after_sel) - 1)]

        # Determine the index at which to split the chromosomes for crossover
        split_index = int(len(parent1) * crossover_prop)

        # Create two children by combining parts of the parents' chromosomes
        child1 = np.concatenate((parent1[:split_index], parent2[split_index:]))
        child2 = np.concatenate((parent2[:split_index], parent1[split_index:]))

        # Append the new children to the next generation population list
        pop_nextgen.append(child1)
        pop_nextgen.append(child2)

    # Return the list of new chromosomes, trimming to the desired size to maintain elitism
    return pop_nextgen[:size - n_elites]


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
    # Calculate the number of positions to mutate in each chromosome
    mutation_range = int(mutation_rate * n_feat)
    pop_next_gen = []

    # Iterate over each chromosome in the population after crossover
    for n in range(0, len(pop_after_cross)):
        # Get the current chromosome
        chromo = pop_after_cross[n]
        # Initialize a list to store the positions that will be mutated
        rand_position = []

        # Randomly select positions in the chromosome to mutate
        for _ in range(0, mutation_range):
            pos = randint(0, n_feat - 1)
            rand_position.append(pos)

        # Mutate the selected positions in the chromosome
        for j in rand_position:
            chromo[j] = not chromo[j]

        # Add the mutated chromosome to the next generation population
        pop_next_gen.append(chromo)

    # Return the list of mutated chromosomes
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
    # Initialize a dictionary to store results for each metric
    result_dict = {'accuracy': [], 'sensitivity': [], 'specificity': []}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Iterate over each k-fold value in the k_folds list
    for k in k_folds:
        # If k is 0, use the entire dataset for training and testing
        if k == 0:
            # Fit the model on the entire dataset
            model.fit(X_train, y_train)

            # Predict the labels using the fitted model
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy of the model's predictions
            sensitivity = recall_score(y_test, y_pred)  # Calculate sensitivity (recall) of the model's predictions

            # Calculate the confusion matrix for evaluating specificity
            tn, fp, _, _ = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)  # Calculate specificity from the confusion matrix
        else:
            # Create a KFold cross-validator with k splits
            cv = StratifiedKFold(n_splits=k, random_state=42)

            # Perform cross-validation and obtain predictions for each fold
            y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

            accuracy = accuracy_score(y, y_pred)
            sensitivity = recall_score(y, y_pred)

            tn, fp, _, _ = confusion_matrix(y, y_pred).ravel()
            specificity = tn / (tn + fp)

        # Append the results for the current k-fold value to the result dictionary
        result_dict['accuracy'].append(accuracy)
        result_dict['sensitivity'].append(sensitivity)
        result_dict['specificity'].append(specificity)

    # Return the dictionary containing the evaluation metrics for each k-fold value
    return result_dict
