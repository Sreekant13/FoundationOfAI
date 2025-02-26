import random
import numpy as np

def random_int_generation(low, high):# Generating a random integer in the range of low and high
    return random.randint(low, high)

def generate_population(size, cities): #Genrating randomized population
    randomized_populated_paths = [] # we will use this variable to store all the possible number of random paths
    for _ in range(size):
        path = cities[:] # this line will create copy of the "cities" list using [:], since we don't want to tamper with the original list
        random.shuffle(path)  # Custom shuffle function, this will randomly shuffle the cities stored in the "path", which might be a possible solution of the TSP
        randomized_populated_paths.append(path) # will insert all possible random paths in "population"
    return randomized_populated_paths

store_and_save_distance_between_each_city = {} # Using Dictionary to store the distance between each of the cities

def compute_distance_between_cities(city1, city2): #Calculating distance between cities and storing them for future use
    if (city1, city2) not in store_and_save_distance_between_each_city: #checking if the distance between two particular cities has already been captured or not 
        store_and_save_distance_between_each_city[(city1, city2)] = np.linalg.norm(np.array(city1) - np.array(city2)) #Caluculating distance between cities using numpy
    return store_and_save_distance_between_each_city[(city1, city2)]

def calculate_path_distance(path): #this funtion will calculate the total distance of the given path through series of cities
    total_distance = 0
    for i in range(len(path) - 1):
        # The loop iterates through each city in the path and adds all the distance between those cities
        total_distance += compute_distance_between_cities(path[i], path[i+1])
    # Finally after the distance between all the cities in the particular path has been added to the total_distance,
    # we add the distance of the path to return to the starting city
    total_distance += compute_distance_between_cities(path[-1], path[0])
    return total_distance

def rank_population(population):
    distances = [(path, calculate_path_distance(path)) for path in population] # Calculating distances for each path in the population
    fitness_scores = [(path, 1 / distance) for path, distance in distances] # Calculating fitness (inverse of distance) for each path
    fitness_scores.sort(key=lambda x: x[1], reverse=True) # Extracting the fitness scores for sorting (just for clarity) in descending order
    return fitness_scores #returning the scores

def maintain_diversity_in_Genetic_Algorithm(population, upper_bound=0.02):
    # we will randomly replace some population members if they are too similar
    unique_population = set(tuple(path) for path in population) #taking unique path population from from the population 
    len_unique_population = len(unique_population)
    len_population = len(population)
    diversity_ratio = len_unique_population / len_population #calculating the diversity ration
    
    if diversity_ratio < upper_bound:  # If diversity is below upper_bound
        random_new_paths = generate_population(len(population) - len(unique_population), list(population[0]))  # we will reintroduce random paths if diversity ratio is less then the upper_bound which we give
        population = list(unique_population) + random_new_paths
    return population

def Greedy_Heuristic_for_nearest_city(cities): #This Greedy Heuristic is being used for initializing the population
    unvisited_Cities = cities[:] #Assigning all the cities as unvisited initially 
    present_city = unvisited_Cities.pop(0)  # Start with the first city
    path = [present_city]
    
    while unvisited_Cities: # while a city is not visited, we will take the next city which has a minimum distance from the present city
        next_city = min(unvisited_Cities, key=lambda city: compute_distance_between_cities(present_city, city)) 
        unvisited_Cities.remove(next_city) # we will then remove the city
        path.append(next_city) # and append it in the path
        present_city = next_city # our present_city becomes the next_city
    return path

def select_parent(ranked_population): # This funstion ensures that better paths(shorter distance) has chance as a parent for reproduction
    total_fitness = sum([result_score for path, result_score in ranked_population]) # It sums up all the fitness score of all paths in the ranked_population
    pick = random.random() * total_fitness # we generate a random number between 0 and total fitness and pick the solution of that path
    current = 0
    for path, fitness in ranked_population: # We then iterate through the rank_population 
        current += fitness
        if current > pick: # When the total current fitness in the ranked_population exceeds the pick(ed) solution, we then return that solution
            return path

def ordered_crossover_for_lower_number_of_cities(first_parent, second_parent): # we will use this for lower values of number of cities
    #Inspired by biological reproduction, this funstion takes traits from both first_parent and second_parent and combine them to form a child
    size_of_first_parent = len(first_parent)
    start_index = random_int_generation(0, size_of_first_parent - 2) # Randomly selecting a starting city for first_parent
    end_index = random_int_generation(start_index + 1, size_of_first_parent - 1) # Randomly selecting an ending city for first_parent
    child = [None] * size_of_first_parent #initializing None to each of the paths in child
    child[start_index:end_index + 1] = first_parent[start_index:end_index + 1] #copying the same selected paths of the cities from first_parent to child

    current_position = end_index + 1
    for city in second_parent: #assigning the rest of the cities in child from second_parent whereever the city is not already present in child returning it
        if city not in child:
            if current_position >= size_of_first_parent:
                current_position = 0
            child[current_position] = city
            current_position += 1
    return child

def partially_mapped_crossover_for_higher_number_of_cities(first_parent, second_parent): # we will use this for higher values of number of cities
    size_of_first_parent = len(first_parent)
    start_index = random_int_generation(0, size_of_first_parent - 2) # Randomly selecting a starting city for first_parent
    end_index = random_int_generation(start_index + 1, size_of_first_parent - 1) # Randomly selecting an ending city for first_parent
    child = [None] * size_of_first_parent #initializing None to each of the paths in child
    child[start_index:end_index + 1] = first_parent[start_index:end_index + 1] #copying the same selected paths of the cities from first_parent to child

    for i in range(start_index, end_index+1):
        if second_parent[i] not in child: #if second_parent(city) is not in child then we will take that city and store it in city
            city = second_parent[i]
            pos = i
            while True:
                pos = second_parent.index(first_parent[pos])
                if child[pos] is None: # we will assign the child at the position of the first parent with the child of the second parent if it is empty
                    child[pos] = city
                    break #giving a break statement so that if there is the same city present at a postion in both first and second parent, it might go to an infinite loop
    for i in range(size_of_first_parent): #checking and filling whatever city in child is empty with the second parent
        if child[i] is None:
            child[i] = second_parent[i]                
    return child

def crossover(first_parent, second_parent, number_of_cities): #crossover function based on the number of coordinates
    if number_of_cities > 100:
        return partially_mapped_crossover_for_higher_number_of_cities(first_parent, second_parent)
    else:
        return ordered_crossover_for_lower_number_of_cities(first_parent, second_parent)

def mutate(path, rate_of_mutation=0.01):
    #This funstion is used to swap/shuffle cities or parts of paths to let variety enter into the population
    n = len(path)
    for i in range(n):
        if random.random() < rate_of_mutation:
            start_index = random_int_generation(0, n - 2) # randomizing start_index
            end_index = random_int_generation(start_index + 1, n - 1) # randomizing end_index
            sub_path = path[start_index:end_index + 1] #creating a sub-path of the path from and to the starting and ending index generating variety
            random.shuffle(sub_path) # shuffling the sub_path
            path[start_index:end_index + 1] = sub_path # storing back the shuffled sub_path in path

def TSP_Homework1_Genetic_Algorithm(number_of_cities, cities, population_size, generations, rate_of_mutation, size_of_paths_of_optimal_solution, diversity_check_interval):
    # Step 1: Create initial population
    # This funstion stores multiple random permutations(paths) of all the cities
    # Each of those randomly generated permutations has the potential to be a discrete solution to this problem
    population = generate_population(population_size - 1, cities)
    population.append(Greedy_Heuristic_for_nearest_city(cities))  # Include the greedy heuristic-based path for better and optimized solution 

    for generation in range(generations):# This loop develops as more and more optimized paths created(using selection, crossover and mutation) become potential solution to this problem
        # In addtion we can say that with each loop, there is a better and optimized solution to this TSP
        # Ranking population by fitness
        # The population is ranked by their fitness based on shortest path distance to be more fit(as they have higher fitness values) for the solution and vice-versa
        # Hence the "ranked_population" is sorted based on fitness, the higher fitness value paths(shorter distnace path) is on top
        ranked_population = rank_population(population)
        # Keep the top 'size_of_paths_of_optimal_solution' solutions (The top tier solutions, which we don't want to lose after the selection, crossover and mutation)
        new_population = [path for path, _ in ranked_population[:size_of_paths_of_optimal_solution]]

        while len(new_population) < population_size:
            # We choose first_parent and second_parent for the crossover and mutation
            # It is highly likely to happen that the population with higher fit path(shorter distance) is chosen as parent
            first_parent = select_parent(ranked_population)
            second_parent = select_parent(ranked_population)
            # The crossover funstion takes 2 random parents and produce 2 children by combining parts of the parent's path
            # Algorithm takes randomly portions of first_parent and fills the rest of the path of cities with second_parent
            first_child = crossover(first_parent, second_parent, number_of_cities) 
            second_child = crossover(second_parent, first_parent, number_of_cities) 
            # As we know that we can be stuck in local optima, this mutation will help us to be free from that
            # It will basically take both the children(first_child and second_child) after selection and crossover for it's funstioning
            # If there is a small possibility that some city might swap position with another city in the path
            # This will be useful to help introduce new genetic diversity(or in other words new updated better path)
            mutate(first_child, rate_of_mutation)
            mutate(second_child, rate_of_mutation)
            # Replacing the new updated better paths with the older one's after crossover and mutation
            new_population.append(first_child)
            new_population.append(second_child)

        if generation % diversity_check_interval == 0:
            population = maintain_diversity_in_Genetic_Algorithm(new_population)
        else:
            population = new_population
    # After generations, we will return the best path found by the Algo
    best_path = rank_population(population)[0][0]
    best_distance = calculate_path_distance(best_path) # Using best_path, we will calculate the the path distance and that will be out final answer
    
    return best_distance, best_path

def read_input(input_filename): # Function to read input.txt file
    with open(input_filename, 'r') as f: # will open the file with reading purpose from it
        lines = f.readlines() #reading each line
    number_of_cities = int(lines[0].strip()) #storing the number of cities(as n)
    coordinates_of_cities = [tuple(map(int, each_line.strip().split())) for each_line in lines[1:]] #storing the coordinates of all the n number of cities in "cities"
    return number_of_cities, coordinates_of_cities

def write_output(output_filename, result): # Funtion to write the output.txt file
    fianl_distance, path = result
    with open(output_filename, 'w') as f: # will open the file with writing purpose to it
        f.write(f"{fianl_distance:.3f}\n") # will write the total distance calculated(of shortest path) and stored in the best_distance
        for city in path: # this loop is used to go through and print all the coordinates(stored in the path variable) once
            f.write(f"{city[0]} {city[1]} {city[2]}\n")
        f.write(f"{path[0][0]} {path[0][1]} {path[0][2]}\n") # According to question, writing the city coordinates(visited in the beginning) again at the end


number_of_cities, coordinates_of_cities = read_input("input.txt")
#best_distance, best_path = TSP_Homework1_Genetic_Algorithm(coordinates_of_cities, population_size=100, generations=150, rate_of_mutation=0.01, size_of_paths_of_optimal_solution=30)
if number_of_cities<200: #number_of_cities is less than 200, we will use the given below values for each
    best_distance, best_path = TSP_Homework1_Genetic_Algorithm(number_of_cities, coordinates_of_cities, population_size=100, generations=150, rate_of_mutation=0.01, size_of_paths_of_optimal_solution=40, diversity_check_interval=10)
elif number_of_cities>=200 and number_of_cities <300: #number_of_cities is between (200,300], we will use the given below values for each
    best_distance, best_path = TSP_Homework1_Genetic_Algorithm(number_of_cities, coordinates_of_cities, population_size=100, generations=150, rate_of_mutation=0.01, size_of_paths_of_optimal_solution=30, diversity_check_interval=25)
else: #number_of_cities is above 299, we will use the given below values for each
    best_distance, best_path = TSP_Homework1_Genetic_Algorithm(number_of_cities, coordinates_of_cities, population_size=100, generations=150, rate_of_mutation=0.01, size_of_paths_of_optimal_solution=40, diversity_check_interval=50)
result = (best_distance, best_path)
write_output("output.txt", result)