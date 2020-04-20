import numpy as np
import random
import operator
import pandas as pd
import sys




class FileReader:

    def makeCities(pathname):
        '''Makes an array of cities based on file input'''
        dimension = 0
        cities = []
        with open(pathname) as file:
            line = file.readline()
            while line != 'NODE_COORD_SECTION\n':
                if line == 'TYPE: TSP\n':
                    dimension = int(file.readline().split(' ')[1])
                if line == 'TYPE : TSP\n':
                    dimension = int(file.readline().split(' ')[2])
                line = file.readline()

            for i in range(dimension):
                coord = file.readline().split(' ')
                cities.append(City(float(coord[1]), float(coord[2]), i + 1))

        return cities


class City:
    # Added parameter index
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

    def distance(self, city):
        xDistance = abs(self.x - city.x)
        yDistance = abs(self.y - city.y)
        euclideanDistance = np.sqrt((xDistance ** 2) + (yDistance ** 2))
        return euclideanDistance

    # Changed __repr__ to return index. Might revert back to coords
    def __repr__(self):
        return str(self.index)


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

    def createRoute(self, cityList):
        route = random.sample(cityList, len(cityList))
        return route

    def initialPopulation(self, popSize, cityList):
        population = []

        for i in range(0, popSize):
            population.append(self.createRoute(cityList))
        return population

    def rankRoutes(self, population):
        fitnessResults = {}
        for i in range(0, len(population)):
            fitnessResults[i] = Fitness(population[i]).routeFitness()
        return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

    def selection(self, popRanked, eliteSize):
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        for i in range(0, eliteSize):
            selectionResults.append(popRanked[i][0])
        for i in range(0, len(popRanked) - eliteSize):
            pick = 100 * random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i, 3]:
                    selectionResults.append(popRanked[i][0])
                    break
        return selectionResults

    def pool(self, population, selectionResults):
        matingpool = []
        for i in range(0, len(selectionResults)):
            index = selectionResults[i]
            matingpool.append(population[index])
        return matingpool

    def crossover(self, parent1, parent2):
        child = []
        childParent1 = []
        childParent2 = []

        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        # ordered crossover
        for i in range(startGene, endGene):
            childParent1.append(parent1[i])

        childParent2 = [item for item in parent2 if item not in childParent1]

        child = childParent1 + childParent2
        return child

    def makeOffspring(self, matingpool, eliteSize):
        children = []
        length = len(matingpool) - eliteSize
        pool = random.sample(matingpool, len(matingpool))

        for i in range(0, eliteSize):
            children.append(matingpool[i])

        for i in range(0, length):
            child = self.crossover(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        return children

    def mutate(self, individual, mutationRate):
        for swapped in range(len(individual)):
            if (random.random() < mutationRate):
                swap = int(random.random() * len(individual))

                city1 = individual[swapped]
                city2 = individual[swap]

                individual[swapped] = city2
                individual[swap] = city1
        return individual

    def mutatePopulation(self, population, mutationRate):
        mutatedPopulation = []

        for ind in range(0, len(population)):
            mutatedIndex = self.mutate(population[ind], mutationRate)
            mutatedPopulation.append(mutatedIndex)
        return mutatedPopulation

    def nextGeneration(self, currentGen, eliteSize, mutationRate):
        popRanked = self.rankRoutes(currentGen)
        selectionResults = self.selection(popRanked, eliteSize)
        matingpool = self.pool(currentGen, selectionResults)
        children = self.makeOffspring(matingpool, eliteSize)
        nextGeneration = self.mutatePopulation(children, mutationRate)
        return nextGeneration

    def geneticAlgorithm(self, population, popSize, eliteSize, mutationRate, generations):
        pop = self.initialPopulation(popSize, population)

        for i in range(0, generations):
            pop = self.nextGeneration(pop, eliteSize, mutationRate)

        fit = (1 / self.rankRoutes(pop)[0][1])
        # Prints final distance as an integer
        print(int(fit))
        bestRouteIndex = self.rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        return bestRoute


if __name__ == '__main__':
    # Run the script through CMD/Terminal 'python ModulePath FilePath'
    cityList = FileReader.makeCities(sys.argv[1])

    pop = 100
    elite = 30
    mut = 0.001
    gen = 500

    print("Shortest path found: ")
    path = Fitness(cityList).geneticAlgorithm(cityList, pop, elite, mut, gen)
    # Disabled path printing for now
    # for i in path:
    #     print(i, end=', ')
