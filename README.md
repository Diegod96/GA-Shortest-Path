# GAShortestPath

Genetic Algorithm that tries to solve the traveling salesmen problem: *“Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city and returns to the origin city?”*

text files contain city coordinates (Lat & Long of cities) of a country.

Due to it being a genetic algortm, it may never get the shortest path. Although, it will get close to it.

You can play around with the elite size, mutation rate, # of generations, and the population to see how close you can get to the shortest path for the given countries tsp data.

To Run:
* Clone the repo
* Import numpy, random, operator, pandas, and sys
* Run main.py in the terminal as such ```python main.py countrycities.txt```

TODO:
* Add some sort of grpahical visulaiazation to show performance of metrics like different population sizes, elite sizes, etc.
* Find larger tsp data of countries and see how the program fairs time wise

<img width="412" alt="Capture" src="https://user-images.githubusercontent.com/25403763/79797160-c7a8a100-8324-11ea-970a-1959c98ff21d.PNG">



