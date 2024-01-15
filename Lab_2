import pygmo as pg
import numpy as np
import pandas as pd

#Тестовая функция №1 (Сфера)
class Sphere:
  def fitness(self, x):
        return [sum(x**2)]

  def get_bounds(self):
        return ([-5, -5], [5, 5])


#Тестовая функция №2 (Растригин)
class Rastrigin:
  def fitness(self, x):
    return [10 * len(x) + sum([xi**2 - 10 * np.cos(2*np.pi*xi) for xi in x])]

  def get_bounds(self):
    return ([-5, -5], [5, 5])

#Создание экземпляров задач
sphere_problem = pg.problem(Sphere())
rastrigin_problem = pg.problem(Rastrigin())

#Создание экземпляров алгоритмов
algorithm1 = pg.algorithm(pg.de(gen = 1000))      #Differential Evolution
algorithm2 = pg.algorithm(pg.pso(gen = 1000))     #Particle Swarm Optimization
algorithm3 = pg.algorithm(pg.sade(gen = 1000))    #Self-adaptive Differential Evolution

#Создание популяции и запуск алгоритмов
populationSph = pg.population(prob = sphere_problem, size = 100)
populationRas = pg.population(prob = rastrigin_problem, size = 100)

#Запуск алгоритмов на популяциях
population1 = algorithm1.evolve(populationSph)
population2 = algorithm2.evolve(populationSph)
population3 = algorithm3.evolve(populationSph)
population4 = algorithm1.evolve(populationRas)
population5 = algorithm2.evolve(populationRas)
population6 = algorithm3.evolve(populationRas)

data = {
    'Алгоритмы': ['DE', 'PSO', 'SADE'],
    'Сфера': [population1.champion_f, population2.champion_f, population3.champion_f],
    'Растригин': [population4.champion_f, population5.champion_f, population6.champion_f]
}

df = pd.DataFrame(data)
print(df)
