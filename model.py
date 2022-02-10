import random
import math
from enum import Enum
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import time
import numpy as np
import numpy as num
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import csv






class State(Enum):
    CSC = 1# Tumor Cells
    inflammatory = 2
    Stem = 3.# Normal Cells
    Dead = 4# Apoptotic Cells



def number_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.state is state])

def number_CSC(model):
    return number_state(model, State.CSC)

def number_Stem(model):
    return number_state(model, State.Stem)

def number_M(model):
    return number_state(model, State.inflammatory)

def number_dead(model):
    return number_state(model, State.Dead)

class TumorModel(Model):
    """A Tumor model with some number of agents"""

    def __init__(self, num_nodes, avg_node_degree, initial_Tumor_cell, Angiogenesis_chance, Tumor_propagation_Frequency,
                  Angioprevention_chance):

        #self.num_nodes = num_nodes
        self.G = nx.read_graphml("Growcluster7.graphml")
        self.E = nx.ego_graph(self.G, n="3", radius=1, center=True, undirected=False, distance=None)
        self.SubG = dict(nx.subgraph_centrality(self.E))
        #self.edges = self.G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])
        #self.CG = nx.average_clustering(self.G, nodes=self.num_nodes, weight=self.edges, count_zeros=True)
        #self.DS = nx.directed.configuration_model(self.edges, self.G)
        #self.CG = nx.random_clustered_graph(self.edges, self.DS)
        #self.path_length = nx.all_pairs_shortest_path_length(self.G)
        #self.distances = np.zeros((len(self.G), len(self.G)))
        #self.Edgelist = nx.read_edgelist("Tumor.edgelist")
        #self.subG = nx.connected_component_subgraphs(self.G)[0]# Extract largest connected component into graph H
        #self.H = nx.convert_node_labels_to_integers(self.subG) # Makes life easier to have consecutively labeled integer nodes
        #self.BM = nx.blockmodel(self.H, self.partitions)  # Build blockmodel graph

        self.D = nx.density(self.G)

        #self.DAT = num.array(self.C)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.initial_Tumor_cell = initial_Tumor_cell if initial_Tumor_cell <= num_nodes else num_nodes
        self.Angiogenesis_chance = Angiogenesis_chance
        self.Tumor_propagation_Frequency = Tumor_propagation_Frequency
        self.Angioprevention_chance = Angioprevention_chance

        self.datacollector = DataCollector({"CSC": number_CSC,
                                            "inflammatory": number_M,
                                            "Dead": number_dead,
                                            "Stem": number_Stem})


        t_start = time.time()
        # Create agents
        for i, node in enumerate(self.G.nodes()):

            a = TumorAgent(i, self, self.Angiogenesis_chance, self.Tumor_propagation_Frequency, self.initial_Tumor_cell,
                           self.Angioprevention_chance)
            self.schedule.add(a)
            # Add the agent to the node
            self.grid.place_agent(a, node)



        Cancer_nodes = random.sample(self.G.nodes(), self.initial_Tumor_cell)
        for a in self.grid.get_cell_list_contents(Cancer_nodes):
            a.state = State.CSC

        self.running = True
        self.datacollector.collect(self)

        w = csv.writer(open("Subgraphdic4.csv", "w"))
        #w = csv.writer(open("Subgraphpatt1.csv", "w"))

        for key, val in dict.items(self.SubG):
            w.writerow([key, val])

        t_end = time.time()
        print("[ Lapse: {} seconds".format(t_end - t_start))



    def Density_Calculation(self):
         try:
          return self.D
         except ZeroDivisionError:
          return math.inf

            #def create_hc(self):
       #Creates hierarchical cluster of graph G from distance matrix
       #for u, p in self.path_length.items():
            #for v, d in p.items():
            #   self.distances[u][v] = d
        # Create hierarchical cluster
            # Y = distance.squareform(self.distances)
            #Z = hierarchy.complete(Y)  # Creates HC using farthest point linkage
        # This partition selection is arbitrary, for illustrive purposes
            #membership = list(hierarchy.fcluster(Z, t=1.15))
        # Create collection of lists for blockmodel
        # partition = defaultdict(list)
        # for n, p in zip(list(range(len(self.G))), membership):
        # partition[p].append(n)
        # return list(partition.values())
        # partitions = create_hc(self.H)  # Create parititions with hierarchical clustering


    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        print("model run")
        for i in range(n):
            self.step()



class TumorAgent(Agent):
    def __init__(self, unique_id, model, initial_state, Angiogenesis_chance, Tumor_propagation_Frequency,
                  Angioprevention_chance):
        super().__init__(unique_id, model)

        self.state = initial_state

        self.Angiogenesis_chance = Angiogenesis_chance
        self.Tumor_propagation_Frequency = Tumor_propagation_Frequency
        self.Angioprevention_chance = Angioprevention_chance

    def try_to_inflammate_neighbors(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        inflammatory_neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes) if
                                 agent.state is State.CSC]

        for a in inflammatory_neighbors:
            if random.uniform(0,1) < self.Angiogenesis_chance:
                a.state = State.inflammatory

    def try_gain_Quiescent(self):
        if random.uniform(0,1) < self.Angioprevention_chance:
            self.state = State.Dead

    def try_check_situation(self):
        if random.uniform(0,1) > self.Tumor_propagation_Frequency:
            self.try_gain_Quiescent
        else:
            self.try_to_inflammate_neighbors()


    def step(self):
            if self.state is State.Stem:
                self.try_to_inflammate_neighbors()
            self.try_check_situation()



