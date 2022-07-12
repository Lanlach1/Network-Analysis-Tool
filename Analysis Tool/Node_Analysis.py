# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 12:27:50 2022

@author: Owner
"""
import math
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pandas as pd



node_path = 'C:/Python_Programs/Fai_ER_Project/Example Networks/Neuromuscular Junction/nodes.xls'#excel file conatining x y coordinates of all nodes within network. Look at github repo for examples.
edge_path = 'C:/Python_Programs/Fai_ER_Project/Example Networks/Neuromuscular Junction/edges.xls' #excel file containing adjecency matrix of network. Please look at examples in Github Repo for reference.
image_path = 'C:/Python_Programs/Fai_ER_Project/Example Networks/Neuromuscular Junction/image.png' #if you extracted network from an image you can use it as a nackround, if no image put None.
image_bounds = [0, 188, 0, 188] #defines the xy bounds of the image in question, if image path is None then you dont need to define this.
name = "network"

save_graph = False #if you want to save graph set to True
save_graph_path = "C:/Users/Owner/Downloads" #name of file you want to save graph to. Will save in both PDF and PNG form.
line_width = 2
line_color = "white"
figure_dpi = 200

class Network:
    def __init__(self, points, edges, ghost_points, name, im):
        self.points = points
        self.edges = edges
        self.ghost_points = ghost_points
        self.name = name
        self.im = im
    def plotdet(self, line_width, line_color):
        D = np.zeros((len(self.edges[:,0]),len(self.edges[:,0])))
        if self.ghost_points is None:
            for i in range(len(self.edges[:,0])):
                for j in range(len(self.edges[:,0])):
                    if D[j,i] == 1: #This stops repetitions
                        pass
                    elif self.edges[i,j] != 0: # if a centroid shares 2 vertices with another centroid then they are connected
                        connectpoints(self.points[:,0],self.points[:,1],i,j, line_width, line_color) #Connects them in the next graph shown
                        D[i,j] = 1            
        else: #this else is the code for plotting periodic graphs
            plt.plot([0,0],[1,0], ':', c = 'black')
            plt.plot([0,1],[0,0], ':', c = 'black')
            plt.plot([0,1],[1,1], ':', c = 'black')
            plt.plot([1,1],[1,0], ':', c = 'black')
            plt.scatter(self.ghost_points[:,0],self.ghost_points[:,1], c = 'black')
            n = 0
            for i in range(len(self.edges[:,0])):
                for j in range(len(self.edges[:,0])):
                    if D[j,i] == 1: #This stops repetitions
                        pass
                    elif self.edges[i,j] != 0: # if a centroid shares 2 vertices with another centroid then they are connected
                        for k in range(len(self.ghost_points[:,0])):
                            if self.ghost_points[k,3] == i and self.ghost_points[k,2] == j:
                                connectpoints_list(self.points[:,0],self.points[:,1],self.ghost_points[:,0],self.ghost_points[:,1],i,k) #Connects a real points to a ghost point in the next graph shown
                                n = 1  
                        if n == 0:
                            connectpoints(self.points[:,0],self.points[:,1],i,j)
                            D[i,j] = 1
                        n = 0
    def theory_prob_gen_opt(self, step):
        A = np.zeros((len(self.points[:,0]), len(self.points[:,0])))
        for i in range(len(self.points[:,0])):
            for j in np.nonzero(self.edges[i,:])[0]:
                A[i,j] = math.ceil(self.edges[i,j]/step)*step
        p = np.zeros((len(self.points[:,0]), len(self.points[:,0])))
        for init in range(len(self.edges[:,0])):
            for i in np.nonzero(self.edges[init,:])[0]:
                p[init, i] = (1/np.count_nonzero(A[init,:]))*(step/A[init, i])*(1/(1-((1/np.count_nonzero(A[init,:]))*np.sum(1-(step / (np.array([j for j in A[init,:] if j != 0])))))))
        return(p)
    def theory_stat(self, targets, step): #will copmute the fendmental matrix given target conditions and a step size
        p = self.theory_prob_gen_opt(step)
        n = 0
        for i in sorted(targets):
            p = np.delete(p, i-n, 0)
            p = np.delete(p, i-n, 1)
            n += 1
        N = np.linalg.inv((np.identity(len(self.edges[:,0])-len(targets)) - p))
        return N
    def average_target_step_theory(self, step, image_bounds, save_graph, save_graph_path, line_width, line_color, figure_dpi): #same as previous except non uniform  transistion probability
        Ave_MFPT = np.zeros(len(self.points))
        occupancy = np.zeros((len(self.points)-1, len(self.points)))
        for i in tqdm(range(len(self.points))):
            occupancy[:,i] = np.sum(self.theory_stat([i], step), axis = 0)
        for i in range(len(self.points)):
            n = 0
            for j in range(len(self.points)-1):
                    n += occupancy[j,i]/(len(self.points)-1)
            Ave_MFPT[i] += n
        #for i in range(len(Ave_MFPT)):
            #if Ave_MFPT[i] >= 22000:
                #Ave_MFPT[i] = 22000
        plt.figure(dpi = figure_dpi)
        #for j in range(len(self.points[:,0])):
            #plt.text(self.points[j,0], self.points[j,1], f"{j}")
        if self.im is None:
            pass
        else:           
            plt.rcParams["figure.figsize"] = [13.00,10.50]
            plt.rcParams["figure.autolayout"] = True
            im = plt.imread(self.im)
            fig, ax = plt.subplots(dpi = figure_dpi)
            im = ax.imshow(im, extent=image_bounds)
        self.plotdet(line_width, line_color)
        plt.scatter(self.points[:,0], self.points[:,1], s = 140, c = Ave_MFPT, cmap = 'rainbow', alpha = 1, zorder = 2)
        plt.colorbar()
        plt.axis('off')
        if save_graph:
            plt.savefig(str(save_graph_path) + f'/{self.name}_Node_ana.pdf', format='pdf')
            plt.savefig(str(save_graph_path) + f'/{self.name}_Node_ana.png', format='png')
        plt.show() 
        plt.rcdefaults()
def connectpoints_list(xone,yone,xtwo,ytwo,p1,p2): #Will make a line between the two points in the next graph, using two differnt lists of points
    x1, x2 = xone[p1], xtwo[p2]
    y1, y2 = yone[p1], ytwo[p2]
    plt.plot([x1,x2],[y1,y2], 'k-', zorder = 1)#, 'white')
def connectpoints(x,y,p1,p2, line_width, line_color): #Will make a line between the two points in the next graph
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2], zorder = 1, c = line_color, linewidth = line_width)# , 'k-'
def ER_network(name, diction, im_path, edge_path, node_path, image_bounds):
    connections_data = pd.read_excel(edge_path, header=None)#use r before absolute file path 
    #connections_data = pd.read_excel(str(edge_path) + f"/edges{order}_matlab_flip_v2.xls",header=None)
    connections = connections_data.values[:,:]
    real_points_data = pd.read_excel(node_path, header=None)
    #real_points_data = pd.read_excel(str(node_path) + f"/verts{order}_matlab_flip_v2.xls",header=None)
    real_points  = real_points_data.values
    edges = np.zeros((len(real_points),len(real_points)))
    for i in range(len(connections)):
        a = int(connections[i,0] - 1) #the minus one is from indexing issues
        b = int(connections[i,1] - 1)
        edges[a,b] = math.sqrt(((real_points[a,0] - real_points[b,0])**2) + ((real_points[a,1] - real_points[b,1])**2))
        edges[b,a] = math.sqrt(((real_points[a,0] - real_points[b,0])**2) + ((real_points[a,1] - real_points[b,1])**2))
    globals()[name] = Network(real_points, edges, None, name, image_path)
    diction[name] = globals()[name]
    plt.rcdefaults()
    
network_dict = {"test" : "Test"}


ER_network(name, network_dict, image_path, edge_path, node_path, image_bounds)
network_dict[name].average_target_step_theory(0.1, image_bounds, save_graph, save_graph_path, line_width, line_color, figure_dpi)