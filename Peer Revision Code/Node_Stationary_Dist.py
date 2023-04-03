# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 12:27:50 2022

@author: Owner
"""
import math
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.linalg import eig 

save_data = False #if true, program will run analysis on network and save MFPT data as excel file to save_data_path. if False, program will take analysis from excel file in save_data_path and not run analysis. If you are adjusting a figure and want to run it multiple times, it is good to switch this from true to false after generating the first plot.
save_data_path = 'C:/Python_Programs/Fai_ER_Project/Peer_Revision_Work/Final Graph Edits 2/MFPT_NMJ_Network 1 weight.xlsx'
image_path = 'C:/Python_Programs/Fai_ER_Project/NMJ_middle2/NMJ_middle2_im1.png' #if you extracted network from an image you can use it as a nackround, if no image put None.
image_bounds = [0, 188, 0, 188] #defines the xy bounds of the image in question, if image path is None then you dont need to define this.
edge_path = 'C:/Python_Programs/Fai_ER_Project/NMJ_middle2/NMJ_middle2_edges1.xls' #excel file containing adjecency matrix of network. Please look at examples in Github Repo for reference.
node_path = 'C:/Python_Programs/Fai_ER_Project/NMJ_middle2/NMJ_middle2_verts1.xls'#excel file conatining x y coordinates of all nodes within network. Look at github repo for examples.
name = "NMJ 1 Stationary Distribution" #name of network, will be used for naming files and such
line_width = 4 #with of the lines connecting juctins, will need to be adjusted for different sized networks. 
save_graph = True #if you want to save graph set to True
save_graph_path = "C:/Python_Programs/Fai_ER_Project/Peer_Revision_Work/Final Graph Edits 2" #name of file you want to save graph to.
class Network:
    def __init__(self, points, edges, ghost_points, name, im):
        self.points = points
        self.edges = edges
        self.ghost_points = ghost_points
        self.name = name
        self.im = im
    def plotdet(self):
        D = np.zeros((len(self.edges[:,0]),len(self.edges[:,0])))
        if self.ghost_points is None:
            for i in range(len(self.edges[:,0])):
                for j in range(len(self.edges[:,0])):
                    if D[j,i] == 1: #This stops repetitions
                        pass
                    elif self.edges[i,j] != 0: # if a centroid shares 2 vertices with another centroid then they are connected
                        connectpoints(self.points[:,0],self.points[:,1],i,j) #Connects them in the next graph shown
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
    def theory_stat(self, targets, step): #will copmute the fundamental matrix given target conditions and a step size
        p = self.theory_prob_gen_opt(step)
        n = 0
        for i in sorted(targets):
            p = np.delete(p, i-n, 0)
            p = np.delete(p, i-n, 1)
            n += 1
        N = np.linalg.inv((np.identity(len(self.edges[:,0])-len(targets)) - p))
        return N
    def stationary_dist(self, step):
        P = self.theory_prob_gen_opt(step)
        S, U = eig(P.T)
        stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
        stationary = stationary / np.sum(stationary)
        return stationary.real
    def stationary_dist_plot(self, step, image_bounds, save_graph, save_graph_path): #same as previous except non uniform  transistion probability
        stationary = self.stationary_dist(step)
        #for i in range(len(Ave_MFPT)):
            #if Ave_MFPT[i] >= 22000:
                #Ave_MFPT[i] = 22000
        plt.figure(dpi = 1000)
        if self.im is None:
            pass
        else:           
            plt.rcParams["figure.figsize"] = [13.00,10.50]
            plt.rcParams["figure.autolayout"] = True
            im = plt.imread(self.im)
            fig, ax = plt.subplots()
            im = ax.imshow(im, extent=image_bounds)
        self.plotdet()
        plt.scatter(self.points[:,0], self.points[:,1], s = 100, c = stationary, cmap = 'rainbow', alpha = 1, zorder = 2)
        #for j in range(len(self.points[:,0])):
            #plt.text(self.points[j,0], self.points[j,1], f"{j}")
        plt.colorbar()
        plt.axis('off')
        if save_graph:
            plt.savefig(str(save_graph_path) + f'/{self.name}.pdf', format='pdf')
            plt.savefig(str(save_graph_path) + f'/{self.name}.png', format='png')
        plt.show() 
        plt.rcdefaults()
def connectpoints_list(xone,yone,xtwo,ytwo,p1,p2): #Will make a line between the two points in the next graph, using two differnt lists of points
    x1, x2 = xone[p1], xtwo[p2]
    y1, y2 = yone[p1], ytwo[p2]
    plt.plot([x1,x2],[y1,y2], 'k-', zorder = 1)#, 'white')
def connectpoints(x,y,p1,p2): #Will make a line between the two points in the next graph
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2], zorder = 1, c = 'white')# , 'k-'
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
network_dict[name].stationary_dist_plot(0.1, image_bounds, save_graph, save_graph_path)