# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:29:37 2022

@author: Owner
"""

import math
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pandas as pd
import networkx as nx


save_data = False #if true, program will run analysis on network and save MFPT data as excel file to save_data_path. if False, program will take analysis from excel file in save_data_path and not run analysis. If you are adjusting a figure and want to run it multiple times, it is good to switch this from true to false after generating the first plot.
save_data_path = 'C:/Python_Programs/Fai_ER_Project/Peer_Revision_Work/Final Graph Edits 2/MFPT_NMJ_Network 1 weight.xlsx'
image_path = 'C:/Python_Programs/Fai_ER_Project/NMJ_middle2/NMJ_middle2_im1.png' #if you extracted network from an image you can use it as a nackround, if no image put None.
image_bounds = [0, 188, 0, 188] #defines the xy bounds of the image in question, if image path is None then you dont need to define this.
edge_path = 'C:/Python_Programs/Fai_ER_Project/NMJ_middle2/NMJ_middle2_edges1.xls' #excel file containing adjecency matrix of network. Please look at examples in Github Repo for reference.
node_path = 'C:/Python_Programs/Fai_ER_Project/NMJ_middle2/NMJ_middle2_verts1.xls'#excel file conatining x y coordinates of all nodes within network. Look at github repo for examples.
name = "NMJ 1 critical" #name of network, will be used for naming files and such
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
    def inversion_SM_formula(self, orig_inverse, u, v): #forumla to compute a rank 1 transformed inverted matrix.
        #A = orig_inverse - ((orig_inverse*u*np.transpose(v)*orig_inverse)/(1+np.transpose(v)*orig_inverse*u))
        u = u.reshape((len(u),1))
        v = v.reshape((len(v),1))
        A = orig_inverse - np.dot(orig_inverse, np.dot(np.dot(u,v.T), orig_inverse)) / ( 1 + np.dot(v.T, np.dot(orig_inverse, u)))
        return A
    def zero_degree_check(self, edges_without_ab): #checks if removing the edge between a and b creates a zero degree node, will say true if it does not.
        if False in (edges_without_ab.any(axis=1)):
            return False
        else:
            return True
    def SM_invert_check(self, orig_inverse, u1, v1, u2, v2):
        u1 = u1.reshape((len(u1),1))
        v1 = v1.reshape((len(v1),1))
        u2 = u2.reshape((len(u2),1))
        v2 = v2.reshape((len(v2),1))
        return (1 - math.isclose((1 + np.dot(v1.T, np.dot(orig_inverse, u1))), 0, abs_tol = 1e-10))*(1 - math.isclose((1 + np.dot(v2.T, np.dot(orig_inverse, u2))),0, abs_tol = 1e-10))
    def MFPT_global_dream_SM_two_junc(self, all_target_inverses, a, b, step, edges_without_ab, edges_copy): #Named is honor of the professors dream, will return the MFPT of all nodes given a missing edge connection between a and b
        Ave_MFPT = np.zeros(len(edges_copy))
        occupancy = np.zeros((len(edges_copy)-1, len(edges_copy)))
        for i in tqdm(range(len(edges_copy)), desc = f"Checking Edge {a}:{b}"):
            occupancy[:,i] = np.sum(self.theory_stat_dream_SM_two_junc(step, all_target_inverses[i,:,:], a, b, i, edges_without_ab, edges_copy), axis = 0)
        for i in range(len(edges_copy)):
            n = 0
            for j in range(len(edges_copy)-1):
                    n += occupancy[j,i]/(len(edges_copy)-1)
            Ave_MFPT[i] += n
        return np.average(Ave_MFPT)
    def theory_stat_dream_SM_two_junc(self, step, target_inverse, a, b, target, edges_without_ab, edges_copy): #will return the fundemental matrix after an edge removal using SM formula, MAKE SURE TO INPUT MATRIX WITH TARGET CONDITION REMOVED, make sure only one target is denoted.
        u1 = np.zeros((len(edges_copy),1))
        u2 = np.zeros((len(edges_copy),1))
        u1[a] = 1
        u2[b] = 1
        u1 = np.delete(u1, target)
        u2 = np.delete(u2, target)
        A = np.zeros((2, len(edges_copy)))
        edge = [a,b]
        for i in range(2):
            for j in np.nonzero(edges_copy[edge[i],:])[0]:
                A[i,j] = math.ceil(edges_copy[edge[i],j]/step)*step
        B = np.zeros((2, len(edges_copy)))
        for i in range(2):
            for j in np.nonzero(edges_without_ab[edge[i],:])[0]:
                B[i,j] = math.ceil(edges_without_ab[edge[i],j]/step)*step
        p_old = np.zeros((2, len(edges_copy)))
        p_new = np.zeros((2, len(edges_copy)))
        for init in range(2):
            for i in np.nonzero(edges_copy[edge[init],:])[0]:
                p_old[init, i] = (1/np.count_nonzero(A[init,:]))*(step/A[init, i])*(1/(1-((1/np.count_nonzero(A[init,:]))*np.sum(1-(step / (np.array([j for j in A[init,:] if j != 0])))))))
            for i in np.nonzero(edges_without_ab[edge[init],:])[0]:
                p_new[init, i] = (1/np.count_nonzero(B[init,:]))*(step/B[init, i])*(1/(1-((1/np.count_nonzero(B[init,:]))*np.sum(1-(step / (np.array([j for j in B[init,:] if j != 0])))))))
        p_old = np.delete(p_old, target, 1)
        p_new = np.delete(p_new, target, 1)
        v1 = p_old[0,:] - p_new[0,:]
        v2 = p_old[1,:] - p_new[1,:]
        #print([(1 + np.dot(v1.T, np.dot(target_inverse, u1))), (1 + np.dot(v2.T, np.dot(target_inverse, u2))), a, b, target])
        #print([math.isclose((1 + np.dot(v1.T, np.dot(target_inverse, u1))),0, abs_tol = 1e-10), math.isclose((1 + np.dot(v2.T, np.dot(target_inverse, u2))), 0, abs_tol = 1e-10), a, b, target])
        return self.inversion_SM_formula(self.inversion_SM_formula(target_inverse, u1, v1), u2, v2)
    def two_junc_dect(self, node): #will detect if a node is 2 junction
        if np.count_nonzero(self.edges[node,:]) == 2:
            return 1
        else:
            return 0
    def edge_seperation_check_two_junc(self, step, target_inverse, a, b, target, edges_without_ab, edges_copy):
        u1 = np.zeros((len(edges_without_ab),1))
        u2 = np.zeros((len(edges_without_ab),1))
        u1[a] = 1
        u2[b] = 1
        u1 = np.delete(u1, target)
        u2 = np.delete(u2, target)
        A = np.zeros((2, len(edges_without_ab)))
        edge = [a,b]
        for i in range(2):
            for j in np.nonzero(edges_copy[edge[i],:])[0]:
                A[i,j] = math.ceil(edges_copy[edge[i],j]/step)*step
        B = np.zeros((2, len(edges_copy)))
        for i in range(2):
            for j in np.nonzero(edges_without_ab[edge[i],:])[0]:
                B[i,j] = math.ceil(edges_without_ab[edge[i],j]/step)*step
        p_old = np.zeros((2, len(edges_copy)))
        p_new = np.zeros((2, len(edges_copy)))
        for init in range(2):
            for i in np.nonzero(edges_copy[edge[init],:])[0]:
                p_old[init, i] = (1/np.count_nonzero(A[init,:]))*(step/A[init, i])*(1/(1-((1/np.count_nonzero(A[init,:]))*np.sum(1-(step / (np.array([j for j in A[init,:] if j != 0])))))))
            for i in np.nonzero(edges_without_ab[edge[init],:])[0]:
                p_new[init, i] = (1/np.count_nonzero(B[init,:]))*(step/B[init, i])*(1/(1-((1/np.count_nonzero(B[init,:]))*np.sum(1-(step / (np.array([j for j in B[init,:] if j != 0])))))))
        p_old = np.delete(p_old, target, 1)
        p_new = np.delete(p_new, target, 1)
        v1 = p_old[0,:] - p_new[0,:]
        v2 = p_old[1,:] - p_new[1,:]
        return self.SM_invert_check(target_inverse, u1, v1, u2, v2)
    def simp_theory_stat(self, targets, step, p): #same as theory stat except you must enter the prob matrix, using the simplified prob matrix without two way junctions.
        n = 0
        for i in sorted(targets):
            p = np.delete(p, i-n, 0)
            p = np.delete(p, i-n, 1)
            n += 1
        N = np.linalg.inv((np.identity(len(p)) - p))
        return N
    def two_junc_list_finder(self, orig_node): #will determine if a node is two junction and also figure out the entire two junction path connected.
        two_list = [orig_node]
        prev_node = orig_node
        node = min(np.nonzero(self.edges[orig_node,:])[0])
        two_list.insert(len(two_list), node)
        while np.count_nonzero(self.edges[node,:]) == 2:
            if min(np.nonzero(self.edges[node,:])[0]) == prev_node:
                prev_node = node
                node = max(np.nonzero(self.edges[node,:])[0])
                two_list.insert(len(two_list), node)
            else:
                prev_node = node
                node = min(np.nonzero(self.edges[node,:])[0])
                two_list.insert(len(two_list), node)
        prev_node = orig_node
        node = max(np.nonzero(self.edges[orig_node,:])[0])
        two_list.insert(0, node)
        while np.count_nonzero(self.edges[node,:]) == 2:
            if max(np.nonzero(self.edges[node,:])[0]) == prev_node:
                prev_node = node
                node = min(np.nonzero(self.edges[node,:])[0])
                two_list.insert(0, node)
            else:
                prev_node = node
                node = max(np.nonzero(self.edges[node,:])[0])
                two_list.insert(0, node)
        two_list.reverse()
        return two_list
    def two_junc_lister(self): #will find all paths of two junction connections and return a list of tuples.
        check_list = range(len(self.points))
        save_list = []
        no_check_list = []
        for i in check_list:
            if self.two_junc_dect(i) == 1:
                if i not in no_check_list:
                    path = self.two_junc_list_finder(i)
                    if self.edges[path[0], path[-1]] == 0:
                        save_list.append(path)
                        for j in path:
                            no_check_list.append(j)
        return save_list
    def simp_prob_gen(self, step):
        edges_copy = np.zeros((len(self.points), len(self.points)))
        for i in range(len(self.points)):
            for j in range(len(self.points)):
                edges_copy[i,j] = self.edges[i,j]
        path_list = self.two_junc_lister()
        index_list = []
        for path in path_list:
            edge = 0
            for i in range(len(path) - 1):
                edge += self.edges[path[i], path[i+1]]
            edges_copy[path[0], path[-1]] = edge
            edges_copy[path[-1], path[0]] = edge
            index_list = index_list + path[1:][:-1]
        for i, j in enumerate(sorted(index_list)):
                edges_copy = np.delete(edges_copy, j - i, 0)
                edges_copy = np.delete(edges_copy, j - i, 1)
        #print(edges_copy)
        #return edges_copy, sorted(index_list)
        A = np.zeros((len(edges_copy), len(edges_copy)))
        for i in range(len(edges_copy)):
            for j in np.nonzero(edges_copy[i,:])[0]:
                A[i,j] = math.ceil(edges_copy[i,j]/step)*step
        p = np.zeros((len(edges_copy), len(edges_copy)))
        for init in range(len(edges_copy[:,0])):
            for i in np.nonzero(edges_copy[init,:])[0]:
                p[init, i] = (1/np.count_nonzero(A[init,:]))*(step/A[init, i])*(1/(1-((1/np.count_nonzero(A[init,:]))*np.sum(1-(step / (np.array([j for j in A[init,:] if j != 0])))))))
        return p, path_list, sorted(index_list), edges_copy
    def plotdet_edge_sm_two_junc(self, step, save_data_path): #provides a matrix where the i'th row will give the x and y co-ordinates of the midpoint of each edge, and the Ave_MFPT should that edge be removed. These are provided in each column respectively.
        P, path_list, ghost_list, edges_copy = self.simp_prob_gen(step)
        #print(edges_copy)
        D = np.zeros((len(P),len(P)))
        A = np.zeros((len(P),len(P), 5))
        all_inverses = np.zeros((len(P),len(P)-1,len(P)-1))
        new_points = np.zeros((len(P), 2))
        m = 0
        for i in range(len(self.points)):
            if i not in ghost_list:
                new_points[m,:] = self.points[i,:]
                m += 1
        target_checker = len(P) - 1
        for i in tqdm(range(len(P)), desc = "Inverting Matrices"):
            all_inverses[i,:,:] = self.simp_theory_stat([i], step, P)
        plt.figure(dpi = 600)
        for i in range(len(edges_copy)):
            for j in range(len(edges_copy)):
                if D[j,i] == 1: #This checks for connection
                    pass
                elif edges_copy[i,j] != 0: # Check for connection
                    edges_without_ab = np.zeros((len(edges_copy), len(edges_copy)))
                    for p in range(len(edges_copy)):
                        for q in range(len(edges_copy)):
                            edges_without_ab[p,q] = edges_copy[p,q]
                    edges_without_ab[i,j] = 0
                    edges_without_ab[j,i] = 0
                    if self.zero_degree_check(edges_without_ab) is True and self.edge_seperation_check_two_junc(step, all_inverses[target_checker,:,:], i, j, target_checker, edges_without_ab, edges_copy) == 1:
                        A[i,j,:] = [(new_points[i,0]+new_points[j,0])/2, (new_points[i,1]+new_points[j,1])/2,  self.MFPT_global_dream_SM_two_junc(all_inverses, i, j, step, edges_without_ab, edges_copy), i, j]
                        D[i,j] = 1    
                    else:
                        A[i,j,:] =  [(new_points[i,0]+new_points[j,0])/2, (new_points[i,1]+new_points[j,1])/2,  None, i, j]
                        D[i,j] = 1
        B = np.zeros((np.count_nonzero(D), 5))
        n = 0
        for i in range(len(edges_copy[:,0])):
            for j in range(len(edges_copy[:,0])):
                if D[i,j] == 1:
                    B[n,:] = A[i,j,:]
                    n += 1
        df = pd.DataFrame(B)
        if save_data_path is not None:
            df.to_excel(excel_writer = str(save_data_path), header = False, index = False)
        return B, all_inverses, ghost_list, new_points, path_list, edges_copy
    def MFPT_global_cheap_two_junc(self, step, all_inverses, new_points, edges_copy): #Named is honor of the professors dream, will return the MFPT of all nodes given no missing edges
        Ave_MFPT = np.zeros(len(edges_copy))
        occupancy = np.zeros((len(edges_copy)-1, len(edges_copy)))
        for i in range(len(edges_copy)):
            occupancy[:,i] = np.sum(all_inverses[i,:,:], axis = 0)
        for i in range(len(edges_copy)):
            n = 0
            for j in range(len(edges_copy)-1):
                    n += occupancy[j,i]/(len(edges_copy)-1)
            Ave_MFPT[i] += n
        return np.average(Ave_MFPT)
    def plotdet_edge_sm_two_junc_cheap(self, step, save_data_path): #provides a matrix where the i'th row will give the x and y co-ordinates of the midpoint of each edge, and the Ave_MFPT should that edge be removed. These are provided in each column respectively.
        P, path_list, ghost_list, edges_copy = self.simp_prob_gen(step)
        A = pd.read_excel(save_data_path, header=None)
        B = A.values[:,:]
        all_inverses = np.zeros((len(P),len(P)-1,len(P)-1))
        new_points = np.zeros((len(P), 2))
        m = 0
        for i in range(len(self.points)):
            if i not in ghost_list:
                new_points[m,:] = self.points[i,:]
                m += 1
        for i in tqdm(range(len(P)), desc = "Inverting Matrices"):
            all_inverses[i,:,:] = self.simp_theory_stat([i], step, P)
        plt.figure(dpi = 600)
        return B, all_inverses, ghost_list, new_points, path_list, edges_copy
    def MFPT_edge_removal_plot_two_junc(self, step, line_width, save_data_path, save_data, image_bounds, save_graph, save_graph_path, cmap, vmin, vmax):
        if save_data == True:
            A, I, ghost_list, new_points, path_list, edges_copy = self.plotdet_edge_sm_two_junc(step, save_data_path)
        else:
            A, I, ghost_list, new_points, path_list, edges_copy = self.plotdet_edge_sm_two_junc_cheap(step, save_data_path)
        index_mapping = []
        for i in range(len(self.points)):
            index_mapping.append(i)
        for i, val in enumerate(ghost_list):
            index_mapping[val] = -1
            for j in range(len(index_mapping))[val+1:]:
                index_mapping[j] -= 1
        #for i in ghost_list:
            #A = np.insert(A, i, [row], axis= 0)
        B = np.zeros((len(A), 5))
        for i in range(len(A)):
            for j in range(5):
                B[i,j] = A[i,j]
        for i in range(len(index_mapping)):
            for j in range(len(A)):
                if A[j,3] == index_mapping[i]:
                    B[j,3] = i
                if A[j,4] == index_mapping[i]:
                    B[j,4] = i
            index_mapping[i] = -1
        A = B
        n = 0
        for path in path_list:
            start = path[0]
            end = path[-1]
            for j in range(len(A)):
                i = j - n
                if A[i,3] == start and A[i,4] == end:
                    color = A[i,2]
                    A = np.delete(A, i, 0)
                    n += 1
                    break
                elif A[i,4] == start and A[i,3] == end:
                    color = A[i,2]
                    A = np.delete(A, i, 0)
                    n += 1
                    break
            for node in range(len(path[:-1])):
                A = np.r_['0,2', A, [0,0, color, path[node], path[node + 1]]]
        H = nx.Graph()
        C = 0
        n = 0
        D_check = 0
        #print(A)
        for i in range(len(A)):
            if np.isnan(A[i,2]) == True:
                n += 1
                if C == 0:
                    D = np.array([[A[i,3:], i]], dtype = object)
                    D_check = 1
                    C = 1
                else:
                    D = np.r_['0,2', D, [A[i,3:], i]]
        if D_check == 1:
            for n, i in enumerate(D[:,1]):
                A = np.delete(A, i - n, 0)
        
        C = 0
        for i, j in A[:,3:]:
            H.add_edge(i,j)
        MFPT_global = self.MFPT_global_cheap_two_junc(step, I, new_points, edges_copy)
        colors = A[:,2] - MFPT_global
        vmin = min(colors)
        vmax = max(colors)
        colors_corrected = np.zeros((len(A)))
        n = 0
        for u,v,a in H.edges(data=True):
            for i in range(len(A)):
                if A[i,3] == u and A[i,4] == v:
                    colors_corrected[n] = A[i,2] - MFPT_global
                    n += 1
                elif A[i,3] == v and A[i,4] == u:
                    colors_corrected[n] = A[i,2] - MFPT_global
                    n += 1
        #for n, i in enumerate(delete_list):
            #colors_corrected = np.delete(colors_corrected, i - n , 0)
        #print(H.nodes(data=True))
        plt.figure(figsize=(6, 6), dpi=1000)
        if self.im is None:
            pass
        else:           
            plt.rcParams["figure.figsize"] = [13.00,10.50]
            plt.rcParams["figure.autolayout"] = True
            im = plt.imread(self.im)
            fig, ax = plt.subplots(figsize=(8, 6), dpi = 200)
            im = ax.imshow(im, extent=image_bounds, aspect = 'auto')
        if D_check == 1:
            for i in range(len(D)):
                #print(int(D[i,0][0]), int(D[i,0][1]))
                connectpoints_r(self.points[:,0],self.points[:,1], int(D[i,0][0]), int(D[i,0][1]), line_width, 'Red')
        G = nx.relabel.convert_node_labels_to_integers(H, ordering = "default",  label_attribute='original_name')
        #print(G.nodes)
        posi = np.zeros((len(A),2))
        n = 0
        for i,j in H.nodes(data=True):
            posi[n,:] = self.points[int(i),:]
            n += 1
        #print(posi)
        nx.draw(G, posi, node_color='black', node_size=0, edge_color='Black', width=line_width, edge_cmap=cmap,
           with_labels=False, vmin=vmin, vmax=vmax)
        #print(self.points)
        #CL.ax.set_ticks(0,20,40,60,80,100)
        #CL.ax.set_yticklabels(['0','20','40','60', "80", ">100"])
        #plt.savefig(f'C:/Python_Programs/NMJ_MFPT_data/{self.name}_MFPT_edge.png')
        #plt.savefig(f'C:/Users/aelam/Documents/thesis_plots/{self.name}_Edge_ana.pdf', format='pdf')
        if save_graph:
            plt.savefig(save_graph_path + f"/{self.name}.png", format='png')
            plt.savefig(save_graph_path + f"/{self.name}.pdf", format='pdf')
        plt.show()
        plt.clf()
def connectpoints(x,y,p1,p2): #Will make a line between the two points in the next graph
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2], zorder = 1, c = 'white')# , 'k-'
def connectpoints_r(x,y,p1,p2, width, color): #Will make a line between the two points in the next graph
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2], color=color, linewidth=width)
def ER_network(name, diction, im_path, edge_path, node_path, image_bounds):
    #plt.figure(dpi=1000)
    #plt.rcParams["figure.figsize"] = [13.00,10.50]
    #plt.rcParams["figure.autolayout"] = True
    #im = plt.imread(f"C:/Python_Programs/Fai_ER_Project/NMJ_middle2/NMJ_middle2_im{order}.png")
    #im = plt.imread(str(im_path) + f"/S2Cell1460_im{order}.png")
    #fig, ax = plt.subplots()
    #im = ax.imshow(im, extent=image_bounds)
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
    #for i in range(len(real_points)):
        #for j in range(len(real_points)):
            #if edges[i,j] != 0:
                #connectpoints(real_points[:,0],real_points[:,1],i,j)
    #plt.axis('off')
    #plt.savefig(f'C:/Users/aelam/Documents/thesis_plots/NMJ_Network {order}_simp_net.eps', format='eps')
    #plt.scatter(real_points[:,0],real_points[:,1], zorder = 2, s = 140)
    #plt.title(f"ER_Image {order}")
    #plt.savefig(f'C:/Users/aelam/Documents/thesis_plots/NMJ_network{order}.eps', format='eps')
    #plt.savefig(f'C:/Users/aelam/Documents/thesis_plots/NMJ_network{order}.png', format='png')
    #plt.show()
    globals()[name] = Network(real_points, edges, None, name, image_path)
    #globals()[f"ER{order}"] = Network(real_points, edges, None, f"ER_Network {order}", str(im_path) + f"/S2Cell1460_im{order}.png")
    diction[name] = globals()[name]
    plt.rcdefaults()
    
network_dict = {"test" : "Test"}


ER_network(name, network_dict, image_path, edge_path, node_path, image_bounds)
#cmaps = [plt.cm.flag, plt.cm.prism, plt.cm.ocean, plt.cm.gist_earth, plt.cm.terrain, plt.cm.gist_stern, plt.cm.gnuplot, plt.cm.gnuplot2, plt.cm.CMRmap, plt.cm.cubehelix, plt.cm.brg, plt.cm.gist_rainbow, plt.cm.rainbow, plt.cm.jet, plt.cm.turbo, plt.cm.nipy_spectral, plt.cm.gist_ncar]
cmaps = [plt.cm.rainbow]
for cmap in cmaps:
    network_dict[name].MFPT_edge_removal_plot_two_junc(0.1, line_width, save_data_path, save_data, image_bounds, save_graph, save_graph_path, cmap, 0, 100)
    
