#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df = pd.read_csv('prepared_abuse_data.csv')


# In[3]:


df_graph = pd.read_csv('withFollowRetweet.csv')
#df_graph.rename(columns={'src':'source','tar':'target','reTweet_wt':'edge_wt'},inplace=True)


# In[4]:


node_wt = pd.read_csv('node_wt.csv')


# In[16]:


df


# In[5]:


df_graph


# In[ ]:


node_wt.to_csv('normal_nodes.csv',index=False)


# In[8]:


node_wt[node_wt['normal'] > 0].to_csv('normal_nodes.csv',index=False)


# In[15]:


node_wt[((node_wt['sexism'] > 0) | (node_wt['racism'] > 0))].to_csv('abuse_nodes.csv',index=False)


# In[51]:


df_graph[['source','target','weight']].to_csv('edge_data.csv',index=False)


# In[73]:


df_graph['weight'] = df_graph['weight'] / sum(df_graph['weight'])


# In[74]:


G = nx.from_pandas_edgelist(df_graph,source='source',target='target',edge_attr=['weight','user_info'])

uncoman = set(node_wt['user_id'].unique()) ^ set(G.nodes())

for i in uncoman:
    G.add_node(i)
    
nx.set_node_attributes(G, pd.Series(node_wt.racism, index=node_wt.user_id).to_dict(), 'racism')
nx.set_node_attributes(G, pd.Series(node_wt.sexism, index=node_wt.user_id).to_dict(), 'sexism')
nx.set_node_attributes(G, pd.Series(node_wt.normal, index=node_wt.user_id).to_dict(), 'normal')

G.remove_nodes_from(list(nx.isolates(G)))


# In[75]:


print(nx.info(G))


# In[30]:


number_of_triangles = sum(nx.triangles(G).values()) / 3


# In[31]:


sorted_components = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
no_components = len(sorted_components)
largest_cc = sorted_components[0]


# In[32]:


no_components


# In[35]:


no_components


# In[33]:


largest_cc


# In[15]:


# Clustering Coefficient

clust = nx.clustering(G)

# Clustering coefficient at specified nodes
#clust
# Avrage Clustering coefficient of nodes

np.average(list(clust.values()))


# In[16]:


cliq_max=  nx.find_cliques(G)

cliques=list(nx.find_cliques(G))

clsa=[]
# Calculating the length of maximum and minimum clique
for i in cliques:
    clsa.append(len(i))
print("Number of max clique-", clsa.count(max(clsa)))
print("Size of max clique-", max(clsa))
print("Number of min clique-", clsa.count(max(clsa)))
print("Size of min clique-", min(clsa))


# In[71]:



"""
Calculate Closeness Centrality for a set of given nodes
"""

def closeness_wt(g,normalize=True,src=None):
    node=[]
    closeness=[]

    if src:
        nodes = [src]
    else:
        nodes = g.nodes()
    for i in nodes:
        path_len = nx.single_source_shortest_path_length(g,i)
        i_paths = path_len.values()
        total = sum(i_paths)

        node.append(i)
        if total>0:
            c_centrality = (len(i_paths)-1) / total
            if normalize:
                norm = (len(i_paths)-1) / (len(g) - 1 )
                c_centrality = norm * c_centrality
            closeness.append(c_centrality)
        else:
            closeness.append(0)

    df_closeness=pd.DataFrame()
    df_closeness['node'] = node
    df_closeness['closeness'] = closeness
    
    return df_closeness

df_closeness = closeness_wt(G)


# In[69]:


node_wt['closeness'] = df_closeness['closeness']


# In[79]:


node_wt.sort_values(by='closeness',ascending=False).head(10)


# In[82]:


import seaborn as sns


# In[84]:


plt.figure(figsize=(6, 4), dpi=80)

sns.scatterplot(df_closeness['closeness'],df_closeness['node'])
x_sr=max(df_closeness['closeness'])
y_node = df_closeness[df_closeness['closeness']==x_sr]['node']
sns.scatterplot(x_sr,y_node)

plt.title('Node Vs Closeness Centrality')
plt.ylabel('Label of nodes')
plt.xlabel('Closeness Centrality')
plt.savefig('Closeness_Centrality.png')


# In[86]:


csr= nx.betweenness_centrality(G)
bt = pd.DataFrame()
bt['node']=list(csr.keys())
bt['score']=list(csr.values())

node_wt['btw_score'] = bt['score']


# In[91]:


node_wt.sort_values(by='btw_score',ascending=False).head(10)


# In[93]:


plt.figure(figsize=(6, 4), dpi=80)

sns.scatterplot(bt['score'],bt['node'])
x_sr=max(bt['score'])
y_node = bt[bt['score']==x_sr]['node']
sns.scatterplot(x_sr,y_node)

plt.title('Node Vs Betweenness Centrality')
plt.ylabel('Label of nodes')
plt.xlabel('Betweenness Centrality')
plt.savefig('Betweenness_Centrality.png')


# In[129]:


def ICM(g,S,p=0.5,mc=50):
    spread = []
    for i in range(mc):
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        while new_active:
            # Find neighbors of each newly active node, that become activated
            new_ones = []
            for node in new_active:
                # Determine neighbors that become infected
                np.random.seed(i)
                success = np.random.uniform(0,1,len(list(G.neighbors(node)))) < p
                new_ones += list(np.extract(success, list(g.neighbors(node))))
            new_active = list(set(new_ones) - set(A))
            
            # Add newly activated nodes to the set of activated nodes
            A += new_active 
        spread.append(len(A))
    return(np.mean(spread))

def greedy(g,k,p=0.1,mc=50):
    S, spread = [], []
    # To find k nodes with largest marginal gain
    for _ in range(k):
        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        for j in set(g.nodes())-set(S):
            s = ICM(g,S + [j],p,mc) # To get the spread
            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j
        # Add the selected node to the seed set
        S.append(node)
        # Add estimated spread
        spread.append(best_spread)
    return(S,spread)


# In[130]:


print('\nPlease wait it takes time for finding infuential nodes..')

print('\n Greedy algorithm with ICM:\n')

greedy_output = greedy(G,10,p = 0.1,mc = 10)
print(f"\n15 most infuential nodes: {greedy_output[0]}\n")
print(f"spread of 15 most infuential nodes: {greedy_output[1]}\n")


# In[55]:


plt.figure(num=None, figsize=(25, 20), dpi=None)

layout=nx.spring_layout(G)
#layout=nx.nx_pydot.graphviz_layout(G)

color_map=['green']
node_deg = nx.degree(G)
nx.draw_networkx(
    G,
    node_size=[int(deg[1])*20 for deg in node_deg],
    arrowsize=10,
    linewidths=1,
    #width = [i['weight']* for i in dict(G.edges).values()],
    pos=layout,
    edge_color='red',
    edgecolors='black',
    node_color=color_map,
    with_labels=False
    #font_size=35
    )


plt.tight_layout()
plt.savefig("graph.jpg", format='jpg')


# ## Girvan Newman model 

# In[76]:


from networkx.algorithms import community
from networkx import edge_betweenness_centrality as betweenness

def most_central_edge(G):
        centrality = betweenness(G, weight='weight')
        return max(centrality, key=centrality.get)
    
    
    
comp = community.girvan_newman(G, most_valuable_edge=most_central_edge)
top_level_communities = next(comp)
next_level_communities = next(comp)


# Modularity
# Modularity (community detection) is a measure of network structure. It was designed to measure the strength of division of a network into modules. Networks with high modularity have dense connections between the nodes within modules but sparse connections between nodes in different modules. Although a diversity of community detection algorithms have been proposed, the quality of community detection is usually measured by modularity and also some benchmark graphs.

# In[77]:


from networkx.algorithms import community
modularity = community.modularity(G,top_level_communities, weight='weight')
print ("Modularity = ", modularity)


# In[86]:


len(top_level_communities[0])


# In[79]:


print(nx.info(G))


# In[100]:


node_wt[node_wt['user_id']==33803412]


# In[96]:


top_level_communities[0]


# In[23]:


next_level_communities


# In[89]:


top_level_communities[1]


# In[80]:


color_map = []
for node in G:
    if node in top_level_communities[1]:
        color_map.append('green')
    else: 
        color_map.append('blue')  

# nx.draw(G, node_color=color_map, with_labels=True)
# plt.show()

plt.figure(num=None, figsize=(25, 20), dpi=None)

#layout=nx.spring_layout(G)
layout=nx.nx_pydot.graphviz_layout(G)

#color_map=['green']
node_deg = nx.degree(G)
nx.draw_networkx(
    G,
    node_size=[int(deg[1])*20 for deg in node_deg],
    arrowsize=10,
    linewidths=1,
    #width = [i['weight']* for i in dict(G.edges).values()],
    pos=layout,
    edge_color='red',
    edgecolors='black',
    node_color=color_map,
    with_labels=False
    #font_size=35
    )


plt.tight_layout()
plt.savefig("newman.jpg", format='jpg')


# In[ ]:





# In[55]:


import collections
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN


class WalkSCAN:

    def __init__(self, nb_steps=2, eps=0.1, min_samples=3):
        self.nb_steps = nb_steps
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_ = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def load(self, graph, init_vector):
        self.graph = graph.copy()
        self.init_vector = init_vector.copy()

    def embed_nodes(self):
        p = {0: self.init_vector.copy()}
        for t in range(self.nb_steps):
            p[t + 1] = collections.defaultdict(int)
            for v in p[t]:
                for (_, w, e_data) in self.graph.edges(v, data=True):
                    if 'weight' in e_data:
                        self.weighted_ = True
                        p[t + 1][w] += float(e_data['weight']) / float(self.graph.degree(v, weight='weight')) * p[t][v]
                    else:
                        self.weighted_ = False
                        p[t + 1][w] += 1.0 / float(self.graph.degree(v)) * p[t][v]
        self.embedded_value_ = dict()
        self.embedded_nodes_ = list()
        for v in p[self.nb_steps]:
            self.embedded_nodes_.append(v)
            self.embedded_value_[v] = np.array([p[t + 1][v] for t in range(self.nb_steps)])
        self.nb_embedded_nodes_ = len(self.embedded_nodes_)

    def find_cores(self):
        if self.nb_embedded_nodes_ > 0:
            P = np.zeros((self.nb_embedded_nodes_, self.nb_steps))
            for (i, node) in enumerate(self.embedded_nodes_):
                P[i, :] = self.embedded_value_[node]
            self.dbscan_.fit(P)
            self.cores_ = collections.defaultdict(set)
            self.outliers_ = set()
            for (i, node) in enumerate(self.embedded_nodes_):
                label = self.dbscan_.labels_[i]
                if label >= 0:
                    self.cores_[label].add(node)
                else:
                    self.outliers_.add(node)
        else:
            self.cores_ = {}
            self.outliers_ = set()

    def compute_core_average_value(self):
        self.core_average_value_ = dict()
        for (core_id, core) in self.cores_.items():
            self.core_average_value_[core_id] = np.zeros(self.nb_steps)
            for node in core:
                for t in range(self.nb_steps):
                    self.core_average_value_[core_id][t] += self.embedded_value_[node][t] / float(len(core))

    def sort_cores(self):
        self.sorted_core_ids_ = self.cores_.keys()
        self.sorted_core_ids_ = sorted(self.sorted_core_ids_, key=lambda i: list(self.core_average_value_[i]),
                                   reverse=True)
        self.sorted_cores_ = [self.cores_[i] for i in self.sorted_core_ids_]

    def aggregate_outliers(self):
        self.communities_ = list()
        for core in self.sorted_cores_:
            community = core.copy()
            for node in core:
                community |= set(nx.neighbors(self.graph, node)) & self.outliers_
            self.communities_.append(community)

    def detect_communities(self, graph, init_vector):
        self.load(graph, init_vector)
        self.embed_nodes()
        self.find_cores()
        self.compute_core_average_value()
        self.sort_cores()
        self.aggregate_outliers()


# In[128]:


# Create a WalkSCAN instance
ws = WalkSCAN(nb_steps=3, eps=0.05, min_samples=2)

# Initialization vector for the random walk
init_vector = {13857342: 0.5, 38397609: 0.5}

# Compute the communities
ws.detect_communities(G, init_vector)

# Print the best community
print(ws.communities_[0])


# In[635]:


color_map = []
for node in G:
    if node in ws.communities_[0]:
        color_map.append('green')
    else: 
        color_map.append('blue')  

# nx.draw(G, node_color=color_map, with_labels=True)
# plt.show()

plt.figure(num=None, figsize=(25, 20), dpi=None)

layout=nx.spring_layout(G)
#layout=nx.nx_pydot.graphviz_layout(G)

#color_map=['green']
node_deg = nx.degree(G)
nx.draw_networkx(
    G,
    node_size=[int(deg[1])*20 for deg in node_deg],
    arrowsize=10,
    linewidths=1,
    #width = [i['weight']* for i in dict(G.edges).values()],
    pos=layout,
    edge_color='red',
    edgecolors='black',
    node_color=color_map,
    with_labels=False
    #font_size=35
    )


plt.tight_layout()
plt.savefig("walkc.jpg", format='jpg')


# In[102]:


import community
louvain = community.best_partition(G, random_state=42,weight='weight')
clusters_count = len(set(louvain.values()))


# In[103]:


no_info_nodes = [node for node in dict(G.degree).keys() if G.degree[node] < 2]


# In[104]:


clusters_count


# In[106]:


louvain.values()


# In[184]:


def get_paired_color_palette(size):
    palette = []
    for i in range(size*2):
        palette.append(plt.cm.Paired(i))
    return palette

plt.figure(figsize=(30, 30))
clusters_count = len(set(louvain.values()))
light_colors = get_paired_color_palette(clusters_count)[0::2]
dark_colors = get_paired_color_palette(clusters_count)[1::2]

for i in set(louvain.values()):
    nodelist = [n for n in G.nodes if (louvain[n]==i) and (n not in no_info_nodes)]
    edgelist = [e for e in G.edges if ((louvain[e[0]]==i) or (louvain[e[1]]==i))]
    node_color = [light_colors[i] for _ in range(len(nodelist))]
    edge_color = [dark_colors[i] for _ in range(len(edgelist))]
    nx.draw_networkx_nodes(G, layout, nodelist=nodelist, node_color=node_color, edgecolors='k')                                                                                                           
    nx.draw_networkx_edges(G, layout, edgelist=edgelist, alpha=1/clusters_count, edge_color=edge_color)

plt.title('Graph with Louvain clustering', fontdict={'fontsize': 40})
plt.axis('off')
plt.show()


# In[561]:





# In[565]:


e_wt=[]
for i in G.edges():
    e_wt.append(G.get_edge_data(i[0], i[1])['weight'])


# In[ ]:





# In[573]:




wt, cnt = np.unique(e_wt, return_counts=True)

plt.figure(figsize=(10, 8))
plt.bar(wt, cnt, width=0.005, color="b")
plt.title("Edge weights histogram")
plt.ylabel("Count")
plt.xlabel("edge weights")
plt.xticks(np.linspace(0, 1, 10))

plt.show()


# In[575]:


from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph


# In[578]:


tr = StellarGraph(G)


# In[579]:


rw = BiasedRandomWalk(tr)


# In[582]:


walk_length = 100  # maximum length of a random walk to use throughout this notebook
weighted_walks = rw.run(
    nodes=G.nodes(),  # root nodes
    length=walk_length,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
    weighted=True,  # for weighted random walks
    seed=42,  # random seed fixed for reproducibility
)
print("Number of random walks: {}".format(len(weighted_walks)))


# In[589]:


from gensim.models import Word2Vec

weighted_model = Word2Vec(
    weighted_walks
)


# In[593]:


# Retrieve node embeddings and corresponding subjects
node_ids = weighted_model.wv.index_to_key  # list of node IDs
weighted_node_embeddings = (
    weighted_model.wv.vectors
)  # numpy.ndarray of size number of nodes times embeddings dimensionality
# the gensim ordering may not match the StellarGraph one, so rearrange
node_targets = l_


# In[594]:


# Apply t-SNE transformation on node embeddings
tsne = TSNE(n_components=2, random_state=42)
weighted_node_embeddings_2d = tsne.fit_transform(weighted_node_embeddings)


# In[600]:


# draw the points
alpha = 0.7

plt.figure(figsize=(10, 8))
plt.scatter(
    weighted_node_embeddings_2d[:, 0],
    weighted_node_embeddings_2d[:, 1],
    c=node_targets[:-1],
    cmap="jet",
    alpha=0.3,
)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[158]:


from node2vec import Node2Vec
# Generate walks
node2vec = Node2Vec(G, dimensions=128, walk_length=16, num_walks=100, workers=2)


# In[160]:


model = node2vec.fit(window=5, min_count=1)


# In[165]:


# Retrieve node embeddings and corresponding subjects
node_ids = model.wv.index_to_key  # list of node IDs
node_embeddings = (
    model.wv.vectors
)


# In[499]:


from sklearn.manifold import TSNE

# Apply t-SNE transformation on node embeddings
tsne = TSNE(n_components=2)
node_embeddings_2d = tsne.fit_transform(node_embeddings)


# In[550]:


conn_nodes['label'] = conn_nodes['abuse'].apply(lambda x: 1 if x>min(conn_nodes['abuse']) else 0)


# In[551]:


l_=[]
for i in node_ids:
    l_.append(list(conn_nodes[conn_nodes['user_id']==int(i)]['label'])[0])


# In[603]:


# draw the points
alpha = 0.4
label_map = {l: i for i, l in enumerate(np.unique(l_))}
node_colours = [label_map[target] for target in l_]

plt.figure(figsize=(10, 8))
plt.scatter(
    node_embeddings_2d[:, 0],
    node_embeddings_2d[:, 1],
    c=node_colours,
    #cmap=['blue','red'],
    alpha=alpha,
)
plt.legend(l_)


# In[461]:


df_n2v = pd.DataFrame(node_embeddings)
df_n2v['uesr_ids'] = node_ids

df_n2v


# In[462]:


from sklearn.cluster import KMeans

Kmean = KMeans(n_clusters=2,random_state=42)
Kmean.fit(df_n2v.iloc[:,:-1])


# In[463]:


df_n2v['K_means'] = Kmean.labels_


# In[532]:


df_n2v


# In[ ]:





# In[636]:


color_map = []
for node in G:
    label = df_n2v[df_n2v['uesr_ids']==str(node)]['K_means'].values
    if (label == 1):
        color_map.append('green')
    else:
        color_map.append('blue') 

# nx.draw(G, node_color=color_map, with_labels=True)
# plt.show()

plt.figure(num=None, figsize=(25, 20), dpi=None)

layout=nx.spring_layout(G)
#layout=nx.nx_pydot.graphviz_layout(G)

#color_map=['green']
node_deg = nx.degree(G)
nx.draw_networkx(
    G,
    node_size=[int(deg[1])*20 for deg in node_deg],
    arrowsize=10,
    linewidths=1,
    #width = [i['weight']* for i in dict(G.edges).values()],
    pos=layout,
    edge_color='red',
    edgecolors='black',
    node_color=color_map,
#     cmap=plt.cm.RdYlBu, 
#     node_color=Kmean.labels_,
    node_layout_kwargs=dict(node_to_community=node_to_community),
    with_labels=False
    #font_size=35
    )


plt.tight_layout()
plt.savefig("kmeans.jpg", format='jpg')


# In[457]:


df_n2v['uesr_ids'] = df_n2v['uesr_ids'].apply(lambda x: int(x))
k_mens_c = [(set(df_n2v[df_n2v['K_means']==1]['uesr_ids']))]
k_mens_c.append(set(df_n2v[df_n2v['K_means']==0]['uesr_ids']))
k_mens_c = tuple(k_mens_c)


# In[403]:


from networkx.algorithms import community
modularity = community.modularity(G,top_level_communities, weight='weight')
print ("Modularity = ", modularity)


# In[404]:


modularity = community.modularity(G,k_mens_c, weight='weight')
print ("Modularity = ", modularity)


# In[405]:


conn_nodes = node_wt[node_wt['user_id'].apply(lambda x: True if x in list(df_n2v['uesr_ids']) else False)]
conn_nodes.sort_values(by='user_id',inplace=True)
conn_nodes['comm'] = list(df_n2v.sort_values(by='uesr_ids')['K_means'])

conn_nodes['abuse'] = conn_nodes['racism'] + conn_nodes['sexism']


# In[406]:


import seaborn as sns


# In[407]:


comm_1 = conn_nodes[conn_nodes['comm']==1]
comm_0 = conn_nodes[conn_nodes['comm']==0]

print("Abuse max and min weight of community 0:",max(comm_0['abuse']), min(comm_0['abuse']))
print("Normal max and min weight of community 0:",max(comm_0['normal']), min(comm_0['normal']))

print("\nAbuse max and min weight of community 1:",max(comm_1['abuse']), min(comm_1['abuse']))
print("Normal max and min weight of community 1:",max(comm_1['normal']), min(comm_1['normal']))


# In[426]:


df_n2v


# In[497]:


Kmean.get_params()


# In[ ]:





# In[ ]:




