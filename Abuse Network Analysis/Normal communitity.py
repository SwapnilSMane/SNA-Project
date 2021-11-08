#!/usr/bin/env python
# coding: utf-8

# In[68]:


# https://programminghistorian.org/en/lessons/exploring-and-analyzing-network-data-with-python


# In[1]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import community
import seaborn as sns


# In[2]:


df_graph = pd.read_csv('withFollowRetweet.csv')


# In[3]:


normal_nodes = pd.read_csv('normal_nodes.csv')
abuse_nodes = pd.read_csv('abuse_nodes.csv')

normal_nodes['abuse'] = normal_nodes['racism'] + normal_nodes['sexism']
abuse_nodes['abuse'] = abuse_nodes['racism'] + abuse_nodes['sexism']


# In[4]:


normal_data = df_graph[df_graph['source'].isin(list(normal_nodes['user_id'])) & df_graph['target'].isin(list(normal_nodes['user_id']))]
normal_data.to_csv('normal_data.csv',index=False)


# In[5]:


G_normal = nx.from_pandas_edgelist(normal_data,source='source',target='target',edge_attr=['weight'])

uncoman = set(normal_nodes['user_id'].unique()) ^ set(G_normal.nodes())

for i in uncoman:
    G_normal.add_node(i)
    
nx.set_node_attributes(G_normal, pd.Series(normal_nodes.racism, index=normal_nodes.user_id).to_dict(), 'racism')
nx.set_node_attributes(G_normal, pd.Series(normal_nodes.sexism, index=normal_nodes.user_id).to_dict(), 'sexism')
nx.set_node_attributes(G_normal, pd.Series(normal_nodes.normal, index=normal_nodes.user_id).to_dict(), 'normal')

G_normal.remove_nodes_from(list(nx.isolates(G_normal)))

plt.figure(num=None, figsize=(25, 20), dpi=None)

#layout=nx.spring_layout(G_normal)
layout=nx.nx_pydot.graphviz_layout(G_normal)

color_map=['black']
node_deg = nx.degree(G_normal)
nx.draw_networkx(
    G_normal,
    node_size=[int(deg[1])*15 for deg in node_deg],
    arrowsize=5,
    linewidths=0.8,
    #width = [i['weight']* for i in dict(G.edges).values()],
    pos=layout,
    edge_color='gray',
    edgecolors='black',
    node_color=color_map,
    with_labels=False
    #font_size=35
    )


plt.tight_layout()
plt.savefig("normal_nw.jpg", format='jpg')


# In[159]:


print(nx.info(G_normal))


# In[156]:


deg=[]
for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        deg.append(G_normal.degree(i))
    else:
        deg.append(0)   
        
normal_nodes['degree'] = deg

print("Avg. degree:",np.average(deg))


# In[157]:


wt_deg=[]
for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        wt_deg.append(G_normal.degree(i,weight='weight'))
    else:
        wt_deg.append(0)   
        
normal_nodes['weighted_degree'] = wt_deg

print("Avg. weighted degree:",np.average(wt_deg))


# In[81]:


closeness_centrality = nx.closeness_centrality(G_normal)

closeness=[]
for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        closeness.append(closeness_centrality[i])
    else:
        closeness.append(0)   
        
normal_nodes['closeness_centrality'] = closeness


# In[82]:


betweenness_centrality = nx.betweenness_centrality(G_normal)

betweenness=[]
for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        betweenness.append(betweenness_centrality[i])
    else:
        betweenness.append(0)   
        
normal_nodes['betweenness_centrality'] = betweenness


# In[83]:


import community

part = community.best_partition(G_normal)


# In[84]:


community_l=[]
for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        community_l.append(part[i])
    else:
        community_l.append(0)   
        
normal_nodes['community'] = community_l


# In[86]:


eigenvector_centrality = nx.eigenvector_centrality(G_normal)

eigenvector=[]
for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        eigenvector.append(eigenvector_centrality[i])
    else:
        eigenvector.append(0)   
        
normal_nodes['eigenvector_centrality'] = eigenvector


# In[87]:


hubs, authorities = nx.hits(G_normal, max_iter = 1000, normalized = True) 

hubs_l=[]
authorities_l = []

for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        hubs_l.append(hubs[i])
        authorities_l.append(authorities[i])
    else:
        hubs_l.append(0)
        authorities_l.append(0)   
        
normal_nodes['hubs'] = hubs_l
normal_nodes['authorities'] = authorities_l


# In[110]:


# modularity 
    
mod = community.modularity(part,G_normal)
mod


# In[154]:


# connected components

# If your Graph has more than one component, this will return False:
print(nx.is_connected(G_normal))

# Next, use nx.connected_components to get the list of components,
# then use the max() command to find the largest one:
components = nx.connected_components(G_normal)
largest_component = max(components, key=len)

# Create a "subgraph" of just the largest component
# Then calculate the diameter of the subgraph, just like you did with density.
#

subgraph = G_normal.subgraph(largest_component)
diameter = nx.diameter(subgraph)
radius = nx.radius(subgraph)
avg_path = nx.average_shortest_path_length(subgraph)

print("Network diameter of largest component:", diameter)
print("Network radius of largest component:", radius)
print("Network average shortest path length of largest component:", avg_path)


print("Components:")
[len(c) for c in sorted(nx.connected_components(G_normal), key=len, reverse=True)]


# In[155]:


# No. triangles
number_of_triangles = sum(nx.triangles(G_normal).values()) / 3

print("number of triangles:", number_of_triangles)


# In[111]:


# density

density = nx.density(G_normal)
print("Network density:", density)


# In[114]:


# triadic closure

triadic_closure = nx.transitivity(G_normal)
print("Triadic closure:", triadic_closure)


# In[91]:


# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html4

clustering = nx.clustering(G_normal)

clustering_l=[]
for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        clustering_l.append(clustering[i])
    else:
        clustering_l.append(0)   
        
normal_nodes['clustering'] = clustering_l


# In[96]:


pr = nx.pagerank(G_normal, alpha=0.9)

pr_l=[]
for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        pr_l.append(pr[i])
    else:
        pr_l.append(0)   
        
normal_nodes['pageRank'] = pr_l


# In[99]:


triangles = nx.triangles(G_normal)

triangles_l=[]
for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        triangles_l.append(triangles[i])
    else:
        triangles_l.append(0)   
        
normal_nodes['triangles'] = triangles_l


# In[127]:


from networkx.algorithms import community
communities = community.greedy_modularity_communities(G_normal)

modularity_dict = {} # Create a blank dictionary
for i,c in enumerate(communities): # Loop through the list of communities, keeping track of the number for the community
    for name in c: # Loop through each person in a community
        modularity_dict[name] = i # Create an entry in the dictionary for the person, where the value is which group they belong to.


# In[129]:


modularity_l=[]
for i in normal_nodes['user_id']:
    if i in G_normal.nodes():
        modularity_l.append(modularity_dict[i])
    else:
        modularity_l.append(0)   
        
normal_nodes['greedy_modularity_communities'] = modularity_l


# In[130]:


normal_nodes


# In[139]:


cliques = nx.find_cliques(G_normal)


# In[140]:


cliques4 = [clq for clq in cliques if len(clq) >= 1]


# In[141]:


cliques4


# In[158]:


normal_nodes.to_csv("normal_graph_data.csv",index=False)


# In[ ]:




