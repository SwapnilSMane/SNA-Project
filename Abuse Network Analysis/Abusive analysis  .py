#!/usr/bin/env python
# coding: utf-8

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


abuse_stat = pd.read_csv('abuse_graph_data.csv')
normal_stat = pd.read_csv('normal_graph_data.csv')

abuse_stat = abuse_stat[abuse_stat['degree']!=0]
normal_stat = normal_stat[normal_stat['degree']!=0]


# In[17]:


normal_stat


# In[41]:



print('Abuse Community Average Degree:',np.average(abuse_stat['degree']))
print('Normal Community Average Degree:',np.average(normal_stat['degree']))
plt.figure(figsize=(10,6)) 
plt.loglog(abuse_stat['degree'],'ro') 
plt.loglog(normal_stat['degree'],'go') 
plt.suptitle('Degree Distribution', fontsize=14,fontweight="bold")
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.legend(['Abuse','Normal'])
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[40]:


print('Abuse Community Average Weighted Degree:',np.average(abuse_stat['weighted_degree']))
print('Normal Community Average Weighted Degree:',np.average(normal_stat['weighted_degree']))
plt.figure(figsize=(10,6)) 
plt.loglog(abuse_stat['weighted_degree'],'ro') 
plt.loglog(normal_stat['weighted_degree'],'go') 
plt.suptitle('Weighted Degree Distribution', fontsize=14,fontweight="bold")
plt.xlabel('Weighted Degree')
plt.ylabel('Frequency')
plt.legend(['Abuse','Normal'])
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[39]:


plt.figure(figsize=(10,6)) 
plt.loglog(abuse_stat['betweenness_centrality'],'ro') 
plt.loglog(normal_stat['betweenness_centrality'],'go') 
plt.suptitle('Betweeness Centrality Distribution', fontsize=14,fontweight="bold")
plt.xlabel('Betweeness Centrality')
plt.ylabel('Frequency')
plt.legend(['Abuse','Normal'])
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[8]:


plt.figure(figsize=(8,6)) 
sns.distplot(abuse_stat['closeness_centrality'],bins=50) 
sns.distplot(normal_stat['closeness_centrality'],bins=50) 
plt.suptitle('Closness Centrality Distribution', fontsize=14,fontweight="bold")
plt.xlabel('Closness Centrality')
plt.ylabel('Frequency')
plt.legend(['Abuse','Normal'])
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[9]:


print('Abuse Community Density:',0.023)
print('Normal Community Density:',0.013)


# In[23]:


plt.figure(figsize=(8,6)) 
sns.countplot(abuse_stat['community'],palette="ch:.25") 

plt.suptitle('Abuse Nodes distribution over best partition community Distribution', fontsize=14,fontweight="bold")
plt.xlabel('community Class')
plt.ylabel('Number of nodes')
#plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[22]:


plt.figure(figsize=(8,6)) 
sns.countplot(normal_stat['community'],palette="ch:.25") 
plt.suptitle('Normal author distribution over best partition community Distribution', fontsize=14,fontweight="bold")
plt.xlabel('community Class')
plt.ylabel('Number of nodes')
#plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[25]:


plt.figure(figsize=(8,6)) 
sns.countplot(abuse_stat['greedy_modularity_communities'],palette="ch:.25") 

plt.suptitle('Abuse Nodes distribution over Greedy Modularity communities Distribution', fontsize=14,fontweight="bold")
plt.xlabel('Modularity Class')
plt.ylabel('Number of nodes')
#plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[24]:


plt.figure(figsize=(8,6)) 
sns.countplot(normal_stat['greedy_modularity_communities'],palette="ch:.25") 
plt.suptitle('Normal author distribution over Greedy Modularity communities Distribution', fontsize=14,fontweight="bold")
plt.xlabel('Modularity Class')
plt.ylabel('Number of nodes')
#plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[38]:


print("Abuse Avg. Clustering Coefficient:",np.average(abuse_stat['clustering']))
print("Normal Avg. Clustering Coefficient:",np.average(normal_stat['clustering']))


plt.figure(figsize=(10,6)) 
plt.loglog(abuse_stat['clustering'],'ro') 
plt.loglog(normal_stat['clustering'],'go') 
plt.suptitle('Clustering Coefficient Distribution', fontsize=14,fontweight="bold")
plt.xlabel('Clustering Coefficient')
plt.ylabel('Count')
plt.legend(['Abuse','Normal'])
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[37]:


plt.figure(figsize=(10,6)) 
plt.loglog(abuse_stat['eigenvector_centrality'],'ro') 
plt.loglog(normal_stat['eigenvector_centrality'],'go') 
plt.suptitle('Eigenvector Centrality Distribution', fontsize=14,fontweight="bold")
plt.xlabel('Eigenvector Centrality')
plt.ylabel('Count')
plt.legend(['Abuse','Normal'])
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[36]:


# HITS 
# E = 1.0E-4

plt.figure(figsize=(10,6)) 
plt.loglog(abuse_stat['hubs'],'ro') 
plt.loglog(normal_stat['hubs'],'go') 
plt.suptitle('Hubs Distribution', fontsize=14,fontweight="bold")
plt.xlabel('Hubs')
plt.ylabel('Count')
plt.legend(['Abuse','Normal'])
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




