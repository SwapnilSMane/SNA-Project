#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv


# In[2]:


df = pd.read_csv('upto_userName.csv')


# In[3]:


len(df['tweet user id'].unique())


# In[4]:



file = open('upto_retweets2.csv')
csvreader = csv.reader(file)
l_rows=[]
for row in csvreader:
        l_rows.append(','.join(row[2:]))


# In[163]:


df[df.isna().any(axis=1)]


# In[ ]:





# In[6]:


df['retweet_user'] = pd.DataFrame(l_rows)

df['tweet user id'] = df['tweet user id'].replace(np.nan,0)
df['tweet user id'] = df['tweet user id'].apply(lambda x: int(x))

df.dropna(inplace=True)
#df['retweet_user'] = df['retweet_user'].apply(lambda x: list(map(int,eval(x))))


# In[233]:


df#.head(10)


# In[8]:


users = df['tweet user id'].unique()
rel=[]
for i in range(len(df)):
    try:
        j = list(map(int,eval(df['retweet_user'][i])))
        s = df['tweet user id'][i]
        for k in j:
            if k in users:
                rel.append([s,k])
    except:
        pass


# In[9]:


retweet_relation = pd.DataFrame(rel,columns=['src','tar'])


# In[10]:


#retweet_relation.to_csv('retweet_relation.csv',index=False)


# In[11]:


df.to_csv('prepared_abuse_data.csv',index=False)


# In[81]:


# count/total retweets
wt = retweet_relation.value_counts()/sum(retweet_relation.value_counts())
rel_wt = pd.DataFrame(list(wt.index),columns=['source','target'])
rel_wt['retweet_weight'] = wt.values


# In[82]:


rel_wt.to_csv('retweet_relation.csv',index=False)


# In[225]:


rel_wt


# In[229]:


rel_wt.iloc[:,0]


# In[226]:


for i in range(len(df)):
    if df.iloc[i,1] != 2:
        


# In[ ]:





# In[234]:


df[df['annotation']==2]


# In[ ]:


5799


# ## Evaluate friendship relation

# In[15]:


file = open('upto_follow.csv')
csvreader = csv.reader(file)
l_rows=[]
users=[]
for row in csvreader:
    try:
        users.append(row[1])
        l_rows.append(','.join(row[2:]))
    except:
        pass
    
    
users.pop(641)
l_rows.pop(641)

users = [eval(i) for i in users]


# In[16]:


u = list(df['tweet user id'])
pair=[]
for i in range(len(users)):
    try:
        j = eval(l_rows[i])
        join = set(j) & set(u)
        if join:
            for k in join:
                pair.append([int(users[i]),int(k)])
    except:
        pass


# In[47]:


df_follow = pd.DataFrame(pair,columns=['source','target'])


# In[48]:


df_follow['weight_follow'] = np.ones([1,5883])[0]


# In[49]:


df_follow


# In[50]:


rel_wt


# In[147]:


df3 = pd.merge(df_follow.iloc[:,:2], rel_wt.iloc[:,:2], how='outer',)


# In[148]:


df3


# In[85]:


temp = pd.merge(df_follow,rel_wt,how='outer')


# In[86]:


temp.replace(to_replace=np.nan,value=0,inplace=True)


# In[78]:


# for cheching
# temp[(temp['source']==14473703) & (temp['target']==21636104)]


# In[96]:


temp['weight'] = temp['weight_follow'] + temp['retweet_weight']
temp['user_info'] = temp['weight'].apply(lambda x: 'both' if x>1 else 'follow' if x==1 else 'retweet')

temp.to_csv('withFollowRetweet.csv',index=False)


# In[97]:


temp[(temp['source']==14473703) & (temp['target']==21636104)]


# In[138]:





# In[134]:





# In[210]:


l_0 = df[df['annotation']==0]['tweet user id'].value_counts() / len(df)
l_0 = l_0.rename('racism')
l_1 = df[df['annotation']==1]['tweet user id'].value_counts() / len(df)
l_1 = l_1.rename('sexism')
l_2 = df[df['annotation']==2]['tweet user id'].value_counts() / len(df)
l_2 = l_2.rename('normal')


# In[223]:


n_wt = pd.concat([l_0,l_1,l_2],axis=1)
n_wt.replace(to_replace=np.nan,value=0,inplace=True)
n_wt['user_id']=n_wt.index
n_wt.to_csv('node_wt.csv',index=False)


# In[113]:


from sklearn.preprocessing import MinMaxScaler
# define standard scaler
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(np.array(temp['weight']).reshape(-1,1))


# In[121]:


temp['weight'] / max(temp['weight'])


# In[ ]:




