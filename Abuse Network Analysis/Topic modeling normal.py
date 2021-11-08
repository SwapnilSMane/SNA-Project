#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk

import pandas as pd
nltk.download('wordnet')


# In[2]:


df = pd.read_csv('prepared_abuse_data.csv')
df = df[df['annotation']==2]


# In[3]:


df


# In[4]:


import preprocessor as p
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[5]:


text = df['tweet text'].apply(lambda x: p.clean(x))


# In[6]:


stop = set(stopwords.words('english'))
exclude = set(punctuation)
lemma = WordNetLemmatizer()

def clean_data(data):
    stop_free = ' '.join([i for i in data.lower().split() if i not in stop])
    punct_free = ' '.join([ch for ch in stop_free.split() if ch not in exclude])
    normalized = " ".join(lemma.lemmatize(word) for word in punct_free.split())
    return normalized


# In[7]:


data = [clean_data(line) for line in text]
# data


# In[8]:


cv = CountVectorizer(stop_words='english',min_df=2,max_df=0.95)
dtm = cv.fit_transform(data)


# In[9]:


LDA = LatentDirichletAllocation(n_components=2,random_state=42)
LDA.fit(dtm)


# In[10]:


single_topic = LDA.components_[0].argsort()
for i in single_topic[-10:]:
    print(cv.get_feature_names()[i])


# In[11]:


for index,topic in enumerate(LDA.components_):
    print(f'Top 15 words of Topic {index}')
    print([cv.get_feature_names()[idx] for idx in topic.argsort()[-10:]])
    print('\n')


# In[12]:


word_dict = {}
for index,topic in enumerate(LDA.components_):
    word = [cv.get_feature_names()[idx] for idx in topic.argsort()[-10:]]
    word_dict[f'topic {index}'] = word
pd.DataFrame(word_dict)


# In[13]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[14]:


word_dict = {}

for index,topic in enumerate(LDA.components_):
    name = []
    word = [cv.get_feature_names()[idx] for idx in topic.argsort()[-30:]]
    for ent in nlp(u' '.join(word)).noun_chunks:
        #print(str(spacy.explain(ent.label_)))
        #print('\n')
        print(ent.text)
        #name.append(str(spacy.explain(ent.label_)))
        name.append(ent.text)
    print('\n')
    word_dict[f'{name[0]}'] = word
named_topic = pd.DataFrame(word_dict)


# In[15]:


name


# In[16]:


named_topic.head()


# In[17]:


topic_one =nlp(u' '.join([(cv.get_feature_names()[idx]) for idx in single_topic[-30:]]))


# In[18]:


topic_one


# In[19]:


doc = nlp(u'swap')
for ent in nlp(u' '.join(cv.get_feature_names()[-10:])).ents:
    print(str(spacy.explain(ent.label_)) )


# In[20]:


from spacy import displacy


# In[21]:


displacy.render(topic_one,style='ent',jupyter=True)


# In[22]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


l ="he bought 2 pencils, 3 erasers, and 1 pencil-sharpener."

len('alice')


# In[24]:


plt.figure(figsize=(10,6)) 
topic_words = ' '.join([cv.get_feature_names()[i] for i in topic.argsort()[-150:]])

word_cloud = WordCloud(background_color='white',
                           max_words=100,
                           max_font_size=30,
                           scale=5,
                           random_state=1).generate(topic_words)

plt.imshow(word_cloud,interpolation="bilinear")
plt.tight_layout()
plt.axis("off")
plt.show()


# In[25]:


ncol=2
n_clust =2
nrows=ceil(n_clust/ncol)
fig, axeslist = plt.subplots(ncols=ncol,nrows=nrows, figsize=(20,20))
for index,topic in enumerate(LDA.components_):
    name = []
    topic_words = ' '.join([cv.get_feature_names()[i] for i in topic.argsort()[-30:]])
    word_cloud = WordCloud(background_color='white',
                           max_words=100,
                           max_font_size=30,
                           scale=3,
                           random_state=1).generate(topic_words)
    for ent in nlp(u' '.join([cv.get_feature_names()[i] for i in topic.argsort()[-30:]])).noun_chunks:
        name.append(ent.text)
    axeslist.ravel()[index].imshow(word_cloud)
    axeslist.ravel()[index].set_title(f'Topic of {name[0]}',fontsize=20)
    axeslist.ravel()[index].set_axis_off()
        
plt.tight_layout()
plt.show()


# In[38]:


data = [clean_data(line) for line in text]
# data
data


# In[39]:


text


# In[ ]:





# In[40]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

df['neg'] = text.apply(lambda x: sid.polarity_scores(x)['neg'])
df['pos'] = text.apply(lambda x: sid.polarity_scores(x)['pos'])
df['neu'] = text.apply(lambda x: sid.polarity_scores(x)['neu'])


# In[41]:


tot = sum([sum(df['neg']), sum(df['pos'])])


# In[42]:


df


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns

#define data
plt.figure(figsize=(12,10)) 
data = [sum(df['pos'])/tot * 100, sum(df['neg'])/tot * 100]
labels = ['Pos', 'Neg']

#define Seaborn color palette to use
colors = sns.color_palette('bright')[0:5]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.show()


# In[ ]:





# In[31]:


#Import the modules
import text2emotion as te

#Call to the function
te.get_emotion(df['tweet text'][0])


# In[44]:


temp_emo = text.apply(lambda x:te.get_emotion(x))


# In[45]:


emotion = pd.DataFrame()
emotion['Happy'] = list(temp_emo.apply(lambda x:x['Happy']))
emotion['Angry'] = list(temp_emo.apply(lambda x:x['Angry']))
emotion['Surprise'] = list(temp_emo.apply(lambda x:x['Surprise']))
emotion['Sad'] = list(temp_emo.apply(lambda x:x['Sad']))
emotion['Fear'] = list(temp_emo.apply(lambda x:x['Fear']))

tot = sum([sum(emotion['Happy']), sum(emotion['Angry']), sum(emotion['Surprise']), sum(emotion['Sad']), sum(emotion['Fear'])])


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns

#define data
plt.figure(figsize=(12,10)) 
data = [sum(emotion['Happy'])/tot * 100, sum(emotion['Angry'])/tot * 100,
        sum(emotion['Surprise'])/tot * 100, sum(emotion['Sad'])/tot * 100,
        sum(emotion['Fear'])/tot * 100]
labels = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']

#define Seaborn color palette to use
colors = sns.color_palette('bright')[0:5]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.show()


# In[ ]:





# In[ ]:




