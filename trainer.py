
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling

# In[2]:


dataset=pd.read_csv('data.csv')
profile=dataset.drop("Unnamed_0",axis=1).profile_report(title='EDA of Training Data')
profile.to_file(output_file="eda_data_train.html")

# In[3]:


for i in range(dataset.shape[0]):
    dataset.at[i,"shot_id_number"]=dataset.at[i,"Unnamed: 0"]+1


# In[4]:


dataset.drop("Unnamed: 0",inplace=True,axis=1)


# In[5]:


dataset.drop("team_id",inplace=True,axis=1)


# In[6]:


dataset.drop("team_name",inplace=True,axis=1)


# In[7]:


dataset.drop("match_event_id",inplace=True,axis=1)


# In[8]:


dataset.drop("match_id",inplace=True,axis=1)


# In[9]:


dataset.drop("date_of_game",inplace=True,axis=1)


# In[10]:


dataset["lat1"]=dataset["lat/lng"]


# In[11]:


dataset["lng1"]=dataset["lat/lng"]


# In[12]:


q=list(dataset[dataset["lat/lng"].isnull()==False].index)


# In[13]:


for i in q:
    dataset.at[i,"lat1"]=dataset.at[i,"lat/lng"].split(",")[0]
    dataset.at[i,"lat1"]=float(dataset.at[i,"lat1"])
    dataset.at[i,"lng1"]=dataset.at[i,"lat/lng"].split(",")[1]
    dataset.at[i,"lng1"]=float(dataset.at[i,"lng1"])
    


# In[14]:


dataset["lat1"]=dataset["lat1"].astype(float)


# In[15]:


dataset["lng1"]=dataset["lng1"].astype(float)


# In[16]:


dataset.drop("lat/lng",inplace=True,axis=1)


# In[17]:


dataset.drop("lat1",inplace=True,axis=1)
dataset.drop("lng1",inplace=True,axis=1)


# In[18]:


md=dataset.groupby('game_season').is_goal.agg(['mean'])
dataset=dataset.reset_index().merge(md,how="left",on="game_season").set_index('index')
dataset = dataset.rename(columns={'mean': 'Avg_goal_game_season'})


# In[19]:


md=dataset.groupby('area_of_shot').is_goal.agg(['mean'])
dataset=dataset.reset_index().merge(md,how="left",on="area_of_shot").set_index('index')
dataset = dataset.rename(columns={'mean': 'Avg_goal_area_of_shot'})


# In[20]:


md=dataset.groupby('shot_basics').is_goal.agg(['mean'])
dataset=dataset.reset_index().merge(md,how="left",on="shot_basics").set_index('index')
dataset = dataset.rename(columns={'mean': 'Avg_goal_shot_basics'})


# In[21]:


md=dataset.groupby('range_of_shot').is_goal.agg(['mean'])
dataset=dataset.reset_index().merge(md,how="left",on="range_of_shot").set_index('index')
dataset = dataset.rename(columns={'mean': 'Avg_goal_range_of_shot'})


# In[22]:


md=dataset.groupby('home/away').is_goal.agg(['mean'])
dataset=dataset.reset_index().merge(md,how="left",on="home/away").set_index('index')
dataset = dataset.rename(columns={'mean': 'Avg_goal_home/away'})


# In[23]:


md=dataset.groupby('type_of_shot').is_goal.agg(['mean'])
dataset=dataset.reset_index().merge(md,how="left",on="type_of_shot").set_index('index')
dataset = dataset.rename(columns={'mean': 'Avg_goal_type_of_shot'})


# In[24]:


md=dataset.groupby('type_of_combined_shot').is_goal.agg(['mean'])
dataset=dataset.reset_index().merge(md,how="left",on="type_of_combined_shot").set_index('index')
dataset = dataset.rename(columns={'mean': 'Avg_goal_type_of_combined_shot'})


# In[25]:


md=dataset.groupby('power_of_shot').is_goal.agg(['mean'])
dataset=dataset.reset_index().merge(md,how="left",on="power_of_shot").set_index('index')
dataset = dataset.rename(columns={'mean': 'Avg_goal_power_of_shot'})


# In[26]:


dataset.drop("power_of_shot",inplace=True,axis=1)
dataset.drop("knockout_match.1",inplace=True,axis=1)
dataset.drop("power_of_shot.1",inplace=True,axis=1)
dataset.drop("home/away",inplace=True,axis=1)
dataset.drop("type_of_combined_shot",inplace=True,axis=1)
dataset.drop("type_of_shot",inplace=True,axis=1)
dataset.drop("range_of_shot",inplace=True,axis=1)
dataset.drop("shot_basics",inplace=True,axis=1)
dataset.drop("area_of_shot",inplace=True,axis=1)
dataset.drop("game_season",inplace=True,axis=1)
dataset.drop("knockout_match",inplace=True,axis=1)
dataset.drop("distance_of_shot.1",inplace=True,axis=1)
dataset.drop("remaining_sec.1",inplace=True,axis=1)
dataset.drop("location_y",inplace=True,axis=1)
dataset.drop("remaining_min.1",inplace=True,axis=1)
dataset.drop("location_x",inplace=True,axis=1)


# In[27]:


dataset.drop("Avg_goal_range_of_shot",inplace=True,axis=1)
dataset.drop("Avg_goal_shot_basics",inplace=True,axis=1)
dataset.drop("Avg_goal_area_of_shot",inplace=True,axis=1)
dataset.drop("Avg_goal_type_of_combined_shot",inplace=True,axis=1)
dataset.drop("Avg_goal_type_of_shot",inplace=True,axis=1)


# In[28]:


dataset["remaining_min"].fillna(dataset["remaining_min"].mean(),inplace=True)
dataset["Avg_goal_power_of_shot"].fillna(dataset["Avg_goal_power_of_shot"].mean(),inplace=True)
dataset["remaining_sec"].fillna(dataset["remaining_sec"].mean(),inplace=True)
dataset["distance_of_shot"].fillna(dataset["distance_of_shot"].mean(),inplace=True)
dataset["Avg_goal_game_season"].fillna(dataset["Avg_goal_game_season"].mean(),inplace=True)
dataset["Avg_goal_home/away"].fillna(dataset["Avg_goal_home/away"].mean(),inplace=True)


# In[29]:


v=list(dataset[dataset["is_goal"].isnull()==False].index)
p=list(dataset[dataset["is_goal"].isnull()].index)


# In[30]:


train = dataset[dataset.index.isin(v)]


# In[31]:


test = dataset[dataset.index.isin(p)]


# In[32]:


o = pd.DataFrame(test, columns=['shot_id_number'])


# In[33]:


o=o.astype(int)


# In[34]:


train.drop("shot_id_number",inplace=True,axis=1)


# In[35]:


test.drop("shot_id_number",inplace=True,axis=1)


# In[36]:


X_train = train.drop(columns=['is_goal'])
y_train = train['is_goal']
X_test = test.drop(columns=['is_goal'])
y_test = test['is_goal']


# In[37]:


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,criterion='mae')
regressor.fit(X_train, y_train)


# In[38]:


y_pred = regressor.predict(X_test)


# In[39]:


y_pred_train = regressor.predict(X_train)


# In[40]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_train, y_pred_train))
print("Root mean squared error of prediction on training data is",rms)


# In[41]:


w = pd.DataFrame(y_pred, columns=['is_goal'])


# In[42]:


o.reset_index(drop="index",inplace=True)


# In[43]:


res= pd.concat([o, w], axis=1)


# In[44]:


res.to_csv('output.csv',encoding='utf-8', index=False)

