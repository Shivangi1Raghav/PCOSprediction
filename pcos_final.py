#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pcos = pd.read_excel("PCOS.xlsx")


# # Data Cleaning

# In[3]:


pcos.tail()


# In[4]:


pcos.drop(["Unnamed: 44","Sl. No","Patient File No."],axis="columns", inplace= True)


# In[5]:


pcos


# In[6]:


pcos.info()


# In[7]:


#Dealing with categorical values.
#In this database the type objects are numeric values saved as strings.
#So I am just converting it into a numeric value.

pcos["AMH(ng/mL)"] = pd.to_numeric(pcos["AMH(ng/mL)"], errors='coerce')
pcos["II    beta-HCG(mIU/mL)"] = pd.to_numeric(pcos["II    beta-HCG(mIU/mL)"], errors='coerce')

#Dealing with missing values. 
#Filling NA values with the median of that feature.

pcos['Marraige Status (Yrs)'].fillna(pcos['Marraige Status (Yrs)'].median(),inplace=True)
pcos['II    beta-HCG(mIU/mL)'].fillna(pcos['II    beta-HCG(mIU/mL)'].median(),inplace=True)
pcos['AMH(ng/mL)'].fillna(pcos['AMH(ng/mL)'].median(),inplace=True)
pcos['Fast food (Y/N)'].fillna(pcos['Fast food (Y/N)'].median(),inplace=True)


# # Modeling Data

# ### Used all the given inputs to predict PCOS
# 
# ### Using Random Forest Classifier

# In[104]:


Models =["Random Forest", "Decision Tree", "SVC", "Logistic Regression", "Random Forest(Chi Square)"]
Scores =[]


# In[9]:


x_train,x_test,y_train,y_test =train_test_split(pcos.drop("PCOS (Y/N)",axis="columns"),pcos['PCOS (Y/N)'], test_size=0.2, random_state=0)


# In[10]:


from sklearn.ensemble import RandomForestClassifier
All_factors = RandomForestClassifier(max_depth=40)
All_factors.fit(x_train,y_train)


# In[11]:


All_factors.score(x_test,y_test)


# #### Predicting Values

# In[12]:


All_factors.predict(x_test)


# In[13]:


y_test


# In[105]:


Scores.append(All_factors.score(x_test,y_test))


# ### RANDOM FOREST CONFUSION MATRIX

# In[15]:


y_pred=All_factors.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()


# In[16]:


from sklearn import tree
All_Factors2 = tree.DecisionTreeClassifier()
All_Factors2.fit(x_train,y_train)
All_Factors2.score(x_test,y_test)


# In[106]:


Scores.append(All_Factors2.score(x_test,y_test))


# ### DECISION TREES CONFUSION MATRIX

# In[18]:


y_pred=All_factors.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, cmap="Purples")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()


# In[19]:


from sklearn.svm import SVC
All_Factors3= SVC()
All_Factors3.fit(x_train,y_train)
All_Factors3.score(x_test,y_test)


# In[107]:


Scores.append(All_Factors3.score(x_test,y_test))


# ### SVC CONFUSION MATRIX

# In[21]:


y_pred=All_Factors3.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, cmap="OrRd")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()


# In[22]:


from sklearn.linear_model import LogisticRegression
All_Factors4 = LogisticRegression()
All_Factors4.fit(x_train,y_train)
All_Factors4.score(x_test,y_test)


# In[108]:


Scores.append(All_Factors4.score(x_test,y_test))


# ### LOGISTIC REGRESSION CONFUSION MATRIX

# In[24]:


y_pred=All_Factors4.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, cmap="Wistia")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()


# In[25]:


corrmat = pcos.corr()
plt.subplots(figsize=(18,18))
sns.heatmap(corrmat,cmap="inferno", square=True);
sns.set(font_scale=1.45)


# In[26]:


from sklearn.feature_selection import mutual_info_classif
imp = mutual_info_classif(pcos.drop("PCOS (Y/N)",axis="columns"), pcos['PCOS (Y/N)'])
feat_imp = pd.Series(imp, pcos.columns[1:len(pcos.columns)])
feat_imp.plot(kind='barh', color='teal', figsize=(12,15), fontsize=14, grid=True)
plt.show()


# In[27]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
chi2_feat= SelectKBest(chi2,k=10)
feat= chi2_feat.fit_transform(pcos.drop("PCOS (Y/N)",axis="columns"), pcos['PCOS (Y/N)'])
feat.shape


# In[28]:


Xc_train,Xc_test,y_train,y_test =train_test_split(feat ,pcos['PCOS (Y/N)'], test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier
chi2_factors = RandomForestClassifier(max_depth=40)
chi2_factors.fit(Xc_train,y_train)
chi2_factors.score(Xc_test,y_test)


# In[29]:


chi2_feat2= SelectKBest(chi2,k=30)
feat= chi2_feat2.fit_transform(pcos.drop("PCOS (Y/N)",axis="columns"), pcos['PCOS (Y/N)'])
feat.shape


# In[101]:


Xc2_train,Xc2_test,y_train,y_test =train_test_split(feat ,pcos['PCOS (Y/N)'], test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier
chi2_factors2 = RandomForestClassifier(max_depth=40)
chi2_factors2.fit(Xc2_train,y_train)
chi2_factors2.score(Xc2_test,y_test)


# In[109]:


Scores.append(chi2_factors2.score(Xc2_test,y_test))


# In[110]:


d1 = pd.DataFrame()
d1["models"]= Models
d1["Score"]=Scores
colour=["Indianred","darksalmon","darkcyan","Slateblue", "red"]
d1.plot(kind="bar", x="models", y="Score", grid=True, figsize=(9,6), fontsize=15, color=colour)
plt.title('How different attributes affect PCOS\n\n', fontsize=20)


# # How Physical Symptoms affect PCOS

# Symptoms that can be observed without medical testing

# GRAPH

# In[33]:


attributes2A=["PCOS (Y/N)",'Weight gain(Y/N)', 'hair growth(Y/N)','Skin darkening (Y/N)',              'Hair loss(Y/N)', 'Pimples(Y/N)','Fast food (Y/N)', 'Reg.Exercise(Y/N)']
g=sns.pairplot(pcos[attributes2A], hue="PCOS (Y/N)", corner=True, diag_kind="kde", markers=["*","o"])
g.map_lower(sns.kdeplot, levels=4, color=".2")


# ### Individual Physical Factors affecting PCOS

# In[34]:


attributes2A1=['Weight gain(Y/N)', 'hair growth(Y/N)','Skin darkening (Y/N)',              'Hair loss(Y/N)', 'Pimples(Y/N)','Fast food (Y/N)', 'Reg.Exercise(Y/N)']
colours=['PaleTurquoise','HotPink']
for i in attributes2A1:
    sns.swarmplot(x=pcos["PCOS (Y/N)"], y=pcos[i], color="black", alpha=0.5 )
    sns.boxenplot(x=pcos["PCOS (Y/N)"], y=pcos[i], palette=colours)
    plt.show()


# ## PREDICTION OF PCOS USING PHYSICAL SYMPTOMS

# In[35]:


Models =["Random Forest", "Decision Tree", "SVC", "Logistic Regression"]
Score1 =[]


# ### - Using Random Forest Classifier

# In[36]:


model1= RandomForestClassifier(max_depth=40)
attributes3=['Weight gain(Y/N)', 'hair growth(Y/N)',       'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',       'Fast food (Y/N)', 'Reg.Exercise(Y/N)']

x_train1,x_test1,y_train1,y_test1 =train_test_split(pcos[attributes3],pcos['PCOS (Y/N)'], test_size=0.2, random_state=0)


# In[37]:


model1.fit(x_train1,y_train1)


# In[38]:


model1.score(x_test1,y_test1)


# In[39]:


Score1.append(model1.score(x_test1,y_test1))


# ### - Using Decision Tree

# In[40]:


model1A = tree.DecisionTreeClassifier()
model1A.fit(x_train1,y_train1)


# In[41]:


model1A.score(x_test1,y_test1)


# In[42]:


Score1.append(model1A.score(x_test1,y_test1))


# ### - Using SVC

# In[43]:


model1B= SVC()
model1B.fit(x_train1,y_train1)


# In[44]:


model1B.score(x_test1,y_test1)


# In[45]:


Score1.append(model1B.score(x_test1,y_test1))


# ### - Using Logistic Regression

# In[46]:


model1C = LogisticRegression()
model1C.fit(x_train1,y_train1)


# In[47]:


model1C.score(x_test1,y_test1)


# In[48]:


Score1.append(model1C.score(x_test1,y_test1))


# In[49]:


d1 = pd.DataFrame()
d1["models"]= Models
d1["Score"]=Score1
colour=["Indianred","darksalmon","darkcyan","Slateblue"]
d1.plot(kind="bar", x="models", y="Score", grid=True, figsize=(9,6), fontsize=15, color=colour)
plt.title('How Physical Symptoms affect PCOS\n\n', fontsize=20)


# # How hormones affect pcos

# GRAPH

# In[50]:


attributes=['PCOS (Y/N)','  I   beta-HCG(mIU/mL)',       'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)','Vit D3 (ng/mL)', 'PRG(ng/mL)','RBS(mg/dl)']
hormone = pcos[attributes]
m= sns.pairplot(data=hormone, hue="PCOS (Y/N)", corner=True, diag_kind="kde", markers=["*","o"])
m.map_lower(sns.kdeplot, levels=4, color=".2")


# ## Individual Hormones Effects on PCOS 

# In[51]:


attributes1=['  I   beta-HCG(mIU/mL)',       'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)','Vit D3 (ng/mL)', 'PRG(ng/mL)','RBS(mg/dl)','TSH (mIU/L)',       'AMH(ng/mL)', 'PRL(ng/mL)']
for i in attributes1:
    sns.swarmplot(x=pcos["PCOS (Y/N)"], y=pcos[i], color="black", alpha=0.5 )
    sns.boxenplot(x=pcos["PCOS (Y/N)"], y=pcos[i], palette=colours)
    plt.show()


# PREDICTION OF PCOS USING THE HORMONES

# In[52]:


Models =["Random Forest", "Decision Tree", "SVC", "Logistic Regression"]
Score =[]


# ## - Using Random Forest Classfier

# In[53]:


attributes4=['  I   beta-HCG(mIU/mL)',       'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)','Vit D3 (ng/mL)', 'PRG(ng/mL)','RBS(mg/dl)','TSH (mIU/L)',       'AMH(ng/mL)', 'PRL(ng/mL)']
model2=RandomForestClassifier(max_depth=40)
x_train2,x_test2,y_train2,y_test2 =train_test_split(pcos[attributes4],pcos['PCOS (Y/N)'], test_size=0.2, random_state=0)
model2.fit(x_train2,y_train2)


# In[54]:


model2.score(x_test2,y_test2)


# In[55]:


Score.append(model2.score(x_test2,y_test2))


# ## - Using Decision Tree

# In[56]:


model2A = tree.DecisionTreeClassifier()
model2A.fit(x_train2,y_train2)
model2A.score(x_test2, y_test2)


# In[57]:


Score.append(model2A.score(x_test2,y_test2))


# ## - Using SVC

# In[58]:


model2B = SVC()
model2B.fit(x_train2,y_train2)
model2B.score(x_test2, y_test2)


# In[59]:


Score.append(model2B.score(x_test2,y_test2))


# ## - Using Logistic Regression

# In[60]:


model2C = LogisticRegression()
model2C.fit(x_train2,y_train2)
model2C.score(x_test2, y_test2)


# In[61]:


Score.append(model2C.score(x_test2, y_test2))


# In[62]:


d2 = pd.DataFrame()
d2["models"]= Models
d2["Score"]=Score
colour=["Indianred","darksalmon","darkcyan","Slateblue"]
d2.plot(kind="bar", x="models", y="Score", grid=True, figsize=(9,6), fontsize=15, color=colour)
plt.title('How Hormone affect PCOS\n\n', fontsize=20)


# # How Blood and Breathing parameters helps in predicting PCOS

# GRAPH

# In[63]:


attributes2=['PCOS (Y/N)','BMI','Blood Group','Pulse rate(bpm) ','RR (breaths/min)','Hb(g/dl)']
n= sns.pairplot(pcos[attributes2], hue='PCOS (Y/N)', corner=True, diag_kind="kde", markers=["*","o"])
n.map_lower(sns.kdeplot, levels=4, color=".2")


# In[64]:


attributes5=['Blood Group', 'Pulse rate(bpm) ', 'RR (breaths/min)', 'Hb(g/dl)']
for i in attributes5:
    sns.swarmplot(x=pcos["PCOS (Y/N)"], y=pcos[i], color="black", alpha=0.5 )
    sns.boxenplot(x=pcos["PCOS (Y/N)"], y=pcos[i], palette=colours)
    plt.show()


# ## PREDICTING PCOS USING BLOOD AND BREATHING REPORTS

# In[65]:


Models =["Random Forest", "Decision Tree", "SVC", "Logistic Regression"]
Score2 =[]


# ### - Random Forest Classifier

# In[66]:


model3=RandomForestClassifier(max_depth=40)
x_train3,x_test3,y_train3,y_test3 =train_test_split(pcos[attributes5],pcos['PCOS (Y/N)'], test_size=0.2)
model3.fit(x_train3,y_train3)


# In[67]:


model3.score(x_test3,y_test3)


# In[68]:


Score2.append(model3.score(x_test3,y_test3))


# ### - Using Decision Tree 

# In[69]:


model3A = tree.DecisionTreeClassifier()
model3A.fit(x_train3,y_train3)
model3A.score(x_test3, y_test3)


# In[70]:


Score2.append(model3.score(x_test3,y_test3))


# ### - Using SVC

# In[71]:


model3B = SVC()
model3B.fit(x_train3,y_train3)
model3B.score(x_test3, y_test3)


# In[72]:


Score2.append(model3B.score(x_test3, y_test3))


# ### - Using Logistic Regression

# In[73]:


model2C = LogisticRegression()
model2C.fit(x_train3,y_train3)
model2C.score(x_test3, y_test3)


# In[74]:


Score2.append(model2C.score(x_test3, y_test3))


# In[75]:


d3 = pd.DataFrame()
d3["models"]= Models
d3["Score"]=Score2
colour=["Indianred","darksalmon","darkcyan","Slateblue"]
d3.plot(kind="bar", x="models", y="Score", grid=True, figsize=(9,6), fontsize=15, color=colour)
plt.title('How Breathing and Blood Parameters affect PCOS\n\n', fontsize=20)


# # Personal Features Affecting PCOS

# In[76]:


attributes6 = [' Age (yrs)', 'Weight (Kg)', 'Height(Cm) ', 'BMI','Cycle(R/I)', 'Cycle length(days)',               'Marraige Status (Yrs)',       'Pregnant(Y/N)', 'No. of aborptions']
for i in attributes6:
    sns.swarmplot(x=pcos["PCOS (Y/N)"], y=pcos[i], color="black", alpha=0.5 )
    sns.boxenplot(x=pcos["PCOS (Y/N)"], y=pcos[i], palette=colours)
    plt.show()


# # Internal Organs Functioning Affecting PCOS

# In[77]:


attributes7 = ['BP _Systolic (mmHg)',       'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)',       'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)']
for i in attributes7:
    sns.swarmplot(x=pcos["PCOS (Y/N)"], y=pcos[i], color="black", alpha=0.5 )
    sns.boxenplot(x=pcos["PCOS (Y/N)"], y=pcos[i], palette=colours)
    plt.show()


# In[78]:


Models =["Random Forest", "Decision Tree", "SVC", "Logistic Regression"]
Score4 =[]


# In[79]:


model4=RandomForestClassifier(max_depth=40)
x_train4,x_test4,y_train4,y_test4 =train_test_split(pcos[attributes4],pcos['PCOS (Y/N)'], test_size=0.2, random_state=0)
model4.fit(x_train4,y_train4)


# In[80]:


model4.score(x_test4,y_test4)


# In[81]:


Score4.append(model4.score(x_test4,y_test4))


# In[82]:


model4A = tree.DecisionTreeClassifier()
model4A.fit(x_train4,y_train4)
model4A.score(x_test4, y_test4)


# In[83]:


Score4.append(model4A.score(x_test4,y_test4))


# In[84]:


model4B = SVC()
model4B.fit(x_train4,y_train4)
model4B.score(x_test4, y_test4)


# In[85]:


Score4.append(model4B.score(x_test4, y_test4))


# In[86]:


model4C = LogisticRegression()
model4C.fit(x_train4,y_train4)
model4C.score(x_test4, y_test4)


# In[87]:


Score4.append(model4C.score(x_test4, y_test4))


# In[88]:


d4 = pd.DataFrame()
d4["models"]= Models
d4["Score"]=Score4
colour=["Indianred","darksalmon","darkcyan","Slateblue"]
d4.plot(kind="bar", x="models", y="Score", grid=True, figsize=(9,6), fontsize=15, color=colour)
plt.title('Internal Organs Functioning Affecting PCOS\n\n', fontsize=20)


# In[111]:


attributes4A=["PCOS (Y/N)",'BP _Systolic (mmHg)',       'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)',       'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)']
g=sns.pairplot(pcos[attributes4A], hue="PCOS (Y/N)", corner=True, diag_kind="kde", markers=["*","o"])
g.map_lower(sns.kdeplot, levels=4, color=".2")

