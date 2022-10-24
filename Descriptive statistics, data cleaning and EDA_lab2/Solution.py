#!/usr/bin/env python
# coding: utf-8

# # **CS418 LAB 2**

# ## DESCRIPTIVE STATISTICS

# ### **Loading the data into dataFrame from the CSV File**

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv (r'/Users/praveenrajveluswami/Documents/CS418 - IDS/lab2/heroes_information.csv')


# ### **Data type description of each column**

# * **ID:**  Discrete
# * **name:**  Nominal
# * **Gender:**  Nominal
# * **Eye color:**  Nominal
# * **Race:**  Nominal
# * **Hair color:**  Nominal
# * **Height:**  Continuous
# * **Publisher:**  Nominal
# * **Skin color:**  Nominal
# * **Alignment:**  Ordinal
# * **Weight:**  Continuous
# 

# ### **Compute the following measures for each column : Central tendencies and Dispersion**

# * For several columns here, especially the nominal and ordinal data types: name, Gender, eye color, Race, Hair color, Publisher, Skin color and Alignment, central tendencies and dispersion except mode cannot be calculated since their values are not numerical.
# * Proposed solution: Replace their values with integers i.e. 0,1,2,3... We replace blank/invalid values with 0.
# * We follow this approach for columns other than 'name' (name is neither qualitative not quantitative) and compute these values. Although we would get a mean,median,etc. values for 'Gender',etc., this will not logically make sense.

# #### Column name: **ID**

# **a.i. Mean :**

# Although mean can be computer for this column as the values are integers, it is not going to be helpful in gaining any insights about the data since these are just indexes of rows in our data.

# In[3]:


print(df["ID"].mean())


# **a.ii. Median :**

# Since the values are just indexes of rows, the median is simply the total number of records divided by 2. It gives us the row at the middle of our dataset although it doesn't serve much purpose.

# In[4]:


print(df["ID"].median())


# **a.iii. Mode :**

# The values in this column are never repeated. This means frequency of each value is 1.

# In[5]:


print(df["ID"].mode())


# **b.i. Standard Deviation :**

# In[6]:


print(df["ID"].std())


# **b.ii. Variance :**

# In[7]:


print(df.var()['ID'])


# **b.iii. IQR :**

# In[8]:


import numpy as np
q75, q25 = np.percentile(df['ID'], [75 ,25])
iqr = q75 - q25
print(iqr)


# **b.iv. Skew :**

# In[9]:


print(df['ID'].skew())


# #### Column name: **Name**

# **a.i. Mean :**

# Since this column has nominal data, it is not possible to compute mean. The values can also not be encoded into numercial form.

# **a.ii. Median :**

# Since this column has nominal data, it is not possible to compute median. The values can also not be encoded into numercial form.

# **a.iii. Mode :**

# Although the values are nominal, frequency for each value can be calculated through which mode can be computed. In this case, this might give insights into duplicate records.

# In[10]:


print(df["name"].mode())


# **b.i. Standard Deviation :**

# Nominal data. Standard Deviation cannot be computed.

# **b.ii. Variance :**

# Nominal data. Variance cannot be computed.

# **b.iii. IQR :**

# Nominal data. IQR cannot be computed.

# **b.iv. Skew :**

# Nominal data. Skew cannot be computed.

# #### Column name: **Gender**

# Gender can be encoded into numerical values. For example, male:1 ; female: 2

# In[11]:


df = df.replace('Male', 1)


# In[12]:


df = df.replace('Female', 2)


# In[13]:


df['Gender'] = df['Gender'].replace('-',0)


# In[14]:


df['Gender'] = df['Gender'].astype('int')


# #### Column name: **Eye color**

# Eye color, being categorical data, can be encoded into numerical values. The unique values in the column were retrieved using .unique() method. These can be encoded as 1,2,3...
# 

# In[15]:


df['Eye color'] = df['Eye color'].replace(['yellow', 'blue', 'green', 'brown', '-', 'red', 'violet', 'white',
       'purple', 'black', 'grey', 'silver', 'yellow / red',
       'yellow (without irises)', 'gold', 'blue / white', 'hazel',
       'green / blue', 'white / red', 'indigo', 'amber', 'yellow / blue',
       'bown'],[1,2,3,4,0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])


# In[16]:


df['Eye color'] = df['Eye color'].astype('int')


# #### Column name: **Race**

# This is same as the Eye color column. Race, being categorical data, can be encoded into numerical values. The unique values in the column were retrieved using .unique() method. These can be encoded as 1,2,3...
# 

# In[17]:


df['Race'] = df['Race'].replace(['Human', 'Icthyo Sapien', 'Ungaran', 'Human / Radiation',
       'Cosmic Entity', '-', 'Cyborg', 'Xenomorph XX121', 'Android',
       'Vampire', 'Mutant', 'God / Eternal', 'Symbiote', 'Atlantean',
       'Alien', 'Neyaphem', 'New God', 'Alpha', 'Bizarro', 'Inhuman',
       'Metahuman', 'Demon', 'Human / Clone', 'Human-Kree',
       'Dathomirian Zabrak', 'Amazon', 'Human / Cosmic',
       'Human / Altered', 'Kryptonian', 'Kakarantharaian',
       'Zen-Whoberian', 'Strontian', 'Kaiju', 'Saiyan', 'Gorilla',
       'Rodian', 'Flora Colossus', 'Human-Vuldarian', 'Asgardian',
       'Demi-God', 'Eternal', 'Gungan', 'Bolovaxian', 'Animal',
       'Czarnian', 'Martian', 'Spartoi', 'Planet', 'Luphomoid',
       'Parademon', 'Yautja', 'Maiar', 'Clone', 'Talokite', 'Korugaran',
       'Zombie', 'Human-Vulcan', 'Human-Spartoi', 'Tamaranean',
       'Frost Giant', 'Mutant / Clone', "Yoda's species"],[1,2,3,4,5,0,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61])


# In[18]:


df['Race'] = df['Race'].astype('int')


# #### Column name: **Hair color**

# This is same as the previous 2 columns. Hair color, being categorical data, can be encoded into numerical values. The unique values in the column were retrieved using .unique() method. These can be encoded as 1,2,3...
# 

# In[19]:


df['Hair color'] = df['Hair color'].replace(['No Hair', 'Black', 'Blond', 'Brown', '-', 'White', 'Purple',
       'Orange', 'Pink', 'Red', 'Auburn', 'Strawberry Blond', 'black',
       'Blue', 'Green', 'Magenta', 'Brown / Black', 'Brown / White',
       'blond', 'Silver', 'Red / Grey', 'Grey', 'Orange / White',
       'Yellow', 'Brownn', 'Gold', 'Red / Orange', 'Indigo',
       'Red / White', 'Black / Blue'],[1,2,3,4,0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])


# In[20]:


df['Hair color'] = df['Hair color'].astype('int')


# #### Column name: **Height**

# This is a continuous data. Hence, calculating central tendencies and dispersion shouldn't be a problem. Several rows have a value '-99' which is going to affect the below values. We will be handling these values during the data cleaning process.

# #### Column name: **Publisher**

# Publisher, being categorical data, can be encoded into numerical values. The unique values in the column were retrieved using .unique() method. These can be encoded as 1,2,3...
# 

# In[21]:


df['Publisher'] = df['Publisher'].replace(['Marvel Comics', 'Dark Horse Comics', 'DC Comics', 'NBC - Heroes',
       'Wildstorm', 'Image Comics', 'nan', 'Icon Comics', 'SyFy',
       'Hanna-Barbera', 'George Lucas', 'Team Epic TV', 'South Park',
       'HarperCollins', 'ABC Studios', 'Universal Studios', 'Star Trek',
       'IDW Publishing', 'Shueisha', 'Sony Pictures', 'J. K. Rowling',
       'Titan Books', 'Rebellion', 'Microsoft', 'J. R. R. Tolkien'],[1,2,3,4,5,6,0,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])


# In[22]:


df['Publisher'] = df['Publisher'].replace(np.nan, 0)


# In[23]:


df['Publisher'] = df['Publisher'].astype('int')


# #### Column name: **Skin color**

# Skin color, being categorical data, can be encoded into numerical values. The unique values in the column were retrieved using .unique() method. These can be encoded as 1,2,3...
# 

# In[24]:


df['Skin color'] = df['Skin color'].replace(['-', 'blue', 'red', 'black', 'grey', 'gold', 'green', 'white',
       'pink', 'silver', 'red / black', 'yellow', 'purple',
       'orange / white', 'gray', 'blue-white', 'orange'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])


# In[25]:


df['Skin color'] = df['Skin color'].astype('int')


# #### Column name: **Alignment**

# In[26]:


df['Alignment'].unique()


# In[27]:


df['Alignment'] = df['Alignment'].replace('good', 1)


# In[28]:


df['Alignment'] = df['Alignment'].replace('bad', 2)
df['Alignment'] = df['Alignment'].replace('neutral', 3)


# In[29]:


df['Alignment'] = df['Alignment'].replace('-',0)


# In[30]:


df['Alignment'] = df['Alignment'].astype('int')


# #### Column name: **Weight**

# This is a continuous data. Hence, calculating central tendencies and dispersion shouldn't be a problem. Several rows have a value '-99' which is going to affect the below values. We will be handling these values during the data cleaning process.

# #### Computing the values for each of the above columns:

# In[31]:


for (columnName, columnData) in df[["Gender", "Eye color","Race","Hair color","Height","Publisher","Skin color","Alignment","Weight"]].iteritems():

    mean = df[columnName].mean()
    median = df[columnName].median()
    mode = df[columnName].mode()
    std = df[columnName].std()
    var = df[columnName].var()
    skw = df[columnName].skew()
    
    q75, q25 = np.percentile(df[columnName], [75 ,25])
    iqr = q75 - q25
    
    print("\033[1m" + "Column : " + columnName + "\033[0m")
    print('---------------------------------')
    print('Mean :', mean)
    print('Median :', median)
    print('Mode :', mode)
    print('Standard Deviation :', std)
    print('Variance :', var)
    print('IQR :', iqr)
    print('Skew :', skw)
    print('=======================================================================')
    


# ### **Cleaning dataset**

# A part of data cleaning has been done earlier in the previous section to calculate central tendencies and dispersion.
#     ---> Categorical data were numerically encoded while '-' values were encoded as 0.
#     
# There are so many rows which have blank/invalid values in certain columns. However, removing these rows will result in loosing a major chunk of our data.
# Therefore, the method followed here is to find useless/junk rows with many blank or invalid values.
# * The code snippet below looks for rows which have more than two columns with blank or invalid values. These rows  are then removed from the data. 

# In[32]:


df = df


# In[33]:


df = df.fillna(0)
for ind, row in df.iterrows():
    count = 0
    for column in df:
        if column != 'ID':
            if row[column] == 0:
                count = count + 1
            elif row[column] == -99:
                count = count + 1
    if count >= 2:
        df.drop(ind,axis=0,inplace=True)
#         print('deleting ',ind)


# 1. Meanwhile, in the other rows, these blank/invalid values are handled as follows:
#     * Categorical data i.e. Gender, Eye color, Race, Hair color, Publisher, Skin color, Alignment : Blank/invalid values are replaced with the mode of the column, since logically, the row has more probability of having the value of mode than any other value in the column. In case the mode itself is 0, we replace it with the next most frequent value.
#     * Continuous data i.e. Height and Weight : Blank/invalid values are replaced with the mean of the column, since this will make our assumption impact the distribution of data neither positively nor negatively.
#     
#     
# 2. Handling outliers: Outliers usually end up skewing the data, thus affecting the central tendencies and dispersion values. They also affect classification, prediction or any kind of inference. For example, there are few records which have a height around 800-1000 which are far from the other records. The idea is to replace these outliers with the upper and low quartiles respectively. Only continous data like height and weight are prone to outliers. Therefore, we will implement outlier removal only for those columns.

# In[34]:


# Confirming that the mode is not 0
for (columnName, columnData) in df.iteritems():
    if columnName != 'ID' and columnName != 'name':
        init_mode = df[columnName].value_counts().idxmax()
        if init_mode == 0:  ## Checking whether the mode itself is 0
            df_temp = df[df[columnName] != 0]  ## If mode is 0, remove the zeroes
            df[columnName] = df[columnName].replace([0], df_temp[columnName].value_counts().idxmax()) ## Replace zeroes in the original df with the new mode
        else:
            df[columnName] = df[columnName].replace([0], df[columnName].value_counts().idxmax())


# #### Outlier removal

# Here, we replace outliers with median value so as to not affect the distribution since mean, mode can be influenced by the outliers. Replacing with upper bound or lower bound was eliminated because height and weight columns mostly will have outliers that are greater than Q3 more than outliers that are less than Q1. We want to avoid the upper bound getting over populated.

# In[35]:


# Height
Q1 = df['Height'].quantile(0.25)
Q3 = df['Height'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

for index, row in df.iterrows():
    if row['Height'] < lower_bound: 
        df['Height'] = df['Height'].replace(row['Height'], df['Height'].median())
    elif row['Height'] > upper_bound: 
        df['Height'] = df['Height'].replace(row['Height'], df['Height'].median())


# In[36]:


# Weight
Q1 = df['Weight'].quantile(0.25)
Q3 = df['Weight'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

for index, row in df.iterrows():
    if row['Weight'] < lower_bound: 
        df['Weight'] = df['Weight'].replace(row['Weight'], df['Weight'].median())
    elif row['Weight'] > upper_bound: 
        df['Weight'] = df['Weight'].replace(row['Weight'], df['Weight'].median())


# ### Scatter plot

# In[37]:


import matplotlib.pyplot as plt


# **For each column, the first plot represents the scatter plot as asked in the question and the second plot is an alternative.**

# #### Gender

# In[38]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.title.set_text('Gender vs ID') 
ax1.scatter(df['ID'], df['Gender'], c ="blue")
plt.ylabel('ID')

ax2.title.set_text('Gender vs Frequency') 
ax2.bar(df['Gender'].value_counts().index, 
       df['Gender'].value_counts().values,
       color = ['darkblue', 'darkorange'])
ax2.set_xticks(range(0, 3))
ax2.set_xticklabels(['0','1 - Male','2 - Female'], fontsize = 14);
plt.show()


# **Learning from the first plot :** Shows that 2 i.e. Females are less in number than 1 i.e. males. Both males and females are evenly distributed in the data.
# **Alternative plot :** The first plot does not give us any quantitave understanding of gender with respect to ID since ID is just an index. The alternative plot helps us compare the frequency of both the genders.
# **Learning from the alternative plot :** Apparently there are more males in the data than females.

# #### Eye color

# In[39]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.title.set_text('Eye color vs ID') 
ax1.scatter(df['ID'], df['Eye color'], c ="blue")
plt.ylabel('ID')

ax2.title.set_text('Eye color vs Frequency') 
ax2.bar(df['Eye color'].value_counts().index, 
       df['Eye color'].value_counts().values,
       color = ['darkblue', 'darkorange'])
# ax2.set_xticks(range(0, 3))
# ax2.set_xticklabels(['0','1 - Male','2 - Female'], fontsize = 14);
plt.show()


# **Learning from the first plot :** Shows that the eye colors are not evenly distributed. The later part of the data seems to be more diverse than the initial part. The difference between the frequency of mode and the least occuring value is way too high.
# **Alternative plot :** The first plot does not give us any quantitave understanding of eye color with respect to ID since ID is just an index. The alternative plot helps us compare the frequency of the eye colors.
# **Learning from the alternative plot :** 2 i.e. Blue seems to be the most common eye color.

# #### Race

# In[40]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.title.set_text('Race vs ID') 
ax1.scatter(df['ID'], df['Race'], c ="blue")
plt.ylabel('ID')

ax2.title.set_text('Race vs Frequency') 
ax2.bar(df['Race'].value_counts().index, 
       df['Race'].value_counts().values,
       color = ['darkblue', 'darkorange'])
# ax2.set_xticks(range(0, 3))
# ax2.set_xticklabels(['0','1 - Male','2 - Female'], fontsize = 14);
plt.show()


# **Learning from the first plot :** Shows that Race is not evenly distributed. The later part of the data seems to be more diverse than the initial part. The difference between the frequency of mode and the least occuring value is way too high. The two dominant races are Human and Mutant.
# **Alternative plot :** The first plot does not give us any quantitave understanding of race with respect to ID since ID is just an index. The alternative plot helps us compare the frequency of the race.
# **Learning from the alternative plot :** 1 i.e. Human seems to be the most common race.
# 
# **However in this case, the scatter plot makes more sense since even the least occuring race is able to be visualized.**

# #### Hair color

# In[41]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.title.set_text('Hair color vs ID') 
ax1.scatter(df['ID'], df['Hair color'], c ="blue")
plt.ylabel('ID')

ax2.title.set_text('Hair color vs Frequency') 
ax2.bar(df['Hair color'].value_counts().index, 
       df['Hair color'].value_counts().values,
       color = ['darkblue', 'darkorange'])
# ax2.set_xticks(range(0, 3))
# ax2.set_xticklabels(['0','1 - Male','2 - Female'], fontsize = 14);
plt.show()


# **Learning from the first plot :** Shows that Hair color is evenly distributed barring few hair colors like Indigo, Red/White and Black/Blue. The least occuring hair colors can be seen more easily. The most occuring colors are more evenly distributed than the less occuring ones.
# **Alternative plot :** The alternative plot helps us compare the frequency of the hair colors.
# **Learning from the alternative plot :** 2 i.e. Black seems to be the most common race.
# 
# **In this case too, the scatter plot makes more sense since even the least occuring race is able to be visualized.**

# #### Publisher

# In[42]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.title.set_text('Publisher vs ID') 
ax1.scatter(df['ID'], df['Publisher'], c ="blue")
plt.ylabel('ID')

ax2.title.set_text('Publisher vs Frequency') 
ax2.bar(df['Publisher'].value_counts().index, 
       df['Publisher'].value_counts().values,
       color = ['darkblue', 'darkorange'])
# ax2.set_xticks(range(0, 3))
# ax2.set_xticklabels(['0','1 - Male','2 - Female'], fontsize = 14);
plt.show()


# **Learning from the first plot :** Shows how the distribution of Publishers have been affected due to data cleaning. Records having values SouthPark, HaperCollins have been completely removed as part of data cleaning as thy might have had more than 2 invalid values.
# **Alternative plot :** The alternative plot helps us compare the frequency of the hair colors.
# **Learning from the alternative plot :** How only 3-4 values form the major part of the data
# 
# **In this case too, both the plots are equally good in representing the nature of the data**

# #### Alignment

# In[43]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.title.set_text('Alignment vs ID') 
ax1.scatter(df['ID'], df['Alignment'], c ="blue")
plt.ylabel('ID')

ax2.title.set_text('Alignment vs Frequency') 
ax2.bar(df['Alignment'].value_counts().index, 
       df['Alignment'].value_counts().values,
       color = ['darkblue', 'darkorange'])
# ax2.set_xticks(range(0, 3))
# ax2.set_xticklabels(['0','1 - Male','2 - Female'], fontsize = 14);
plt.show()


# **Learning from the first plot :** 3 i.e. neutral seems to be very low compared to good and bad.
# **Alternative plot :** The alternative plot helps us compare the frequency of the hair colors.
# **Learning from the alternative plot :** How only 3-4 values form the major part of the data
# 
# **In this case, the scatter plot doesn't help much in understanding the difference between the distribution of 'good' and 'bad' i.e. 1 and 2. However, the alternative plot acheives this.**

# #### Height

# In[44]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.title.set_text('Height vs ID') 
ax1.scatter(df['ID'], df['Height'], c ="blue")
plt.ylabel('Height')
plt.xlabel('ID')

# ax2.title.set_text('Height vs Frequency') 
# ax2.bar(df['Height'].value_counts().index, 
#        df['Height'].value_counts().values,
#        color = ['darkblue', 'darkorange'])

plt.plot(df['ID'], df['Height'])
plt.title('Height Vs ID')
plt.xlabel('ID')
plt.ylabel('Height')
plt.show()

# ax2.set_xticks(range(0, 3))
# ax2.set_xticklabels(['0','1 - Male','2 - Female'], fontsize = 14);
plt.show()


# **Learning from the first plot :** Shows that the heights are mostly concentrated around 180-190. Records in the bottom of the data tend to be more taller than the ones on the top. We can also see the values being capped at the upper and lower bounds which we did during outlier removal.
# **Alternative plot :** The alternative plot helps us infer the trend in the height as we go down the records.While scatter plots doesn't perform well in showing the trends in the densely populated areas like height:200, the alternative plot achieves this.
# **Learning from the alternative plot :** The values are more skewed as we go further down our data.

# #### Weight

# In[45]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.title.set_text('Weight vs ID') 
ax1.scatter(df['ID'], df['Weight'], c ="blue")
plt.ylabel('Weight')
plt.xlabel('ID')


plt.plot(df['ID'], df['Weight'])
plt.title('Weight Vs ID')
plt.xlabel('ID')
plt.ylabel('Weight')
plt.show()

# ax2.set_xticks(range(0, 3))
# ax2.set_xticklabels(['0','1 - Male','2 - Female'], fontsize = 14);
plt.show()


# **Learning from the first plot :** Shows that the weights are mostly concentrated around 50-75. There's a nearly even distribution of weights among all the records. We can also see the values being capped at the upper and lower bounds which we did during outlier removal.
# **Alternative plot :** The alternative plot helps us infer the trend in the weight as we go down the records. Helps us infer the trends even in the highly dense areas like weight:50-75.
# **Learning from the alternative plot :** There are more overweight superheroes as we go down the data.

# ### Box-plots

# In[46]:


# c = 1
# for (columnName, columnData) in df.iteritems():
#     plt.subplot(4, 2, c)
# #     plt.title('Weight Vs ID')
# #     plt.xlabel('ID')
# #     plt.ylabel('Weight')
#     boxplot = df.boxplot(column=columnName)
#     c = c + 1
for (columnName, columnData) in df.iteritems():
    if columnName != 'name':
        plt.figure()
        boxplot = df.boxplot(column=columnName)


# **Boxplot for 'name' cannot be plotted since it is not numerical.
# Although the question asks for boxplot for each column, box plots for categorical data are not usually helpful 
# in any means since they are not quantitative data. In our case the boxplots only for height and weight are useful.**

# ## EXPLORATORY DATA ANALYSIS

# ### Describing the data

# In[47]:


df.describe()


# ### Visualizing Distribution with Histograms

# In[48]:


his = df.hist(figsize=(30,30))


# ### Scatter matrix for four columns : Height, Weight, Race, Alignment

# In[49]:


import seaborn as sns
sns.pairplot(df[["Height", "Weight", "Race", "Alignment"]])


# ### Scatter plot for variables with no relationship

# **Columns chosen: Publisher and Height**

# In[50]:


ax1 = df.plot.scatter(x='Publisher',y='Height',c='DarkBlue')


# Covariance:

# In[51]:


cov_unr = df[["Publisher","Height"]].cov()
print(cov_unr)


# Correlation:

# In[52]:


corr = df['Publisher'].corr(df['Height'])
print(corr)


# ### Scatter plot for variables which seem to have a relationship

# **Columns chosen: Height and Weight**

# In[53]:


ax1 = df.plot.scatter(x='Height',y='Weight',c='DarkBlue')


# Covariance:

# In[54]:


cov_r = df[["Weight","Height"]].cov()
print(cov_r)


# Correlation:

# In[55]:


corr_r = df['Weight'].corr(df['Height'])
print(corr_r)


# In[ ]:




