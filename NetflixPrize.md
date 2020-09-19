## Netflix Prize Code for a Movie Recommendation System

> The winner's code that has received the largest number of up-votes.
>
> ##### comments written by Astro Lee
>
> reference: [Netflix Prize Note](https://bit.ly/2FN160C)

*Note*

```
Each data file (there are 4 of them) contains below columns: 

- Movie ID (as first line of each new movie record / file)
- Customer ID
- Rating (1 to 5)
- Date they gave the ratings
There is another file contains the mapping of Movie ID to the movie background like name, year of release, etc
```



In [1]:

```python
import pandas as pd					# Pandas : High-level data manipulation tool
import numpy as np					# Numpy  : a library used for working with arrays 
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt		# matplotlib : Visualization tool
import seaborn as sns				# seaborn : Fancy statistics chart tool
from surprise import Reader, Dataset, SVD		# Surpise provides predictive algorithms like SVD
from surprise.model_selection import cross_validate				# cross_validate 
sns.set_style("darkgrid")
```



In [2]:

```python
# Skip date
df1 = pd.read_csv('../input/combined_data_1.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])                    # reading a dataset and make a table 

df1['Rating'] = df1['Rating'].astype(float)    # Rating to float-typed data

print('Dataset 1 shape: {}'.format(df1.shape)) # shows the number of rows and columns
print('-Dataset examples-')
print(df1.iloc[::5000000, :])                  # steps by 500000
												# .iloc() means making index with integer 
```

![image-20200918163612750](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918163612750.png)



In [3]:

```python
#df2 = pd.read_csv('../input/combined_data_2.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
#df3 = pd.read_csv('../input/combined_data_3.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
#df4 = pd.read_csv('../input/combined_data_4.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])


#df2['Rating'] = df2['Rating'].astype(float)
#df3['Rating'] = df3['Rating'].astype(float)
#df4['Rating'] = df4['Rating'].astype(float)

#print('Dataset 2 shape: {}'.format(df2.shape))
#print('Dataset 3 shape: {}'.format(df3.shape))
#print('Dataset 4 shape: {}'.format(df4.shape))


# loading other datasets
```



In [4]:

```python
# load less data for speed

df = df1
#df = df1.append(df2)
#df = df.append(df3)
#df = df.append(df4)                 # merge different datasets by append()

df.index = np.arange(0,len(df))
print('Full dataset shape: {}'.format(df.shape))
print('-Dataset examples-')
print(df.iloc[::5000000, :])               
```

![image-20200918163726240](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918163726240.png)





### ``Data Viewing``

In [5] :

```python
p = df.groupby('Rating')['Rating'].agg(['count'])			#grouping df by Rating, and to count grouped Rating 

# get movie count
movie_count = df.isnull().sum()[1]			# sum all NaN values to allocate em to movie count 

# get customer count
cust_count = df['Cust_Id'].nunique() - movie_count			# Cust_Id's unique values minus NaN values

# get rating count
rating_count = df['Cust_Id'].count() - movie_count			# Cust_Id's total minus movie_count

ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
plt.axis('off')			# generating a bar chart shwoing different ratings

for i in range(1,6):
    ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')			# showing ratio of each rating.															# .iloc() makes index with integer
```

![image-20200918163850754](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918163850754.png)



### `Data Cleaning`

In [6]:

```python
df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan = df_nan.reset_index()

movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

# Account for last record and corresponding length
# numpy approach
last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

print('Movie numpy: {}'.format(movie_np))
print('Length: {}'.format(len(movie_np)))
```

![image-20200918163946459](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918163946459.png)



In [7]: 

```python
# remove those Movie ID rows
df = df[pd.notnull(df['Rating'])]

df['Movie_Id'] = movie_np.astype(int)
df['Cust_Id'] = df['Cust_Id'].astype(int)
print('-Dataset examples-')
print(df.iloc[::5000000, :])
```

![image-20200918164148958](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918164148958.png)



### `Data Slicing`

In [8]:

```python

df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

print('Movie minimum times of review: {}'.format(movie_benchmark))

df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.7),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

print('Customer minimum times of review: {}'.format(cust_benchmark))
```

![image-20200918164237849](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918164237849.png)



In [9]:

```python
print('Original Shape: {}'.format(df.shape))
df = df[~df['Movie_Id'].isin(drop_movie_list)]
df = df[~df['Cust_Id'].isin(drop_cust_list)]
print('After Trim Shape: {}'.format(df.shape))
print('-Data Examples-')
print(df.iloc[::5000000, :])
```

![image-20200918164315064](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918164315064.png)



In [10]:

```python
df_p = pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')

print(df_p.shape)

# Below is another way I used to sparse the dataframe...doesn't seem to work better

#Cust_Id_u = list(sorted(df['Cust_Id'].unique()))
#Movie_Id_u = list(sorted(df['Movie_Id'].unique()))
#data = df['Rating'].tolist()
#row = df['Cust_Id'].astype('category', categories=Cust_Id_u).cat.codes
#col = df['Movie_Id'].astype('category', categories=Movie_Id_u).cat.codes
#sparse_matrix = csr_matrix((data, (row, col)), shape=(len(Cust_Id_u), len(Movie_Id_u)))
#df_p = pd.DataFrame(sparse_matrix.todense(), index=Cust_Id_u, columns=Movie_Id_u)
#df_p = df_p.replace(0, np.NaN)
```

![image-20200918164405850](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918164405850.png)



### `Data Mapping`

In [11]:

```python
df_title = pd.read_csv('../input/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)
print (df_title.head(10))
```

![image-20200918164602959](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918164602959.png)



### `Recommend with Collaborative Filtering`

In [12]:

```python
reader = Reader()

# get just top 100K rows for faster run time
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:], reader)
#data.split(n_folds=3)

svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'])
```

![image-20200918164831525](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918164831525.png)



In [13]:

```python
df_785314 = df[(df['Cust_Id'] == 785314) & (df['Rating'] == 5)]
df_785314 = df_785314.set_index('Movie_Id')
df_785314 = df_785314.join(df_title)['Name']
print(df_785314)
```

![image-20200918164929954](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918164929954.png)



In [14]:

```python
user_785314 = df_title.copy()
user_785314 = user_785314.reset_index()
user_785314 = user_785314[~user_785314['Movie_Id'].isin(drop_movie_list)]

# getting full dataset
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)

trainset = data.build_full_trainset()
svd.fit(trainset)

user_785314['Estimate_Score'] = user_785314['Movie_Id'].apply(lambda x: svd.predict(785314, x).est)

user_785314 = user_785314.drop('Movie_Id', axis = 1)

user_785314 = user_785314.sort_values('Estimate_Score', ascending=False)
print(user_785314.head(10))
```

![image-20200918165018055](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918165018055.png)



### `Recommend with Pearsons' R correlations`

In [15]:

```Python
def recommend(movie_title, min_count):
    print("For movie ({})".format(movie_title))
    print("- Top 10 movies recommended based on Pearsons'R correlation - ")
    i = int(df_title.index[df_title['Name'] == movie_title][0])
    target = df_p[i]
    similar_to_target = df_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
    corr_target.dropna(inplace = True)
    corr_target = corr_target.sort_values('PearsonR', ascending = False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(df_title).join(df_movie_summary)[['PearsonR', 'Name', 'count', 'mean']]
    print(corr_target[corr_target['count']>min_count][:10].to_string(index=False))
```



In [16]:

```python
recommend("What the #$*! Do We Know!?", 0)
```

![image-20200918165123670](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918165123670.png)



In [17]:

```python
recommend("X2: X-Men United", 0)
```

![image-20200918165202484](C:\Users\user\LearnGoodCode\NetflixPrize.assets\image-20200918165202484.png)





# END



