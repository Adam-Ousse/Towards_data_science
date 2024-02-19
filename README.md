## Trying to learn the ways of data science

Why clean data :
It's crucial to clean the data we use as it **_directly influence the insights_** we extract from our data.
Garbage in Garbage out.

### Data type constraints :
Text data : str
Numeric data : int, float
Dates and times : datetime
Categories : category
Binary : bool
To check the data types used in a Pandas dataframe we can use :
```python
df.dtypes
#or
df.info()
```
Examples of the importance of data types :
```python
sales['Revenue'].sum()
#return "214$41$24312$1313$12313$"
#meaning that the data type was a string to fix that :
sales['Revenue'] = sales['Revenue'].str.strip('$')
sales['Revenue'] = sales['Revenue'].astype('int')
assert sales['Revenue'].dtype == 'int'
```

#### Categorical data :
Numeric or categorical ? 
marriage status : single:1, married:2, divorced:3
This is a categorical data, but it's stored as a numeric data.
To fix that we can use the .astype() method to convert the data to a category type.
```python
sales['marriage_status'] = sales['marriage_status'].astype('category')
sales.describe()
```
#### Data in range constraint :
if for example a movie rating is supposed to be with 0 and 5, but we have a 10 in the data, we are sure that this is a mistake because it well beyond the range.
or for example having dates in the future.
for this last example  we could do for example :
```python
import datetime as dt
today = dt.date.today()
#convert to date : 
user_signups['signup_date'] = pd.to_datetime(user_signups['signup_date'])
user_signups[user_signups['signup_date'] > today]
```
We can drop the out of range values, but this is only preferable if the number of outlays is small.
We can set maximums and minimums.
We can treat the data as missing or impute.
or assign a custom value
```python
movies[movies['avg_rating'] > 5] #select the movies with a rating higher than 5
#we can drop them :
movies = movies[movies['avg_rating'] <= 5]
#or by thr .drop() method
movies.drop(movies[movies['avg_rating'] > 5].index, inplace=True)

#we can set a maximum
movies.loc[movies['avg_rating'] > 5, 'avg_rating'] = 5



```
#### Duplicate data / uniqueness :
finding duplicates values : 
```python
duplicates = df.duplicated()
print(duplicates) #True or False values
df[duplicates] #to see the duplicated values
df.duplicated(subset=['column1', 'column2']) #to check for duplicates in a subset of columns
df.drop_duplicates(keep = ) # "first" or "last" or False(keep all duplicates) 
#drop complete duplicates (same values for all columns):
df.drop_duplicates(inplace=True)
#for incomplete duplicates with different values we can combine them with avg : .groupby() and .agg()
df.groupby(by=["name", "family_name","adres"]).agg({'height':'max',"weight":"mean"}).reset_index()

```




## Possible operations on a dataframe column : 
```python
df["lol"].sum()
df["lol"].str.strip("something")
df["lol"].count()
df["lol"].mean()
df["lol"].median()
df["lol"].max()
df["lol"].min()
df["lol"].sort_values(by = "column_name")
``
