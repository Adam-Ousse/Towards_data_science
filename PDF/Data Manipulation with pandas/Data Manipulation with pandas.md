# Dataframes


### Exploratory 
`df.head(n=5)` returns the first n rows of a Dataframe object.
`df.tail(n=5)` returns the last n rows of  Dataframe object.
`df.index` the index of the df rows
`df.columns` the labels of the df
`df.dtypes` types of each column
`df.select_dtypes(inclue=[], exclude=[])` , include and exclude should not overlap, i returns the subset of df columns with the specified types :
	-  for numeric "number", "object" for strings, "datetime" for datetime, "category" for categorical dtypes, "bool" for boleans, "float64" for floats,"int64" for ints.
`df.info()` print's a concise summary of a df
`df.values` numpy representation of the df
`df.axes` returns a list of the axis , indexs and columns
`df.shape` retusna  tuples representing the shape of the df (nb rows, nb columns)
## Sorting and subsetting
### Sorting 
`df.sort_values(by=["col1","col2"..] , ascending=[True,False..],axis=[0,0(default0)],inplace=bool`) sort the values by either one or multiple columns or indexs,

## Subsetting 
### on columns
`df[["col1","col2"]]` returns 
### on rows 
`df[df[col1]<2` returns rows of df satisfying the boolean condition 
#### boolean series 
`df["col"].isin([val1,vl2..])` True if the value is in the passed list
We can't use the `and` and `or` but `&` and `|`
`df["col"].notna()` true if it's not null

### on both 
#### loc
`df.loc[rows,columns_names]`
#### iloc
`df.iloc[0:3,5]` uses indices 
## Descriptive stats
`df.describe(include=[]) ` description of each column, include and exclude same style as .select_dtypes , `include=['O']` for objects (strings)




`df.equals(df1)` True if the dataframes are identical


### Converting a columns to rows using .melt()
```
df = pd.melt(measure, id_vars= "name",value_vars="height")
#... Name value
#... Height 192..
```
opposite of .pivot(index, columns ,values)
