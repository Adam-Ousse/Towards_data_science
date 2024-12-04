reviewing and cleaning data o derive insights and generate hypotheses.

- First Look : 
	- `df.head(n) or df.tail(n)` to get the first and last n rows respectively
	- `df.info()` to check for missing data (null) and data types
	- `df.value_counts("column")` counting the number of appearances of unique values in a categorical column or `df["col"].values_counts()`
	- `df.describe()` lists some summary statistics for numerical columns
	- `sns.histplot(x="col",data=df,binwidthg=.1)` for visualizing the distribution of a numerical varible
- Data validation :
	- `df.dtypes` for columns types
	- `df["col"]= df["col"].astype(int)` to change the column type
		- str for strings, int for ints, float for floats, dict, list bool, "category" for categories
	- `df.astype({"col_name":"new_type","col_nam2"..})`
	- `pd.to_numeric(df)` to numeric values
	- `pd.to_datetime(df,)`
	- `df[~df["genre"].isin(["genre1","genre2"]]` validating categorical data
	- `df.select_dtypes("number")` for validating numeric data
	- `sns.boxplot(data=df,x="numeric_col")` for visualizing numeric variables
- Data summarization
	- `df.groupby("categorical_column")` to group the data by a col 
		- we can link it with aggregating functions like .mean(), .count() , .sum(), .min and max , .var() and .std() , .median(), .mode() (most appearing value)
		- we can link it with the `.agg()` which can take : 
			- a list containing the functions to apply to each category like `["mean","std"]` and all numeric cols
			- multiple parameters `new_col_name=("col_to_apply","function_name")` which creates a new col the result of the agg function 
- Addressing missing  
	- `df.isna().sum()` number of missing data per column
	- usually we drop rows if they represent less than 5% of the total rows : 
		- `threshold = len(df)*0.05`
		- `columns_to_drop= df.columns[df.isna().sum()<=threshold]` to select columns with a number of missing data less than the threshold
		- `df.dropna(subset=columns_to_drop, inplace=True)` to drop them
	- we impute the rest of missing values with a summary statistic 
		- `left_cols = df.columns[df.isna().sum()>0]`
		- then for each col in left_cols `df[col].fillna(df[col].mode()[0])` or .mean()..etc
	- Imputing by sub-group
		- `group_dic = df.groupby("category_col")["col_to_impute"].mean().to_dict()` (here taken the mean)
		- `df["col_to_impute"].fillna(df["category_col"].map(group_dic))`
- categorical data 
	- `df.select_dtypes("object")` to select non numeric values
	- `df["cat_col"].value_counts()` a df of the number of occurrence of a value
	- `df["cat_col"].nunique()` returns the number of unique values 
	- extracting value from a category : 
		- `df["cat_col"].stf.contains(s)` : 
			- `s= "Scientist|Doctor"` searches for scientist or doctor
			- `s="^A"` means starts with A
			- s="string$" ens  with string
			- `s=".lol"` the . matches any single character except newline
			- `s="\\d{3}` matches 3 consecutive digits `\\s` for whitespace
			- `s="[aeio]` matches any of those characters ^aeio for negated
		- New categories from alot of values : 
			- `new_categories=["excelent","average","failure"]`
			- `excelent= "A+|A|A-"`
			- `average= "B+|B|B-|C+|C-|C`
			- `failure= "^[CDEF]`
			- `conditions = [(df["grade"].str.contains(excelent)),(df["grade"].str.contains(average)),(df["grade"].str.contains(failure))]`
			- `df["new_cat"] = np.select(conditions,new_categories,default="Other")`
- Numeric data 
	- Transforming a column from strings to numeric : 
		- `df["col"].str.replace("to_replace","replacement")`
		- `df["col"]=df["col"].astype("float")`
	- Adding a column of summary statistics by subgroups 
		- `df["new_summary_byècategory"]=df.groupby("category")["based_on_col"].transform(lambda  x: x.mean())` makes a new column where each value is the mean value of "based_on_col" for that category
		- `df[["col","new_summary_bycategory"]].value_counts()` category : summary stats : value count
- Outliers : 
	- outliers normally 75th percentile + 1.5 IQR and 25th - 1.5 IQR
	- `IQR = df["col"].quantile(0.75) - df["col"].quantile(0.25)`
	- `upper = df["col"].quantile(0.75) +1.5*IQR`
	- `lower = df["col"].quantile(0.25)-1.5*IQR`
	- `df["col"]=df[(df["col"]<upper) & (df["col"]>lower)]` removes outliers
	- histograms and boxplots are a great way to find outliers 
	- `df["col"].describe()` is also good comparing the mean  with the max and min
- Patterns over time
	- Datetime columns should be manually declared : 
		- `pd.read_csv("data.csv",parse_dates=["col1","col2"])` 
		- `df["col1"]=pd.to_datetime(df["col1"])`*
		- `df["date"]=pd.to_datetime(df[["month","day","year"]])`
	- extracting info from datetime column 
		- `df["col"].dt.year` .month , .day , dt.weekday 0 to 6
	- lineplots are great to see patterns overtime, 
- Correlations 
	- `df.corr()` returns a dataframe containing the Pearson linear correlation between numeric variables
	- `sns.heatmap(df.corr(),annot=True)` plots a heatmap
	- `sns.pairplot(data=df,vars=["col1","col2"])` scatter plots of different cols with others having the diagonal as a histograms
- Factor relationships (correlation categorical data)
	- `sns.histplot(x="col",hue="categorical_col",data=df)`
	- `sns.kdeplot(x="col",hue="categorical_col",data=df,cut=0,cumulative=False)` estimates the distribution function of the column by category, cut=0 means not going out of the min and max values, for cumulative distribution function cumulative=True
- Considerations for categorical data is the data representative of the population ,=> class imbalance :
	- `df["cat"].value_counts(normalize=True)` sum =1 
	- Cross-tabulation, index and columns are unique values of a column, and the values are the count. 
		- `pd.crosstab(df["source"],df["destination"])`
		- `pd.crosstab(planes["Source"], planes["Destination"],values=planes["Price"], aggfunc="median")` instead of the count it's the median of the Price column for each (source,destination) pairs
- Generating new features :
	- `df["month"]= df["date"].dt.month` .weekday .hour
	- Creating categories from numeric values : 
		- `bins = [df["$"].min(), df["$"].quantile(.25), df["$"].median(), df["$"].quantile(.75), df["$"].max()]`
		- `labels = ["entry","senior", "manager","exec"]`
		- `df["$_category"] = pd.cut(df["$"], bins= bins ,labels=labels)` fills the rows accordingly 
		- `sns.countplot(x="category",data=df, hue="$_category")`
- Generating hypotheses
	- data snooping, excessive EDA 