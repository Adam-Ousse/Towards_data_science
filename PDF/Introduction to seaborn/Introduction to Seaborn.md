Seaborn : Statistics + Matplotlib
- easy to use
- works well with pandas
- built on top of matplotlib
```python
#basic setup
import seaborn as sns
import matplotlib.pyplot as plt
#scatter plot
sns.scatterplot(x=X,y=Y)
plt.show()
#coutplot bar plot of the number of times a certain categorical value is in the data
sns.countplot(x=X)
plt.show()
```

### Pandas usage
```python
import pandas as pd
df = pd.read_csv("masculinity.csv") #creates a dataframe object
sns.countplot(x="how_masculine",data=df)# as in x= df["how_masculine"]
plt.show()
```

### Adding a third variable with hue
```python
df = sns.load_dataset("tips")
hue_colors = {"Yes":"black","No":"red"}
#you can use hex code for colors 
# hue_colors= {"Yes":"#FF8343","No":"#179BAE"}
sns.scatterplot(x="total_bill",y="tip",data=df,hue="smoker",hue_order=["No","Yes"])
plt.show()
```
![](Pasted%20image%2020240805143544.png)
### Relational plots
relation between variables can change depending on the subgroups 
### relplot()
stands for relational plots 
sns.regplot() for scatterplot with a linear regression line
```python
#we can create subplots 
sns.replot(x="total_bill",y="tip",data=tips,kind="scatter",col="smoker")
#create 2 subplots first column for smokers and second for non smokers
#to make it vertical we could use row="smoker"
#we can also use thme together
sns.relplot(x="total_bill",y="tip",data=df,kind="scatter",col="smoker",row="time")
plt.show()
#col_wrap=2 means max of 2 columns per row
#col_order=["Yes","No"]
#row_order=["Lunch","Dinner"]
```
![](Pasted%20image%2020240805145101.png)
### Customizing scatter plots 
relation between 2 quantitative variables
#### point size
```python
sns.relplot(..., size="size")#where size is a column in the data sets representing the number of people

```
#### point style
```python
sns.relplot(..., style="smoker")#smokers and non smokers are represented wityh different symbols
```
#### point transparency
```python
sns.relplot(...,alpha=0.4) # 1 is solid, 0 totally transparent
```

### Line plots
usually used to track a variable over time, it's more like a "continuous scatter plot"
```python
sns.relplot(x="hour",y="co2",data=df,kind="line",hue="location",style="location",markers=True)#dashes=False for no dashed lines 
```
for data with multiple y values for the same x values seaborn will automatically aggregate the data, display the mean and the confidence level
to show the standard deviation interval 
`ci="sd"`
we can disable the confidence interval by setting the `ci=None`
```python
# Import Matplotlib and Seaborn

import matplotlib.pyplot as plt
import seaborn as sns
# Add markers and make each line have the same style
sns.relplot(x="model_year", y="horsepower", 

            data=mpg, kind="line", 

            ci=None, style="origin", 

            hue="origin",markers=True,dashes=False)
```
![](Pasted%20image%2020240805175721.png)
## Categorical plots 
involves categorical data (small finite possible values).
It's used to make comparisons between certain groups.
we use
`catplot()` to create different kinds of categorical plots.
### count plots
displays the number of times a categorical value appeared
```python
category_order = ["no answer", "not at all", "somewhat", "very"]
#instead of sns.countplot()
sns.catplot(x="how_masculine", data=masculinity, kind="count",order=category_order)
```
### bar plots
displays the mean (with yerr) of a variable for each group (has a certain categorical value)
```python
sns.catplot(x="day",y="total_bill",kind="bar")
#automatically shows the confidence level (yerr in amtplot) which we cand siable with ci=None
```
### box plot
distribution of quantitative data for each group
```python
order = ["Lunch","Dinner"]
sns.catplot(x="type",y="total_bill",data=df,kind="box",order=order)
#sym="" to not show the outliers it stands for symbol, so u can change the outliers symbols to "o" or" "x" etc..
#by default the whiskers extends to 1.5*IQR ,
#whis=2.0 means 2*IQR or whis = [5,95] means 5th and 95 percentile
#for min max whiskers [0,100] they extend to the min and max values 


```

### Point plots
shows the mean of a quantitative variable for different groups and trhe 95 confidence level , it's like line splots but more discrete.

```python
sns.catplot(x="age_group",y="masculinity",data=dt,kind="point")
#to remove the line linking the groups : join=False
#to use another estimator instead of the mean : estimator=np.median
#capsize=0.2 (adds horizantal lines of width 0.2 at the end of the confidence level)
#ci=None to remove the confidence interval
```

## Changing style
### background and axes:
`sns.set_style()` with values : (defaults tyle white)
- white , dark, whitegrid, darkgrid, ticks
### color of main elements 
`sns.set_palette()` with values 
- for scales with opposite extremes and a neutral mid value : ![](Pasted%20image%2020240805183806.png)
- variables with cntinuous scale (like time or size) ![](Pasted%20image%2020240805183918.png)
- for custome palette `[color names in hex]`
### Scale of plot
to change the scale the plot elements and albels 
`sns.set_context()` from smallest to largest : 
- paper
- notebook
- talk
- poster

## Titles and labels
### Title
#### FacetGrid
```python
g=  sns.catplot(,col="group") # or sns.relpot() , returns a FacetGridobject (can make mutiple plots) ,multple columns 
g.fig.suptitle("Lol test",y=1.03) #adds a title for the whole figure 
g.set_titles("This is {col_name} group)
g.set(xlabel="xlabel",ylabel="ylabel") 
			 #or
g.set_axis_labels("xlabel","ylabel")
plt.xticks(rotation=90) #rotate 90deg
plt.subplots_adjut(top=.93) #the top of figures is 93%
```
#### AxesSubplot
```python
g=sns.barplot()# returns a AxesSubplot object
g.set_title("Lol",y=1.03)

```