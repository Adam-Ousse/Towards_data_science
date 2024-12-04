### one plot
pyplot submodule is the oop interface.
```python
import matplotlib.pyplot as plt
fig,ax = plt.subplots() # creates 2 objects, figure object is a container that holds evertyhing in the page, meanwhile the axis hold the data.
plt.show()#figure witgh empty axis
```

```python
#plotting command 
ax.plot(x,y)
plt.show()
```
```python
#customizing the plots
ax.plot(x,y,marker="o")#adds a point marker, it's used to indicate that the data visualized is actually discrete , and the lines are just linking between them.
```
[markers](https://matplotlib.org/stable/api/markers_api.html) : 
- "o" : point
- "v": triangle up
- "^" : triangle down
- ">": triangle right
- "<": triangle left
- "s" : square...
```python
#line style
ax.plot(x,y,linestyle="--") #dashed lines
```
[linestyle](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html) : 
- "solid"
- "dotted"
- "dashed"
- dashdot
- "None" : no lines
```
#color 
ax.plot(x,y,color="r)#red lines
```
[colors](https://matplotlib.org/stable/gallery/color/named_colors.html)  :
- "r"
- "b"
- "g"
- "c"
- "m"
- "y"
- "k" (black)
- "w"
Set_ :
```python
#set_....
ax.set_xlablel("Time in NY")
ax.set_ylabel("Average temperature in NY")
ax.set_title("Weather in NY)
plt.show()
```
### small multiples plots 
```python
#for visualizing for example the precipitation in multiple cities
n,m = 3,2
fix,ax = plt.subplots(n,m,sharey=True) # n rows, m columns
#sharey ensures that all subplots have the same yaxis range of values
#ax is not longer one axe but an array of axis with the shape of (n,m)
ax[0,0].plot(x,y)
#when working with n or m =1, ax is a one dimensional array therefore we can use ax[0], ax[1] .plot() etc..


```
### Plotting time-series 
```python
import pandas as pd
df=pd.read_csv("data.csv")
df["time"] = pd.to_datetime(df["time"],format="%Y-%m-%d")
df.set_index("time",inplace=True)
#or 
#climate_change = pd.read_csv("climate_change.csv",parse_dates = ["date"], index_col="date")
#this reads the data, parses the time columns as date, and set it's index to the date 
ax.plot(df.index, df["co2"])
plt.show()
sixties = df["1960-01-01":"1969-12-31"] #sixties

```
Plotting, plots of different variable according to same time
We need to set different scales to each plot
```python
#to do that we first plot the first 
ax.plot(x, y1,color="b")
ax.set_ylabel("y1",color="blue")
#then we create a twin, that shares the x axis but not the y axis
ax2 = ax.twinx()
ax2.plot(x,y2,color="r")
ax2.set_ylabel("y2", color="red")
#we can also distinguish them using the colors of y ticks 
ax.tick_params("y",colors="blue")
ax2.tick_params("y",colors="red")
plt.show()

```

Annotating time-series data aka drawing an arrow to part of a plot and adding text.
```python
ax.annotate(">1 degree", xy= (pd.Timestamp("2015-10-07"),1), xytext=(pd.Timestamp("2008-10-07"),-0.2), arrowprops={"arrowstyle":"->","color":"gray"})
# arrowprops stands for arrow propreties,which defines the propreties of the arrow linking the text (xytext) and the point of annotations (xy)
```

### Quantitative comparisons : 
#### bar-charts : value of a variable in different conditions 
like number of medals for different countries
```python
ax.bar(medals.index,medals["Gold"],label="Gold")
#stacked bar chart
ax.bar(medals.index,medals["Silver"], bottom = medals["Gold"],label="Silver")
ax.bar(medals.index, medals["Bronze"],bottom = medals["Gold"] + medals["Silver"],label="Bronze")
ax.set_xticklabvels(mdels.index, rotation=90)
ax.legend()
plt.show()
```
#### Histograms : distribution of values of a variables
numbers of occurrence of values within bins, 
```python
n =10 #by default
#or we can provide a list delimiting the bins : 
#bins = [150,160,...]
ax.hist(mens_rowing["Height"],label="Rowing",bins=10,histtype="step")#histype by default is bar, aka filled rectanges, step on the other hand are just the thin lines
ax.hist(mens_gymnastics["Height"],label="Gymnastics",bins=10,histtpe="step)
ax.legend()
```

### Statistical plotting
#### Adding error bars to bar charts 
```python
ax.bar(x,y,yerr=y.std())# y error 
```
#### Adding error bars to plots
```python
ax.errorbar(x,y,yerr=yerr) # lineplot
```
#### Boxplots
```python
ax.boxplot([mens_rowing["height"],mens_gymnastics["height"]])
ax.set_xticklabels(["Rowing","Gymnastics"])
#yellow line : median
#edge of the boxes :  Interquartile Range, 25 and 75 percetile
# the wiskers : 1.5 IQR off of the IQR range 
# dots : outliers , out of the range where 99% of data should be (considering a gaussian),   $Q1-1.5IQR$ or above $Q3 +1.5IQR$  
```

### Scatter plots : comparison of different variables across observations : bi-variate-comparaison
```python
ax.scatter(climate["co2"],climate["temp"],c = climate.index) #c = continuous color, here taken as the time, dark blue points are earlier, bright yellow newer observations

```
![](Pasted%20image%2020240804190557.png)

### Plot style : 
```
plt.style.use("ggplot") #R library style
#if color is important consider choosing a colorblind-friendly options like "seaborn-colorblind"
#if it will be pronted b&w : "grayscale"
```
[styles available ](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)
#### saving plots  :
```python
#control size of fig
fig.set_size_inches([5,3])
fig.savefig("image.png",dpi=300) #higher dpi => higher resolution
fig.savefig("image.jpg",quality=80)
fig.savefig("image.svg")
