`statsmodels` are optimized for insights
`scikit-learn` are optimized for prediction

Fitting a linear regression model 
```python
from statsmodels.formula.api import ols #stands for ordinary least squares
model =old("y_column ~ x_column",data=df) # "y ~ x + 0" for no intercept (gives u a value of intercept but consider it a 0)
model = model.fit()
print(model.params)
```

Predicting data 
extrapolating : making predictions outside the range of observed data.
```python
model = ols("mass ~ length", data= df).fit() #mass is the response variable and length is the explanatory variable
explanatory_data = pd.DataFrame({"length" : np.arange(140,200)})
explanatory_data = explanatory_data.assign(mass=model.predict(explanatory_data))
#we get a dataframe with a column for x and y
sns.scatterplot(x="length",y="mass",data=explanatory_data)
```

Getting the fitted data : 
`model.fittedvalues` returns a pandas dataframe
Getting the risiduals (difference from the tagert)
`model.resid` returns a pandas series 
Getting a summary on the model
`model.summary()`


Regression to the mean is an important idea in statistics, which stipulates that extreme cases converge to the average cases on the long run. for example very tall fathers will have on average shorter kids, and very short fathers will have a bit taller kids.

Evaluating a model 
r-squared when we use only one explanatory variable and R-squared if we have multiple variables
we can acces it via `model.rsquared` , which means that r-squares % of the explanatory variables explain the response var. 
In the case of on explanatory variable it is equal to `df["x"].corr(df["y"]) **2 ` correlation between explanatory and response variables squared.

MSE : mean squared error accessible via `model.mse_resid` => RSE = sqrt(MSE)
we can retrieve the RSE residual standard error by taking the square root of MSE
	RSE : `sqrt(sum(residual **2) / degrees of freedom )` when degrrees of freedom is the length of the data - number of coefficients
	RMSE : Root mean s quared error /length instead of degrees 

Residuals are generally normally distributed, and we want them to have a mean of 0
Key metrics for evaluating model fit were introduced:

- **Coefficient of Determination (R-squared):** You learned that R-squared measures the proportion of variance in the response variable that can be predicted from the explanatory variable. An R-squared value of 1 indicates a perfect fit, while 0 means the model does not improve prediction over using the mean of the response variable. This metric helps in understanding the strength of the relationship between your model's inputs and outputs.
    
- **Residual Standard Error (RSE):** RSE quantifies how much predictions deviate from actual values, providing a measure of the typical size of the residuals. You discovered that a lower RSE indicates a model that more accurately predicts the response variable. The RSE is particularly useful because it is in the same units as the response variable, making it intuitively easier to understand the average error the model might make.
    
- **Mean Squared Error (MSE) and Root-Mean-Square Error (RMSE):** While MSE is the squared residuals' mean, RMSE is its square root, offering another perspective on prediction accuracy. However, you learned that RSE is generally preferred for model comparison because RMSE does not adjust for the number of model coefficients.
Visualizing model fit 
Residuals vs fitted : (resid plot)
residuals = f(fitted values) , if the residuals are normally distributed with mean 0 the graph should closely follow y=0
```python
sns.residplot(x="length",y="mass", data=df, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
#which is the same as : 
sns.regplot(data=df,x="length",y="residuals",lowess=True)

```
Q-Q plot 
sample quantiles from the data set = f(theoretical quantiles), if they gfollow the 45° line it is normally distributed.
```python
from statsmodels.api import qqplot
qqplot(data=model.resid,fit=True,line="45") #the 45° line

```
Scale-location plot
sqrt(standarized residuals) = f(fitted values) , checks whether the size of residuals increase 
```python
#extract the normalized residuals 
model_norm_resid = model.get_influece().resid_studentized_internal
sns.regplot(x=np.sqrt(np.abs(model_norm_resid)),y=model.fittedvalues,lowess=True)

```
in a good model the size of residuals shouldnt change much as the fitted values change.

Leverage and influence 
leverage : measure of how extreme the explanatory variables are
Influece : measures how much the model changes if we left the observation about equal residual vs leverage
```python
summary = model.get_influence().summary_frame()
df["leverage"] = summary["hat_diag"]
#cook's distance
df["cooks_dist"] = summary["cooks_d"]

```

Logistic regression 
```python
from statsmodels.formula.api import logit
model = logit("y ~ x", data = df).fit()
sns.regplot(x="x",y="y",logistic=True)
```
turning probability >0.5 to 1 else 0 
```python
prediction = model.predict(explanatory_data)
df["churn"] = np.round(prediction)
df["odds_ratio"] = prediction_data/(1-prediction_data)
```

To evaluate the logistic regression model w elook  at the confusion matrix
false positive : predicted as true but it's false
false negative : predicted as false but it's true

```python
actual_response = churn["has_churned"]
predicted_response = np.round(md_recency.predict())
outcomes= DataFrame({"actual":actual_response, "predicted":predicted_response})
outcomes.values_counts(sort=False) #confusion matrix
# or automatically generated : 
conf_matrix =md_recency.pred_table()

```
visualization 
```python
from statsmodels.graphics.mosaicplot import mosaic
mosaic(conf_matrix)
```

$$accuracy = \frac{TN + TP}{TN+FN+FP+TP}$$
$$sensitivity = \frac{TP}{FN+TP}$$
$$specificity = \frac{TN}{TN + FP}$$
higher => better
