## Introduction to sampling
### Sampling basics
Population  : is the complete dataset
Sample      : is a subset of the population
`df.sample(n=5)` to get a sample of 5 rows , rows from the original df only appear once to change that we can set `replace=True``

population parameter suhc as mean : `np.mean(df['column'])`
point estimate : `np.mean(sample['column'])`

If the sample isn't representative of the population we talk about sampling bias.
convenience sampling : selecting the easiest population members to reach like : 
- taking the first 10 rows of a dataset instead of sampling 10 randomly we can visualize the difference through a histogram. 
### Pseudo-random number generator
for true random generation we usual physical processes :
- atmospheric noise (random.org)
- radioactive decay (hotbits)
but it's expensive and slow
So we use pseudo-random number generator: each number is calculated from the previous one, the seed is the starting point of the sequence.
numpy has many methods for generating pseudo random number from statistical distributions : 
- `np.random.seed(0)` to set the seed
- `np.random.rand()` to generate a random number between 0 and 1
- `np.random.beta(a,b)` to generate a random number from a beta distribution with parameters a and b`
- `np.random.normal(mean,std,size)` to generate a random number from a normal distribution with mean and std
- `np.random.choice(a,size,replace=True)` to generate a random sample of size from a
- `np.random.shuffle(a)` to shuffle the array a
- `np.random.binomial(n,p,size)` to generate a random number from a binomial distribution with n and p
- `np.random.poisson(lambda,size)` to generate a random number from a poisson distribution with lambda
- `np.random.exponential(lambda,size)` to generate a random number from an exponential distribution with lambda
- `np.random.geometric(p,size)` to generate a random number from a geometric distribution with p
- `np.random.hypergeometric(ngood,nbad,nsample,size)` to generate a random number from a hypergeometric distribution with ngood,nbad,nsample
- `np.random.logistic(mean,scale,size)` to generate a random number from a logistic distribution with mean and scale
- `np.random.lognormal(mean,sigma,size)` to generate a random number from a lognormal distribution with mean and sigma
- `np.random.chisquare(df,size)` to generate a random number from a chi-square distribution with df

## Sampling methods
### Simple random sampling
each member of the population has an equal chance of being selected (uniformly at random) `df.sample(n=5,random_stae=sed)`
### Systematic sampling
we select a random starting point and then select every kth element, the tricky part is finding the right interval k to avoid bias.
$interval = \frac{population size}{sample size}$
`sampeled = df.iloc[::interval]`
Trouble with systematic sampling is that features could be correlated with the order of the population., aka the index :
```python
df_with_index = df.reset_index()
df_with_index.plot(x="index",y="column",kind="scatter")
```
so only safe when there s no pattern in the index
to avoid this we can shuffle the dataset
``df=df.sample(frac=1)` 
`df_with_index = df.reset_index(drop=True).reset_index()`
### Stratified sampling
sample a population made-up of distinct subgroups (strata) separately, then combine the results.
`df.groupby("column").sample(frac=0.1,random_state=seed)` the sample proportion per group is now closer to that of the original population.
### weighted sampling
we can assign weights to the population members, the probability of selecting a member is proportional to its weight.
`weight= np.where(df["col"]==group, 2,1)` => twice as likely to be sampled
`df.sample(frac=0.1,weights=weight,random_state=seed)`
check proportions : 
`df["col"].value_counts(normalize=True)`
This sort of sampling is very common in political polling
### Cluster sampling
we divide the population into clusters (randomly sample from the population's subgroups), then randomly sample some clusters and include all members of the selected clusters.
`clusted = random.sample(df[column].unique(),k=n)`
`filtered = df[df[column].isin(clusters)]`
	`filtered[column]= filitered[column].cat.remove_unused_categories()`
`sample = filtered.groupby(column).sample(frac=0.1,random_state=seed)`


## Sampling distributions 
### Sampling distribution of the mean
The larger the sample size the closer the sample mean is to the population mean.
Relative errors (absolute value):
$$ 100 *\frac{|sample mean - population mean|}{population mean}$$
![](../../Pasted%20image%2020241130134548.png)
### Central limit theorem
The sampling distribution of the sample mean will be normally distributed for large sample sizes.
$$\bar{X} \sim N(\mu,\frac{\sigma}{\sqrt{n}})$$
where $\mu$ is the population mean, $\sigma$ is the population standard deviation and n is the sample size.

## Approximate sampling distributions
expand grid : 4dice
```python
dice = pd.expand_grid({"die1":np.arange(1,7),"die2":np.arange(1,7),"die3":np.arange(1,7),"die4":np.arange(1,7)})
					   # all possible outcomes
dice["mean"]=dice.mean(axis=1)
#mean value takes in discrete values => barplo instead of hist
dice["mean"]= dice["mean"]].astype("category")
dice["mean"].value_counts(sort=False).plot(kind="bar"))
```
100 dice => 6^100 nearly the number of atoms in the universe

### standard errors :
for population : `df[col].std(ddf=0)` , np.std(sampling_distribution_5,ddof=0)
for sample : `sample[col].std(ddf=1)`
standard deviation of the sampling distribution of the sample mean : $\frac{\sigma}{\sqrt{n}}$
	important tool for hypothesis testing and confidence intervals, 

## Bootstrapping
resampling with replacement, the sample size is the same as the original sample size.
- sampling : going from population to a sample
- bootstrapping : going from a sample to building  a theoretical population
`df = df.reset_index()`
`s =df.sample(frac=1,replace=True,random_state=seed)`
`s["index"].value_counts()` the index is repeated 
bootstrapping process :
1. make a sample with repetition of the same size
2. calcluate the statistic of interest
3. repeat 1 and 2 many times
=> bootstrap distribution
- bootstrap distribution is very close to the original sample mean, which is not always good the original sample might not be representative of the population.
- so it can't correct any biases from sampling
- np.std(bootstrap_distn, ddof=1) x sqrt(n) is the standard error of the sample mean => good for estimating the std 



## Confidence intervals
"values within one standard deviation of the mean" is a confidence interval
- starting from the sample mean (our prediction) we can build a confidence interval around it
- the interval is centered around the sample mean
like : 25°C (20°C,30°C) or 25 +- 5°C => 25°C is the point estimate, 20°C and 30°C are the confidence interval
	(mean-np.std(ddof=1), mean+np.std(ddof=1)) 
Quantile method: 
	calculating quantiles : 
		- `np.quantile(df,0.025)` 2.5% quantile
		- `np.quantile(df,0.975)` 97.5% quantile
		
Standard error method

- calculate the sample mean 
- calculate the standard error of the sample mean : np std ddof 1
- `scipy.stats.norm.ppdf(quantile, loc=0,scale=1)` to get the z-score
- `lower = norm.ppf(0.025,loc=sample_mean,scale=standard_error)`
- `upper = norm.ppdf(0.975,loc=sample_mean,scale=standard_error)`
- `confidence_inteval = (lower,upper)`
$$IC = \bar{x} \pm Z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

`print(np.quantile(cv_results, [0.025, 0.975]))`