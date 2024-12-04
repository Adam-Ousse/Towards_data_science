# Chapter 1 :
## Summary statistics

The type of data we work with is important, there are 2 main types : 
- Numerical data : 
	- Continuous like speed of planes 
	- Discrete, like number of followers 
- Categorical data :
	- Nominal : as in unordered like married/unmarried
	- Ordinal: as in ordered like: agree strongly/ agree/ indifferent/disagree/disagree strongly
		Categorical data can be represented using numbers
### Measure of center : 
- Mean : $$\bar{x} =\frac{\sum_{i=0}^{N} x_i}{N}$$
- Median : middle point after sorting data
- Mode: most frequent value
Depending on the relative position of the mean to the median we can say that the data is : 
- Right-skewed : if the mean is on the right of the median
- Left-skewed: if the mean is on the left of the median

## Measure of spread:

Variance : average distance between the data points and the mean :
##### Population Variance $$ \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 $$
#####  Sample Variance $$ s^2 = \frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2 $$
	 np.var(msleep['sleep_total'], ddof=1)
(it's an estimate of the actual variance, we use /n-1 to apply the Bessel correction, for a more accurate estimation of the variance (no bias) which is used when we don't know the actual mean but only it's estimation)

##### Population Standard Deviation 

$$

\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}

$$


##### Sample Standard Deviation

$$

s = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2}

$$
Standard deviation (SD) penalizes longer distances between data points and the mean more (because of the ^2) than MAD : 

$$MAD = \frac{\sum_{i=0}^{N} |x_i - \bar{x}|}{N}$$
which penalizes distances equally

### Quantiles : 
 k-th q-quantile is the data value where the cumulative distribution function crosses _k_/_q_. That is, x is a k-th q-quantile for a variable X if

Pr[X<x] ≤ k/q
and
Pr[X ≤ x] ≥ _k_/_q_.

Second quantile is median,

#### Quantiles
They are general terms and can be used to describe any division of the data.

- **Definition**: A quantile is a value below which a certain percentage of data falls.
- **Examples**:
    - **Median (50th percentile)**: Divides the data into two equal parts.
    - **Quartiles (25th, 50th, 75th percentiles)**: Divide the data into four equal parts.
    - **Deciles (10th, 20th, ..., 90th percentiles)**: Divide the data into ten equal parts.
    - **Percentiles (1st, 2nd, ..., 99th percentiles)**: Divide the data into 100 equal parts.

#### Quartiles
Quartiles are specific types of quantiles that divide the data into four equal parts.
- **Definition**: Quartiles are the three points that divide the data into four equal parts.
- **Types**:
    - **First Quartile (Q1)**: The 25th percentile, below which 25% of the data falls.
    - **Second Quartile (Q2)**: The 50th percentile (also the median), below which 50% of the data falls.
    - **Third Quartile (Q3)**: The 75th percentile, below which 75% of the data falls.
#### Interquartile Range (IQR)
The Interquartile Range (IQR) is a measure of statistical dispersion, which is the spread of the data points in a dataset. It is the range between the first quartile (Q1) and the third quartile (Q3).

#### Calculation

The IQR is calculated as follows:
$$IQR = Q3-Q1$$

Where:

-  (First Quartile) is the 25th percentile of the data.
-  (Third Quartile) is the 75th percentile of the data

1. **Identifying Outliers**: Data points that lie below  $Q1-1.5IQR$ or above $Q3 +1.5IQR$  are often considered outliers 
2. **Descriptive Statistics**: The IQR is used to describe the spread of the data in summary statistics.
3. **Box Plots**: The IQR is a key component of box plots, which visually represent the distribution of data.
# Chapter 2 
## Measure of chance : Probability :
### Discrete probability space : 
$$P(event) = \frac{card(ways event cant happen)}{card(possible outcome)}$$
example : Coinflip $P(heads) = \frac{1 way to get heads}{2 possible outcomes heads or tails} = \frac{1}{2}$
#### Sampling :
Sampling from a dataframe : 
```python
db.sample() #one sample aka picking one row from the data
```

Second sampling : 
- without replacement : 
	- the sampled rows are not taken into account again : if we had 3 balls one red and one blue, and one yellow, and we got the red ball in the first sample, then the probability of getting the blue ball would be 50% instead of 33% :
	- ``db.sample(2)``
- with replacement :
	- the sampled rows are taken into account
	- ``db.sample(2,replace=True)``
#### Independent events : 
Two events A and B are independent if the probability of the second event isn't affected by the outcome of the first event : 
$$P(A\cap B) = P(A) * P(B)$$
example :
sampling with replacement, each new sample (event) is independent of the previous one, but without replacement the events are dependent.
#### Discrete distributions 
X in finite or infinitely countable values.
Fonction de repartition , Cumulitive distribution function : 
$$F_X(x)=P(X<=x)$$
the sampled distribution is a bit different of the theoretical distribution, however the more samples we have the close we get to it.
##### Binomial distribution 
###### Bernoulli distribution B(p)
event = {1,0} success or a failure
```python
from scipy.stats import binom
number_coins = 1
p=0.5
binom.rvs(number_coins, p, size=1)#1 binom trial
```
$P(X=1)= p, P(X=0) = 1-p$

```python
from scipy.stats import binom
number_coins = 10
p=0.5
binom.rvs(number_coins, p, size=1)#number of successes in flipping 10 coins, one time => [4] for exp
```
The binomial distribution describes the number of successes in a sequence of *independet* bernoulli trials.
B(n,p) 
$$P(X=k) = C_n^k p^k * (1-p)^{n-k}$$
- n= number of bernoulli trials 
- p = probability of success of one bernoulli trial
$E(X) = n*p$ , $V(X) = n*p*(1-p)$

```python
#probability of certrain number of successes
n_success = 5
binom.pmf(n_success,number_coins,p) # p(X= 5) .pmf stands for Probability Mass Function


```
##### Poisson distribution : 
poisson process, events apear to happen at a certain rate (same avg) but completely at random, like number of people arriving at a restaurant per hour.
The poisson distribution describe the probability of some number of events happening over a fixed period of time.
$P(\lambda)$ where lambda describesthe average numer of events per time interval 
$$P(X=k) = \frac{\lambda^k}{k!} e^{-\lambda}$$
```python
from scipy.stats import poisson
lamb=8
poisson.pmf(5,lamb) #P(x=5)

```
### Continuous distributions 
X in uncountable/continuous values 
##### Uniform distribution : 
$U(a,b)$ 
distribution  :
$$f_{U(a,b)}(x)= \frac{1}{b-a} * 1_{[a,b]}(x)$$
cdf : 
$$F(x) = P(X<=x) = \int_a^x f(x)dx=\frac{1}{b-a}*(min(x,b)-a)$$
$$P(c<=x<=d) = F(d)-F(c)$$
```python
from scipy.stats import uniform
a=0
b=12
uniform.cdf(7,a,b) // P(x<=7) for x following a U(a,b), cdf stands for cumulitive distribution function
```
```python
//generating random samples of uniform distribution 
uniform.rvs(a,b,size=10) // rvs stands for random variates, retusn and array 
```
##### Normal distribution : 
$$N(m,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} exp(\frac{-(x-m)^2}{2\sigma^2})$$
$N(0,1)$ is called the standard normal distribution.
for $X \sim N(0,1)$ the
- $P(|X|<=\sigma$ = 0.68 
- $P(|X|<=2\sigma$ = 0.95
- $P(|X|<=3\sigma$ = 0.997
```python
from scipy.stats import norm
m = 161
std = 7
norm.cdf(154,m,7)
norm.ppf(0.9,161,7) #90% falls undes this valuew hich is 168.97 ppdf stands for percent point function aka the inverse of cdf
norm.ppf(1-0.9,m,std) #90% falls above this values 
```

##### Exponential distribution : 
can describe the probability of time between poisson events 
$\mathcal{E}(\lambda)$ where lambda is the same poisson rate.
$\lambda$ requests per hour => $1/\lambda$ average time between requests 
$$P(X=x) = \lambda 1_{R^+}(x) e^{-\lambda x}$$
```python
from scipy.stats import expon
lamb = 0.5 #0.5 ticket per hour
expon.cdf(1,scale = 1/lamb) # p(wait_time < 1 hour), we pass 1/lamb not the poisson lambda 

```
##### Student's (t-distribution) distribution :
it has a degrees of freedom parameters, lower df => thicker tails aka higher std
df highers => approaches normal distribution
$Y = ln(X)$ where $X \sim N$ 
##### Log-normal distribution :
variable whose logarithm is normally distributed like length of chess games.

### Central limit theorem
 The Central Limit Theorem states that if you have a population with mean $\mu$ and standard deviation $\sigma$, and take sufficiently large random samples from the population with replacement, then the distribution of the sample means will be approximately normally distributed. This can be expressed as:

$$
\frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} \xrightarrow{d} N(0, 1)
$$

where:
- $\bar{X}$ is the sample mean
- $\mu$ is the population mean
- $\sigma$ is the population standard deviation
- $n$ is the sample size
- $N(0, 1)$ denotes the standard normal distribution


### Correlation

#### Correlation coefficient :
a quantity between -1 and 1 that describe how linear is the relationship between 2 variables.
- >0.2 : strong relation
- $\sim 0$ no relation
- negative : x increases => y decreases
```python
import seaborn as sns
sns.scatterplot(x="sleep_total",y="sleep_rem",data=df_sleep)
sns.lmplot(x="sleep_total",y="sleep_rem",data=df_sleep, ci=None)# draws the resulat of a LR model, ci stands for confidence interval
```
```python
df["x"].corr(df["y"])#to calculate the correlation 
```
There are many ways to calculate the correlation 
$$ \rho_{X,Y} = \frac{\operatorname{cov}(X,Y)}{\sigma_X \sigma_Y} $$
where $cov(X,Y) = E[(X-E(X))(Y-E(Y))] = E[XY]-E[X]E[Y]$
 For a sample, the formula is given by: $$ r_{xy} = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i  
- \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i  
- \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i  
- \bar{y})^2}} $$Correlations doesn't not entail nor imply causation, if x is correlated with y, it doesn't necessarily mean x causes y.
![](Pasted%20image%2020240803225343.png)
So: high correlation entails that the 2 variables are associated at the least. 


### Experiments : 
experiment aims to answer : what is the effect of the treatment on the response ? 
Treatment : explanatory/independent variable (x)
Response:  response/dependent variable (y)
E.g. : What is the effect of an ad on the number of products purchases.

- Participants are assigned by researchers to either treatment group or control group
- Groups should be comparable so that causation can be inferred
- else it could lead to confounding
Gold standard ,use randomized controlled trail : random asignment of participants
placebo : resembles treatment, but has no effect, participants will not know which group they're in.
Double blind trial : the person administering doesn't know whether the treatment is real or a placebo.
Observational studies , participants assign themselves, usually based on preexisting characteristics, like when the effect of smoking on cancer, you can't force ppl to start smoking.
so these studies only establic association not causation, because we can't make comparable groups.

logitudinal study : long term observation
cross-sectional study : data from a single snapshot in time
