# Hypothesis Testing

- A frequentist method to make statistical inference 

- Concepts:
    - want to know how extreme our observed data is based on the belief
    - if the probability of having data more extreme than current data is very small, then we could say the belief may not be correct



- Building blocks
    - belief: null hypothesis
    - level of extreme: limiting distribution 
    - probability of having data more extreme: p-value

## Data

Now we could generate random samples to illustrate above concepts.

First use the i.i.d random sample from binomial distribution.



```python
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# binomial setups
p = 0.05
size = 1000

# random sample
samples = np.random.binomial(n=1, p=p, size=size)
```

Theoratically, the variance of a binomial distribution will follow:  
$Var(X) = p \cdot (1-p)$


```python
print(f"Theoratical value: {p * (1-p)}")
print(f"Empirical   value: {samples.var()}")
```

    Theoratical value: 0.0475
    Empirical   value: 0.044791


## Limiting Distribution

We could obtain the limiting distribution mathematically.

Goes from sampling distribution (the distribution of statistic);

To the limiting distribution (the distribution when size goes to infinitely large)

### Sampling Distribution

For random sample $X_i \sim (\mu, \sigma^2)$

the sampled mean will be $\bar{X_n} = \frac{\sum_i X_i}{n}$, where $n$ is the size of sample

Then we will know that:

- $E[\bar X_n] = \frac{1}{n} \cdot n \cdot E[X_i] = \mu$

- $Var(\bar X_n) = \frac{1}{n^2} \cdot n \cdot Var(X_i) = \frac{\sigma^2}{n}$

The sampling distribution of sampled mean will be $\bar X_n \sim (\mu, \frac{\sigma^2}{n})$

### Asymptotic Property

By the C.L.T, we know that:

As $n \rightarrow \infty$, $\bar X_n \sim^d N(\mu, \frac{\sigma^2}{n})$

## How could we understand it empirically?

Empirically, it is like you repeat the random draw over and over agian,
and collect the mean each time. (bootstrapping method)

The histogram of the means will be the sampling distribution.


```python
# simulate the random draws for 500 times

n_simulation = 500
bootstrap_sampled_mean = np.zeros(n_simulation)

for i in range(n_simulation):
    
    bootstrap_samples = np.random.choice(samples, size=len(samples), replace=True)
    
    bootstrap_sampled_mean[i] = bootstrap_samples.mean()
```


```python
sns.histplot(bootstrap_sampled_mean, stat='probability')
plt.title(fr"Sampling Distribution for $\bar X_n$, N={len(samples)}")
plt.show()
```


    
![png](hypothesis-testing_files/hypothesis-testing_11_0.png)
    


Add the PDF for the empirical and theoratical distribution


```python
from scipy.stats import norm
```


```python
smean = samples.mean()
svar = samples.var() / len(samples)
sstd = np.sqrt(svar)
```


```python
sns.kdeplot(bootstrap_sampled_mean, label='empirical', color='salmon')
sns.lineplot(
    x=(x := np.linspace(min(bootstrap_sampled_mean), max(bootstrap_sampled_mean))), 
    y=norm.pdf(x, loc=smean, scale=sstd),
    color='skyblue',
    label='theoratical'
)
plt.title(fr"PDF for $\bar X_n$, N={len(samples)}")
plt.legend()
plt.show()
```


    
![png](hypothesis-testing_files/hypothesis-testing_15_0.png)
    


## Hypothesis Testing

Now suppose we want to know if the mean is $p = 0.075$

We construct the belief, the null hypothesis, $H_0: p = 0.075$

Given the null hypothesis is true we construct the theoratical distribution for sampled mean.


```python
smean = 0.075
svar = smean * (1-smean) / size
sstd = np.sqrt(sstd)
```


```python
x = np.linspace(smean - 3 * sstd, smean + 3 * sstd)
y = norm.pdf(x, loc=smean, scale=sstd)
```


```python
sns.lineplot(x=x, y=y, label='theoratical')
plt.axvline(x=samples.mean(), linestyle='-', color='salmon', label='observed mean')
plt.legend()
plt.show()
```


    
![png](hypothesis-testing_files/hypothesis-testing_19_0.png)
    


Obtain the probability $Pr(X <= observed\_mean | H0)$


```python
norm.cdf(samples.mean(), loc=smean, scale=sstd)
```




    0.3660763734790762



### Compare to the test functions


```python
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_1samp
```


```python
ttest_1samp(samples, popmean=0.075)
```




    Ttest_1sampResult(statistic=-4.181628010126493, pvalue=3.1483606196336255e-05)




```python
proportions_ztest(samples.sum(), len(samples), value=0.075)
```




    (-4.183720393549963, 2.867767991818581e-05)




```python
smean = samples.mean() - 0.075
sstd = samples.std() / np.sqrt(len(samples))
```


```python
sns.lineplot(x=(x:=np.linspace(-4, 4)), y=norm.pdf(x, loc=0, scale=1), label='theoratical')
plt.axvline(x=samples.mean() - 0.075, linestyle='-', color='salmon', label='observed mean')
plt.legend()
plt.show()
```


    
![png](hypothesis-testing_files/hypothesis-testing_27_0.png)
    


### [ToDo] Why the results are different
