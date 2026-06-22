# Important Interview Questions

This file tracks questions marked as **IMP** during project development.

For each selected question, record:

- PDF question number and wording
- Short interview answer
- Relevant PropLens file or module
- Common follow-up or twisted question
- Practice status

## Questions

### Q134. What is the difference between MCAR, MAR, and MNAR?

**Short answer**

- **MCAR:** Missingness kisi observed ya unobserved variable se related nahi hoti.
- **MAR:** Missingness ko available observed variables explain kar sakte hain.
- **MNAR:** Missingness missing value khud ya kisi unobserved reason se related hoti hai.

**PropLens examples**

- MCAR: Random technical error ke kaaran kuch properties ka `facing` missing.
- MAR: Older properties mein furnishing details zyada missing.
- MNAR: Expensive listings mein intentionally `Price on Request`.

**Common follow-up**

Can simple imputation handle MNAR?

No. MNAR may require domain knowledge, missing indicators, sensitivity analysis,
or explicit missingness modelling because simple imputation can introduce bias.

**Practice status:** V1 completed

### Q149. What is Simpson's paradox? Give an example.

**Short answer**

Simpson's paradox tab hota hai jab aggregated data mein ek trend dikhe, lekin
meaningful subgroups mein analysis karne par trend reverse ya disappear ho jaye.
Yeh often confounding variable aur unequal group distribution ki wajah se hota
hai.

**PropLens example**

Overall data mein houses ki average price flats se higher dikh sakti hai. Lekin
individual sectors ke andar flats ki average price houses se higher ho sakti
hai, because houses expensive sectors mein disproportionately listed hain.
`sector` yahan confounding variable hai.

```python
df.groupby("property_type")["price"].mean()

df.groupby(
    ["sector", "property_type"]
)["price"].mean()
```

**Common follow-up**

Is Simpson's paradox a data error?

Not necessarily. Both aggregated and subgroup calculations can be mathematically
correct. The correct interpretation depends on grouping, confounders, sample
sizes, and the business question.

**Practice status:** V1 completed

### Q147. What is a Q-Q plot?

**Short answer**

A Q-Q (Quantile-Quantile) plot observed data quantiles ko theoretical
distribution, usually normal distribution, ke quantiles se compare karta hai.

```python
import scipy.stats as stats

stats.probplot(
    df["price"].dropna(),
    dist="norm",
    plot=plt,
)
```

**Interpretation**

- Straight line ke paas points: approximately normal
- Curved ends: tail differences
- S-shaped pattern: skewness or different tail behaviour
- Isolated distant points: possible extreme observations

**PropLens relevance**

Property prices aur model residuals ki normality, skewness, and tail behaviour
inspect karne ke liye use ho sakta hai.

**Common follow-up**

Must every ML feature be normally distributed?

No. Most ML models do not require every feature to be normal. The assumptions
of the selected model, especially residual assumptions in statistical models,
matter more.

**Practice status:** V1 completed

### Q140. What is kurtosis?

**Short answer**

Kurtosis distribution ki tail heaviness measure karti hai, yani extreme values
kitni likely hain. Pandas commonly excess kurtosis report karta hai:

- `0`: Normal-like tails
- `> 0`: Heavy tails and more extreme observations
- `< 0`: Light tails and fewer extreme observations

```python
df["price"].kurt()
```

**PropLens relevance**

Property prices mein high kurtosis luxury properties, mixed market segments, ya
data-quality problems ki wajah se aa sakti hai. High kurtosis automatically data
error prove nahi karti; domain validation zaroori hai.

**Common follow-up**

Is kurtosis simply a measure of how peaked a distribution is?

That explanation is incomplete. Modern interpretation focuses primarily on tail
heaviness and the likelihood of extreme observations.

**Practice status:** V1 completed
