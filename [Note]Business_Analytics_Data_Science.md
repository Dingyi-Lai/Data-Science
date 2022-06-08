Github Repo: https://github.com/Aubreyldy/bads.git
# Introduction
## What is it?
Analytics is the use of data, IT, statistical analysis, quantitative methods, and mathematical or computer-based models to help managers ==gain improved insight about their business operations and make better, fact-based decisions.==(Evans, 2013)

## The scopes of business analytics
![[The Scope of Business Analytics.png]]
From to inform decision-makers to to automate decision-making (inform -> analyze -> act)

## The analytics process model
![[The Analytics Process Model.png]]

## Big data defined
![[Big data defined.png]]

# Foundations of Descriptive Analytics
## Characteristics
- Data set with several features and no target variable
Find structure / patterns in the data
- Multiple forms
	- Clustering
	- Dimensionality reduction
	- Association rule mining
	- Sequence rule mining
- Scalable as plain data is easily available
- Often hard to ensure that detected patters are relevant to the business
- Inform decision-making but do not recommend concrete actions
## Cluster Analysis
### Hierarchical
- Agglomerative
- Divisive
### Non-hierarchical (eg. K-means)
- Exclusive (eg. normal K-means, assigns every case to exactly one cluster)
- Non-exclusive (eg. Gaussian mixture models, GMM, mixture of k Gaussians)
### Distance and its measurement
#### Principle
- Maximize intra-cluster homogeneity
- Maximize inter-cluster heterogeneity
#### Methods
- Numeric varaibles: Euclidian/Manhatten/Generalization/Angle (Cosine distance)
- Nominal variables
- Text(Hamming distance, eq/ Levenshtein distance, uneq)
- Graphs, time series, gene strings, streams, etc
- General notion of similarity/distance
# Foundations of Predictive Analytics
## Principle: Labeled data
Model development (aka estimation, building) is the process in which the learning algorithm crafts the model in such a way that some measure of the agreement (aka fit; antonym error, loss) between the model and the data is maximized.
## Supervised learning
Model specification -> estimation -> forecast
## Case Study: Product Return Management in E-Commerce
![[615865.ipynb]]
# Explanatory Data Analysis (EDA) & Data Preparation
## Type of Data
### Numerical / Continuous
### Categorical / Discrete
#### Binary
#### Nominal
#### Ordinal
## Data Preparation Process
![[Data Preparation Process.png]]
### Data cleaning (noise / missing values / outliers) -> (detect / correct / keep / delete / replace / impute / group / scale)
### Data pipeline construction
#### Basic cleaning of null values, duplicates and outliers
#### Encode variables in the most appropriate way
- Numeric data
	- Log transformation
	- Box Cox transformation
	- Yeo-Johnson transformation (`stats.yeojohnson(feature)`)
- Nominal data
	- One hot encoding
	- WoE

#### Ensure variables fit statistical assumptions/model requirements
```python
from scipy import stats

stats.normaltest(feature)

from statsmodels.graphics.gofplots import qqplot

qqplot(feature, line='s')
plt.show()
```
#### Filter out variables with low predictive power (WoE)
- Weight of evidence encoding or WoE can be used to check if a variable has predictive power and can replace categories with that predictive power.
- apply the WoE transformtion also to a continuous variable after binning that variable.
- WoE can be more appropriate once the number of dummy variables could start to get very high. This high dimensionality could be a large concern for overfitting.
- Weight-of-Evidence coding (WoE)
	- Information Value
	- Adjusted Weight of Evidence
```python
import scorecardpy as sc
bins = sc.woebin(train_df, y="BAD", x='CLAGE')

sc.woebin_plot(bins)

sc.woebin_ply(train_df, bins) # if you would like to replace the variable with its WoE, you can do so using the woebin_ply function
```
#### One-hot encode categorical variables (dummy variables)
```python
train_len = len(X_train)

X_full = pd.concat(objs=[X_train_scaled, X_test_scaled], axis=0)
X_full = pd.get_dummies(X_full, columns = ['REASON', 'JOB'], drop_first=True)

X_train_dummies = X_full[:train_len]
X_test_dummies = X_full[train_len:]

```
#### Double check the model only includes appropriate variables
- Pearson Correlation (continuous + continuous)
- Fisher Score (either the target or variable in question is binary)
- Analysis of Variance (ANOVA, more than 2 categories with a continuous target)
- Chi-Square Analysis (both the feature of interest and the target are both categorical variables)
- Cramer's V (both the feature of interest and the target are categorical variables)
- Information Value (a categorical variable and a binary target)
- Information Gain / Entropy: both a discrete feature and discrete target.
### Overview table

| Goal | Possible Code |
| --- | --- |
| Get data types of each column | `df.dtypes` |
| Select all float columns of a df | `df.select_dtypes(include='float64')` |
| Create min-max scaling or normalization object | `MinMaxScaler()` |
| Create standardization object | `StandardScaler()` |
| Modify `X` with a Box Cox transformation (positive values only) | `stats.boxcox(X) ` |
| Modify `X` with a Yeo Johnson transformation | `stats.yeojohnson(X)` |
| Test if a variable is normally distributed (p-val > 0.05 points to normality) | `statsnormaltest(X)` |
| Create QQ plot for `X` (lines should correspond) | `qqplot(feature, line='s')` |
| Get WoE weights of `feature_i` | `scorecardpy.woebin(train_df, y="target", x='feature_i')` |
| Plot weights of `woebin` | `scorecardpy.woebin_plot(woebin_obj)` |
| Replace the feature calculated in `woebin` with its results| `scorecardpy.woebin_ply(X_train, woe)` |
| Pearson's coefficient | `stats.pearsonr(X, y)` |
| Calculate Chi2 value for X and y | `chi2(X, y)` |
| Get dummy values for `feature` with prefix `name_` | `pd.get_dummies(X.feature, prefix='name')` |
| Create forward variable selection object | `SequentialFeatureSelector()` |
| Returns `X` of only selected features | `sfs_model.transform(X_train)` |
| Create backward variable selection object | `RFECV()` |
| Use fitted RFECV model to predict on test set | `rfecv_model.predict(X_test)` |
| Create object which uses `fun` on cols in `col_list` and keeps unchanged columns | `ColumnTransformer(transformers=('step_name', fun(), list_cols), remainder='passthrough')` ** |
| Create pipeline object | `Pipeline(('step_name', fun))` ***|
| Pass X_train through pipeline for fitting | `pipe_name.fit(X_train, y_train)`|
| Pass X_test through pipeline for prediction | `pipe_name.fit(X_test)`|


** In `ColumnTransformer`, you can add a list of transformers of that format in the `transformer` argument

*** In `Pipeline`, you can add a list of steps of that format as the first argument. It is encouraged to put your classifier in here as the last step

# Algorithms for Supervised Learning
## Differences between supervised learning algorithms
### Support of continuous vs. binary vs. multinomial target variables
### Underlying assumptions
### Parametric and non-parametric approaches
### Representational capacity
### Ensembles versus single models
## Logistic regression
Logistic regression is equivalent to linear regression using the
log-odds as (new) target variable.
## Decision tree learning
- Rule-based approach to partition data set into homogeneous sub-groups
- Branching objectives
	- Increase purity of the data
	- Avoid splits that apply to only a few examples
- Common choices of the impurity function to calculate gain
![[impurity function.png]]
- Example
	- C4.5: entropy
	![[C4.5 algorithm.png]]
	- CART: Gini
	![[CART algorithm.png]]
- Splitting criteria for regression trees
![[Splitting criteria for regression trees.png]]
- Decision tree pruning by early stopping (pre-pruning & post-pruning)
# Prediction Model Assessment
## Accuracy
### Regression
![[Measurement of Predictive Accuracy in Regression.png]]
### Classification
- Common metrics
![[Measurement of Predictive Accuracy in Classification.png]]
- Construction of the ROC Curve
![[Construction of the ROC Curve.png]]
- The Brier Score (MSE)
![[The Brier Score.png]]
## Scalability
### Consumption of time resources
### Time needed to build model (training time) and to generate predictions
Both time factors differ substantially across algorithms
### Consumption of memory resources
### Sensitivity with respect to hyperparameters
### Parallelization important
- model building
- model tuning
## Robustness 
### Real-world data is noisy
- Missing values
- Erroneous data entries
- Wrong labels
- Irrelevant / correlated attributes
### Real-world phenomena change over time
- Concept drift
- Model recalibration versus re-estimation 
### How to these factors affect the model?
- During model building
- After model building
## Comprehensibility 
### Alternative terms: interpretability, transparency, white-box (vs. black-box) model
### Global / Local interpretability
- Global: How do covariates govern predictions
- Local: How was the prediction of a specific observation determined by covariate values
### Prediction versus insight and correlation versus causality
## Justifiability
### Does the way in which attribute values affect predictions agrees with prior beliefs or business rules?
### Credit risk example
Test: does WOE show this trend
## Calibration
### Feature of probabilistic predictions
### Credit Scoring Example
- Model makes risk forecasts for 100 credit applications
- Forecasts are all the same and predict default of 90%
- Then, we should eventually observe 90 actual defaults
## Simulation of real-life application of the model
Performance = f(training error, model complexity)
### Resubstitution estimate
- well-established for explanatory models
- inappropriate for predictive models
### Split-sample method
![[Split-Sample Method.png]]
### Cross-validation
![[N-Fold Cross-Validation.png]]
## Model selection: learning curve analysis
### Grid search
### Cross-validation (CV)
# A Primer of Statistical Learning
## Bayes Theorem
## The Problem of Overfitting
- The point of predictive modeling is generalization
- Overfitting means that a model also embodies the idiosyncratic noise of the training sample
## The Bias-Variance Trade-Off
Bias and variance both decrease predictive accuracy
![[Bias-Variance Trade-Off.png]]
## The Curse of Dimensionality
### In tendency, it is easier to separate classes in higher dimensions
### High dimensionality increases sparseness of attribute space
### All classifiers construct a class boundary. Finding such boundary is difficult, if most areas of the attribute are virtually empty
### Even relatively simple classifiers may overfit when dimensionality is high
## Regularization
### In the face of overfitting problem, regularization revises practices to estimate models
### Balance between two conflicting objectives
#### low training error
Introducing bias to decrease variance, and error
#### low complexity
Penalizing model complexity
### Implementation
#### LASSO penalty: $L_1(\hat{\beta}) = \sum^p_{j=1}|\hat{\beta}_j|$
LASSO complicates model estimation but gives sparser models
#### Ridge penalty: $L_2(\hat{\beta}) = \sum^p_{j=1}\hat{\beta}_j^2$
Ridge imposes stronger penalty on (very) large coefficients
#### Elastic net penalty
$L_{enet}(\hat{\beta}) = \frac{1-\alpha}{2}\sum^p_{j=1}\hat{\beta}_j^2+\alpha\sum^p_{j=1}|\hat{\beta}_j|$
### Regularized logistic regression
![[Regularized logistic regression.png]]
# Ensemble Learning
Combine multiple models to raise predictive accuracy
- Multi-step modeling approach
	- Develop a set of (base) models
	- Aggregate their predictions
- Goal is to raise predictive accuracy
	- Models learn different patterns from the same data and can complement each other
	- Consider simple trees with different rule sets
- Differences across approaches
	- How base models are developed
	- How base model forecasts are combined
	- Whether modeling pipeline involves base model pruning
## Bagging and Random Forest
### Homogeneous ensemble strategy
### Bootstrap sample: random drawn sample ==with replacement==
### Every bootstrap sample provides one base model
### Ensemble prediction calculated as simple average over base model predictions
### Random forest: Bagging + draw a random sample of attributes while growing an individual tree (mtry)
### The success of an ensemble depends on the strength of and the diversity among the base models
### Better to use larger forest; reduce bootstrap sample size if needed
## The Boosting Algorithm
### The Boosting Principle: additive modeling
- Begin with a simple model (often called weak learner)
- Calculate the errors of that model
- Build a second model that ‘corrects’ those errors
- Combine the first and second model to form an ensemble
- Continue with
	- Building and adding base models (one at a time)
	- That ‘correct’ the errors of the current ensemble
	- Until some stopping condition is met (typically max. iterations)
### Family Tree of Boosting Algorithms
- Adaptive boosting (Adaboost) by Freund & Schapire (1997): weights data points
- Gradient boosting (GBM) by Friedman (2001, 2002): fits base models to residuals/neg. gradients: First-order algorithm that finds a local minimum of a function by iteratively changing solutions proportional to the negative of the gradient (or of the approximate gradient) of the function at the current point.
- Extreme gradient boosting (XGB) by Chen & Guestrin (2016)
- Latest revisions
## Heterogeneous Ensemble Learning
### Ensemble Learning Strategies
#### Homogeneous ensembles
- Inject diversity at the data level
	- Drawing training cases at random (e.g., Bagging)
	- Drawing variables at random (e.g., Random Subspace)
- Use the same algorithm for base model production
#### Heterogeneous ensembles
- Inject diversity at the algorithm level
	- Different prediction methods for base model production (ANN, SVM, …)
	- Different meta-parameter settings per prediction method
- All methods receive the same training data
- Also called multiple-classifier-systems (if dealing with classification)
### Dimension Pruning Strategy
### Dimension Prediction Strategy
### The Stacking Algorithm
pools predictions by a 2nd stage model
# Feature Engineering and Selection
## Feature Engineering: Create informative variables for an analytical model
### Truncation of outliers (z-score, inter-quartile range)
### Scaling of numeric variables
#### Z-transformation
#### Min/Max scaling
### Categorization of numeric variables and re-grouping
#### Equal width/frequency binning
#### Supervised discretization (decision trees, chi-square analysis)
### Coding of categorical variables
### Ratio variables (very common in finance)
### Aggregation and trend variables
### Transform feature values to increase normality and stabilize variance
#### Logarithmic transformation
#### Box Cox transformation
#### Yeo Johnson Transformation
#### Shift values if transformation is defined for $x>0$ only
#### Weight of evidence (WOE) coding
- Measures association with target variable per category level
- Pitfalls
	- Sparsely populated levels: include artificial data points and adapt WOE calculation (Zdravevski et al., 2011)
	- Novel levels: set WOE = 0 for novel level -> average risk (Moeyersoms & Martens, 2015)
## Feature Selection: Reduce number of inputs in an analytical model
### Filter approach
![[Filter Methods for Feature Selection.png]]
- Uses statistical indicator
- Assess one variable at a time
### Wrapper approach
- Uses prediction model
- Iteratively build & assess model with different variables
- Examples: forward selection, backward elimination
### Comparison
![[Filters versus Wrappers.png]]
Hybrid strategy often useful
### Embedded Methods
Often implemented via regularization penalty
# Model Interpretaion and Diagnostics
## Desiderata of an Interpretable Model Based on Lipton (2016)
- Trust
- Causality
- Transferability
- Informativeness
- Fairness
## Global Explanation Methods
### Surrogate Models / Pedagogical Rule Extraction
- Use a white-box model to explain a black-box model
- Trade-off between quality of explanation and interpretability
### Two-Stage Models / Multi-Stage Models
- Combine interpretable white-box model with black-box model
### Permutation-Based Feature Importance
- A feature is important if model accuracy decreases when it is not available
- Pros:
	- Easy to understand
	- Accounts for feature interactions
	- Does not require costly re-estimation of a model (c.f. methods that delete features)
- Cons:
	- Requires labelled data
	- Possibly large variance with random permutation
	- Suffers from feature correlation
		- Biased toward unrealistic data instances
		- Correlated features ‘share’ importance, which might underestimate their merit
	- When using Random Forest, make sure you actually look at permutation importance (as opposed to Gini importance, see Strobl et al., 2007)
### Partial Dependence Plot (PDP)
![[Partial Dependence Plot (PDP).png]]
- Examine the marginal effect of a feature on model predictions
- Depict how predictions change with changes in a feature when keeping everything else constant
- Easy to understand and implement
- Causal interpretation of the model prediction
- Make sure to examine feature distribution
- Does not scale to more than two features per plot
- Assume that features in set $S$ and set $C$are not correlated
- Heterogeneous effects (e.g. interactions) may be overlooked
## Local and Example Based Explanations (LIME)
Local surrogate models are interpretable models that are used to explain individual predictions of black-box models
### Characteristic features
#### Model-agnosticism
#### Locality
#### Interpretability
### Pseudo-code representation
- Select your instance of interest
- Perturb your dataset
- Get the black box predictions for these new samples
- Weight the new samples according to their proximity to the instance of interest
- Train a weighted, interpretable model on the dataset with the variations
- Explain the prediction by interpreting the local model
### Remarks
- Easy to use and good software support (e.g., Python library lime )
- Independence between prediction and local surrogate model
- Concise explanation
- Defining a meaningful neighborhood is difficult
- Sampling data instances from a Gaussian while ignoring feature correlation
## SHAP (SHapley Additive exPlanations)
- Goal is to explain the prediction of an instance $x_i$ by computing the contribution of each feature to the prediction
- Local, model agnostic explanation methods
## Recent IML Developments
- Anchors (Ribeiro et al., 2018)
- Counterfactual examples
- Adversarial examples
# Imbalanced & Cost-sensitive Learning
## Imbalanced Learning
### Model Evaluation in the Face of Class Imbalance
#### Classification accuracy revisited
- Percentage correctly classified / classification accuracy: Correct classifications /all classification -> (TP+TN) / (TP+FP+TN+FN)
- Error rate: Wrong classifications /all classifications -> (FP+FN) / (TP+FP+TN+FN)
- Specificity: Correctly classified bad risks /no. of bad risk in total -> TN / (TN+FP)
- Sensitivity: Correctly classified good risks /no. of good risk in total -> TP / (TP+FN)
#### Alternative performance measure
![[Threshold metrics with higher robustness to class skew.png]]
### Graphical evaluation frameworks
#### Receiver Operating Characteristics Curves
- y: True positive rate (TPR), i.e. sensitivity
- x: False positive rate (FPR), i.e. 1-specificity
#### Alternative charts
- Cost-Curves (Drummond/Holte, 2006)
- Brier-Curves (Hernández-Orallo et al., 2011)
- Precision-Recall Curves
### Class imbalance affects model development
### Algorithmic Adaptation of Standard Methods
#### Subagging approach of Paleologo et al. (2010)
#### Resampling strategies
- Undersampling & Oversampling
- Synthetic Minority Class Oversampling (SMOTE)
- SMOTE generates the same number of synthetic cases for each original minority class case: Borderline SMOTE
- Sampling with data cleaning
	- Tomek links: pair of minimally distanced nearest neighbors of opposite classes
	- If two cases form a Tomek link:
		- Either one of them is noise
		- Or they both are near a class border
	- Use Tomek links to “clean-up” class overlap after synthetic sampling, e.g. Batista et al. (2004)
## Cost-Sensitive Learning
### Data space weighting
The Expected Value (EV) Framework: Approximate business value of the classifier
- Enumerate possible outcomes
- Weight by their entry probability
### Algorithmic adaptation
#### Decision threshold (Bayes risk)
- Predict the class with minimum risk
- Two class credit scoring setting
	- ![[Two class credit scoring setting.png]]
	- ![[Cost-minimal classification cut-off.png]]
	- Threshold for classifier to classify x as BAD, iif:
	$p(b|x) \geq \tau^{*} = \frac{C(b,G)-C(g,G)}{C(b,G)+C(g,B)-C(b,B)-C(g,G)} =  \frac{C(b,G)}{C(b,G)+C(g,B)}$
	- Implication: not the actual costs but their ratio drive classification decision
	- Set cut-off such that the share of predicted positives equals the prior probability of the positive class in the training set & Empirical tuning (supported by e.g., sklearn)
#### Model estimation
- Develop classification model
- Relabel training data
- Develop final classifier from relabeled training set
#### Model combination (in ensembles)

# Marketing Decision Models
## Data-Driven Marketing Fundamentals
## Marketing Decision Model Evaluation
### Lift analysis
Improvement over random targeting through prediction model
### Cost-benefit framework
- Churn prediction: Objective: Prevent customer defection
![[Dynamics of a churn management program.png]]
- Profit of a customer retention program
![[Profit of a customer retention program.png]]
- Maximum profit criterion (MP)
- Expected maximum profit criterion (EMPC)
## Marketing Decision Model Development
Bhattacharyya (1998) uses a genetic algorithm for direct lift maximization
# Summary
![[BADS_summary.ipynb]]