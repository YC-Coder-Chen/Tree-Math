Tree-Math
============
Machine learning study notes, contains Math behind all the mainstream tree-based machine learning models, covering basic decision tree models (ID3, C4.5, CART), boosted models (GBM, AdaBoost, Xgboost, LightGBM), bagging models (Bagging Tree, Random Forest, ExtraTrees).  

Bagging Tree Models
------------
**Extremely Randomized Trees (ExtraTrees)**
> One Sentence Summary:  
Train multiple strong base learners on the whole dataset parallelly and aggregate their results as the final predictions, but in each splits inside each base learners, only allow to use separate subsets of features and possible split points are drawn randomly.  

- **a. Difference between Random Forest and Extremely Randomized Trees**  
The main difference between Random Forest and Extremely Randomized Trees is that in Extremely Randomized Trees, it uses random search to find the best split points & best split features of each node of a base learner whereas, in Random Forest, it uses the exact greedy method.  

  | Aspects  | Bagging Tree  | Random Forest | ExtraTrees |
  | :-----------: | :-----------:  |:-----------:  |:-----------:  |
  | **Subset of samples**| Yes | Yes | No (But can) |
  | **Subset of features**| No | Yes | Yes |
  | **Exact Greedy Method**| Yes | Yes | No (Random Search instead) |
 

- **b. The Extremely Randomized Trees Classification Algorithm**  
*Suppose we also introduce subsets of samples*.  

  *Model Input*:  
    - Dataset with M features: ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20D%20%3D%20%5C%7B%28x1%2Cy1%29%2C%20...%2C%20%28x_i%2C%20y_i%29%2C%20...%2C%20%28x_N%2C%20y_N%29%5C%7D%20%2C%20y_i%20%5Cin%20%5C%7B-1%2C1%5C%7D%2C%20x_i%20%5Cin%20%5Cboldsymbol%7BR%7D%5EM)  
    - Base Learner: ![img](https://latex.codecogs.com/svg.latex?%5Cepsilon%28x%29)
    - Number of base learner: T
    - Number of samples in each data subset: n
    - Number of features allowed in each split inside a base learner: m

    *Model Output*: Final classifier: G(x)  

    *Steps*:  
    - For t = 1, 2, 3, ..., T: 
      - Select random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D) from the Dataset D with replacement. Each random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D) contains n samples. 
      - Based on the random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D), train a base learner ![img](https://latex.codecogs.com/svg.latex?%5Cepsilon_t%28x%29). Specifically, in each split of a base learner, we select m attributes ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20%5C%7Ba_%7Bt1%7D%2C%20a_%7Bt2%7D%2C%20...a_%7Bti%7D%2C%2C...%20a_%7Btm%7D%5C%7D). Then for each attribute ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D), we draw a random cut-point ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20ac_%7Bti%7D) uniformly in [![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D__%7BMIN%7D), ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D__%7BMAX%7D)], where ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D__%7BMIN%7D) and ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D__%7BMAX%7D) are the minimal and maximal value of attribute ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D) in ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D). So now we will have m splits points ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20%5C%7Bac_%7Bt1%7D%2C%20ac_%7Bt2%7D%2C...%2C%20ac_%7Bti%7D%2C...%2Cac_%7Btm%7D%5C%7D), each corresponds with one attribute. Among these m splits, we select the feature that reduce the gini or entropy the most at corresponding cut point.  
    - Output the final model ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20G%28x%29%20%3D%20%5Cunderset%7By%5Cin%20%5C%7B-1%2C1%5C%7D%7D%7Bargmin%7D%20%5Csum_%7Bt%3D1%7D%5E%7BT%7D%20%5Cmathbb%7BI%7D%28%5Cepsilon_t%28x%29%20%3D%20y%29) using majority vote. 

- **c. The Extremely Randomized Trees Regression Algorithm**  
*Suppose we also introduce subsets of samples*.  

  *Model Input*:  
    - Dataset with M features: ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20D%20%3D%20%5C%7B%28x1%2Cy1%29%2C%20...%2C%20%28x_i%2C%20y_i%29%2C%20...%2C%20%28x_N%2C%20y_N%29%5C%7D%20%2C%20y_i%20%5Cin%20%5Cboldsymbol%7BR%7D%2C%20x_i%20%5Cin%20%5Cboldsymbol%7BR%7D%5EM)  
    - Base Learner: ![img](https://latex.codecogs.com/svg.latex?%5Cepsilon%28x%29)
    - Number of base learner: T
    - Number of samples in each data subset: n
    - Number of features allowed in each split inside a base learner: m  

    *Model Output*: Final regressor: G(x)  

    *Steps*:  
    - For t = 1, 2, 3, ..., T: 
      - Select random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D) from the Dataset D with replacement. Each random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D) contains n samples. 
      - Based on the random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D), train a base learner ![img](https://latex.codecogs.com/svg.latex?%5Cepsilon_t%28x%29). Specifically, in each split of a base learner, we select m attributes ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20%5C%7Ba_%7Bt1%7D%2C%20a_%7Bt2%7D%2C%20...a_%7Bti%7D%2C%2C...%20a_%7Btm%7D%5C%7D). Then for each attribute ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D), we draw a random cut-point ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20ac_%7Bti%7D) uniformly in [![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D__%7BMIN%7D), ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D__%7BMAX%7D)], where ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D__%7BMIN%7D) and ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D__%7BMAX%7D) are the minimal and maximal value of attribute ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20a_%7Bti%7D) in ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D). So now we will have m splits points ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20%5C%7Bac_%7Bt1%7D%2C%20ac_%7Bt2%7D%2C...%2C%20ac_%7Bti%7D%2C...%2Cac_%7Btm%7D%5C%7D), each corresponds with one attribute. Among these m splits, we select the feature that reduce the MSE the most at corresponding cut point.  
    - Output the final model ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20G%28x%29%20%3D%20%5Cfrac%7B1%7D%7BT%7D%20%5Csum_%7Bt%3D1%7D%5E%7BT%7D%5Cepsilon_t%28x%29) by taking take the average prediction of each base learners.  

**Reference**  

1. Geurts, Pierre, Damien Ernst, and Louis Wehenkel. "Extremely randomized trees." Machine learning 63.1 (2006): 3-42.   
2. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html  
3. https://scikit-learn.org/stable/modules/ensemble.html