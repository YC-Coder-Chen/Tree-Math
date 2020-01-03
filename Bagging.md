Tree-Math
============
Machine learning study notes, contains Math behind all the mainstream tree-based machine learning models, covering basic decision tree models (ID3, C4.5, CART), boosted models (GBM, AdaBoost, Xgboost, LightGBM), bagging models (Bagging Tree, Random Forest, ExtraTrees).  

Bagging Tree Models
------------
**Bagging Tree**
> One Sentence Summary:  
Train multiple strong base learners on different subsets of dataset parallelly and take the average or majority vote as the final predictions.  

- **a. Difference between Bagging and Boosting**  
The logic behind the boosting method is adding weak base learners step by step to form a strong learner and correct previous mistakes. But the core idea behind bagging is training and aggregating multiple strong base learners at the same time to prevent overfitting.  

  | Aspects  | Boosting  | Bagging |
  | :-----------: | :-----------:  |:-----------:  |
  | **Ensemble Category**| Sequential Ensembling: weak base learners are generated sequentially | Parallel Ensembling: Strong base learners are generated parallelly | 
  | **Overall Target** | Reduce Bias | Reduce Variance|
  | **Target of individual base learners** | Reduce previous weak learners' error     | Reduce the overfitting of each strong base learners|  
  | **Parallel Computing** | Parallel computing within a single tree (XGboost)  | Parallel computing within a single tree & across running different trees |

- **b. The Bagging Tree Classification Algorithm**  
*Model Input*:  
  - Dataset: ![img](https://latex.codecogs.com/svg.latex?D%20%3D%20%5C%7B%28x1%2Cy1%29%2C%20...%2C%20%28x_i%2C%20y_i%29%2C%20...%2C%20%28x_N%2C%20y_N%29%5C%7D%20%2C%20y_i%20%5Cin%20%5C%7B-1%2C1%5C%7D)  
  - Base Learner: ![img](https://latex.codecogs.com/svg.latex?%5Cepsilon%28x%29)
  - Number of base learner: T
  - Number of samples in each data subset: n

  *Model Output*: Final classifier: G(x)  

  *Steps*:  
  - For t = 1, 2, 3, ..., T: 
    - Select random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D) from the Dataset D with replacement. Each random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D) contains n samples. 
    - Based on the random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D), train a base learner ![img](https://latex.codecogs.com/svg.latex?%5Cepsilon_t%28x%29)  
  - Output the final model ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20G%28x%29%20%3D%20%5Cunderset%7By%5Cin%20%5C%7B-1%2C1%5C%7D%7D%7Bargmin%7D%20%5Csum_%7Bt%3D1%7D%5E%7BT%7D%20%5Cmathbb%7BI%7D%28%5Cepsilon_t%28x%29%20%3D%20y%29) using majority vote. 

- **c. The Bagging Tree Regression Algorithm**  
*Model Input*:  
  - Dataset: ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20D%20%3D%20%5C%7B%28x1%2Cy1%29%2C%20...%2C%20%28x_i%2C%20y_i%29%2C%20...%2C%20%28x_N%2C%20y_N%29%5C%7D%20%2C%20y_i%20%5Cin%20%5Cboldsymbol%7BR%7D)  
  - Base Learner: ![img](https://latex.codecogs.com/svg.latex?%5Cepsilon%28x%29)
  - Number of base learner: T
  - Number of samples in each data subset: n

  *Model Output*: Final regressor: G(x)  

  *Steps*:  
  - For t = 1, 2, 3, ..., T: 
    - Select random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D) from the Dataset D with replacement. Each random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D) contains n samples. 
    - Based on the random subsets ![img](https://latex.codecogs.com/svg.latex?D_%7Bsubset_t%7D), train a base learner ![img](https://latex.codecogs.com/svg.latex?%5Cepsilon_t%28x%29)   
  - Output the final model ![img](https://latex.codecogs.com/svg.latex?%5Csmall%20G%28x%29%20%3D%20%5Cfrac%7B1%7D%7BT%7D%20%5Csum_%7Bt%3D1%7D%5E%7BT%7D%5Cepsilon_t%28x%29) by taking take the average prediction of each base learners.  

**Reference**  

1. Breiman, Leo. "Bagging predictors." Machine learning 24.2 (1996): 123-140.  
2. Zhihua Zhou. Machine Learning[M]. Tsinghua University Press, 2018. [Chinese]  
3. https://towardsdatascience.com/decision-tree-ensembles-bagging-and-boosting-266a8ba60fd9  
4. https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/   
5. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html

