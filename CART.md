Tree-Math
============
My study notes, contains Math behind all the mainstream tree-based machine learning models, covering basic decision tree models (ID3, C4.5, CART), boosted models (GBM, AdaBoost, Xgboost, LightGBM), bagging models (Random Forest).



Decision Tree
------------
**CART Tree Model**
> One Sentence Summary:   
Using the Gini index(classifier) or MSE(regression tree) to find the features to split the current node into two parts in each split.

- **a. What is MSE in regression decision tree**  
In each split, we are choosing the best split feature j and split point value s to seperate the current node area into two parts T1, T2. The corresponding predicted value at each sub-tree is C1, C2.  

  ![img](https://latex.codecogs.com/svg.latex?T_1%28j%2C%20s%29%20%3D%20%5Cleft%20%5C%7B%20x%20%7C%20x%5E%7B%28j%29%7D%20%3C%3D%20s%20%5Cright%20%5C%7D%2C%20T_2%28j%2C%20s%29%20%3D%20%5Cleft%20%5C%7B%20x%20%7C%20x%5E%7B%28j%29%7D%20%3E%20s%20%5Cright%20%5C%7D)  
![img](https://latex.codecogs.com/svg.latex?C_%7Bm%7D%20%3D%20%5Cfrac%7B1%7D%7BN_m%7D%20%5Csum_%7BX_i%20%5Cin%20T_m%28j%2C%20s%29%7D%5E%7B%20%7Dy_i%20%5C%2C%20%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20m%20%3D%201%2C2)   
![img](https://latex.codecogs.com/svg.latex?MSE%28j%2C%20s%29%20%3D%20%5Csum_%7Bx_i%20%5Cin%20T_1%28j%2C%20s%29%7D%5E%7B%20%7D%7B%28y_i%20-%20C_1%29%7D%20%5E2%20&plus;%20%5Csum_%7Bx_i%20%5Cin%20T_2%28j%2C%20s%29%7D%5E%7B%20%7D%7B%28y_i%20-%20C_2%29%7D%20%5E2)

- **b. What is gini index in classification decision tree**  
Suppose that there are K class in the dataset, Pk is the probability of class Ck, then Gini index of dataset D is as below:  
![img](https://latex.codecogs.com/svg.latex?Gini%28D%29%20%3D%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20P_k%281-P_k%29%20%3D%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%7B%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%7D%281-%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%29%20%3D%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%7B%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%7D%20-%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%7B%7B%28%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%7D%29%5E2%7D)  
![img](https://latex.codecogs.com/svg.latex?Gini%28D%29%20%3D%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%7B%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%7D%20-%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%7B%7B%28%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%7D%29%5E2%7D%20%3D%201%20-%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%7B%7B%28%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%7D%29%5E2%7D)  
So the Gini index of dataset given split on feature A is as below:  

  ![img](https://latex.codecogs.com/svg.latex?Gini%28D%2C%20A%29%20%3D%20%5Cfrac%7B%7CT_1%7C%7D%7B%7CD%7C%7D%20*%20Gini%28T_1%29%20&plus;%20%5Cfrac%7B%7CT_2%7C%7D%7B%7CD%7C%7D%20*%20Gini%28T_2%29)  

- **c. The actual CART model & algorithm**  
During each split, we find the feature that gives the dataset the minimum Gini index after the split.  

  Model Input: dataset D, feature set A, stopping threshold e.  
  Model Output: CART decision tree.  
 
  **Recurrent Algorithm for Classification Tree**:  (very similar to ID3/C4.5)  
  (1) If all the samples in D already only belongs one class Ck, stop splitting at this node, set this final node to be class Ck.  
  
  (2.1) We loop through all the features in A, for each feature j, we will loop through all the possible value of the feature j, find the best split point s to seperate the dataset that bring the minimum Gini index. Suppose that now the Gini index is Gini_after, and the Gini index before the split is Gini_before.  

  How to split the dataset into two parts in each split:  
  
  ![img](https://latex.codecogs.com/svg.latex?T_1%28j%2C%20s%29%20%3D%20%5Cleft%20%5C%7B%20x%20%7C%20x%5E%7B%28j%29%7D%20%3C%3D%20s%20%5Cright%20%5C%7D%2C%20T_2%28j%2C%20s%29%20%3D%20%5Cleft%20%5C%7B%20x%20%7C%20x%5E%7B%28j%29%7D%20%3E%20s%20%5Cright%20%5C%7D)  
 
  Gini Before:   
  ![img](https://latex.codecogs.com/svg.latex?Gini_%7B%5C%2C%20%5C%2C%20%5C%2C%20before%7D%28D%29%20%3D%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%7B%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%7D%20-%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%7B%7B%28%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%7D%29%5E2%7D%20%3D%201%20-%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%7B%7B%28%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%7D%29%5E2%7D)   
  
  How to find Gini_after:    
  ![img](https://latex.codecogs.com/svg.latex?%5Cunderset%7Bj%2C%20s%7D%7BMin%7D%20%5C%2C%5C%2C%20%5C%2C%20Gini%28D%2C%20j%2C%20s%29%20%5CRightarrow%20%5Cunderset%7Bj%2C%20s%7D%7BMin%7D%20%5C%2C%20%5C%2C%20%5C%2C%20%5B%20%5Cunderset%7BT_1%7D%7BMin%7D%7B%5Cfrac%7B%7CT_1%7C%7D%7B%7CD%7C%7D%20*%20Gini%28T_1%29%7D%20%5C%2C%20%5C%2C%20&plus;%20%5C%2C%20%5C%2C%20%5Cunderset%7BT_2%7D%7BMin%7D%5Cfrac%7B%7CT_2%7C%7D%7B%7CD%7C%7D%20*%20Gini%28T_2%29%5D)  
  
  (2.2.1) If Gini decrease = Gini_before - Gini_after > threshold e, then we split on feature j, and break dataset into separate subset T1, T2 based on splitting value s. For each subset dataset in {T1, T2}, treat Ti as the new dataset D, recurrently continue this splitting process.  
  (2.2.2) If Gini decrease <= threshold e, then we split stop splitting at this node, set this final node to be the class that has the most samples.

  **Recurrent Algorithm for Regression Tree**:  (very similar to ID3/C4.5)  
  (1) If all the samples in D have the same output value y, stop splitting at this node, set this final node prediction to be y.  
  
  (2.1) We loop through all the features in A, for each feature j, we will loop through all the possible value of the feature j, find the best split point s to seperate the dataset that bring the minimum MSE. Suppose that now the MSE is MSE_after, and the MSE before the split is MSE_before.  

  How to split the dataset into two partsin each split:  
  
  ![img](https://latex.codecogs.com/svg.latex?T_1%28j%2C%20s%29%20%3D%20%5Cleft%20%5C%7B%20x%20%7C%20x%5E%7B%28j%29%7D%20%3C%3D%20s%20%5Cright%20%5C%7D%2C%20T_2%28j%2C%20s%29%20%3D%20%5Cleft%20%5C%7B%20x%20%7C%20x%5E%7B%28j%29%7D%20%3E%20s%20%5Cright%20%5C%7D)  
  
  MSE_before:  
  ![img](https://latex.codecogs.com/svg.latex?MSE_%7Bbefore%7D%20%3D%20%5Csum_%7Bx_i%20%5Cin%20D%7D%5E%7B%20%7D%7B%28y_i%20-%20C%29%7D%20%5E2%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20C%20%3D%20%5Cfrac%7B1%7D%7B%7CD%7C%7D%20%5Csum_%7BX_i%20%5Cin%20D%7D%5E%7B%20%7Dy_i%3B)  

  How to find MSE_after:  
  ![img](https://latex.codecogs.com/svg.latex?C_%7Bm%7D%20%3D%20%5Cfrac%7B1%7D%7BN_m%7D%20%5Csum_%7BX_i%20%5Cin%20T_m%28j%2C%20s%29%7D%5E%7B%20%7Dy_i%20%5C%2C%20%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20m%20%3D%201%2C2)  
  ![img](https://latex.codecogs.com/svg.latex?%5Cunderset%7Bj%2C%20s%7D%7BMin%7D%20%5C%2C%5C%2C%20%5C%2C%20MSE%28j%2C%20s%29%20%5CRightarrow%20%5Cunderset%7Bj%2C%20s%7D%7BMin%7D%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5B%20%5Cunderset%7BC_1%7D%7BMin%7D%5Csum_%7Bx_i%20%5Cin%20T_1%28j%2C%20s%29%7D%5E%7B%20%7D%7B%28y_i%20-%20C_1%29%7D%20%5E2%20%5C%2C%20%5C%2C%20&plus;%20%5C%2C%20%5C%2C%20%5Cunderset%7BC_2%7D%7BMin%7D%5Csum_%7Bx_i%20%5Cin%20T_2%28j%2C%20s%29%7D%5E%7B%20%7D%7B%28y_i%20-%20C_2%29%7D%20%5E2%5D)  
  (2.2.1) If MSE decrease = MSE_before - MSE_after > threshold e, then we split on feature j, and break dataset into separate subset T1, T2 based on splitting value s. For each subset dataset in {T1, T2}, treat Ti as the new dataset D, recurrently continue this splitting process.  
  (2.2.2) If MSE decrease <= threshold e, then we stop splitting at this node, set this final node to be the average of the output variable y that belongs to this subset. 

**Scikit-learn Application**
> **Decision Tree Function**:   
***Class*** sklearn.tree.DecisionTreeClassifier (criterion=’gini’,     splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

- **criterion** : string, optional (default=”gini”)  
  **'gini'** for gini impurity, **'entropy'** for information gain which is discussed in C4_5 decsion tree.
 
- **splitter** : string, optional (default=”best”)   
  **'best'** means searching through all features and values and picking the best split based on the criterion (gini impurity or entropy value). 
  **'random'** means considering a random subset of features and picking the best split within this subset. This option is less computation-intensive and less prone to overfitting. It makes more sense if all variables are somewhat equally relevant to the output variable.

- **max_depth** : int or None, optional (default=None)  
  How deep the tree can grow. For example, a tree with a max_depth of 2 is allowed to split into a maximum of 4 leaf nodes. This term is often tuned to regularize the tree and prevent overfitting. 

- **min_sample_split & min_sample_leaf** : int, float, optional    
  min_sample_split defines the minimum number of samples required to split an internal node. min_sample_leaf defines the minimum number of samples required to be at a leaf node. Recall that in the recurrent algorithm, after splitting the dataset D into subset {T1, T2}, we treat each subset as the new dataset. However,if you set the min_sample_split = A, and the min_sample_leaf = B, then every dataset (subset) D will continue splitting only when the subset itself contains more than A samples and the subsets D splits into contain more than B samples. So these two hyperparameters are used to regularize the growth of the tree.
  
  Values for these two hyperparameters can be int or float. If int, it specifies exact sample numbers. If float, it means fraction and equals to ceil(fraction*total_number_of_samples). For example, set float value as 0.3 equals to set an int value of 30 in a dataset with 101 samples. 

- **min_weight_fraction_leaf** : float, optional (default=0.)  
  The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
  

- **max_features** : int, float, string or None, optional (default=None)  
  max_features represents the number of features to consider when looking for the best split. For example, if you set max_feature at 10, then at each splitting, the algorithm will randomly select 10 features and search for the best one out of the ten features. Note that sometimes the search will exceed the max_features if the algorithm cannot find a valid partition of node samples in the first search. Similar to min_sample_split, values for this hyperparameter could be int or float or {'auto','sqrt','log2','none'}, please refer to the official document for more explanation. There is also an article introducing how to pick the max_features in decision tree and random forests.

- **max_leaf_nodes** : int or None, optional (default=None)    
  Grow a tree with max_leaf_nodes in best-first fashion. Please refer to the introduction of lightGMB for difference between level-wise tree and leaf-wise tree (best-first search algorithm).

- **min_impurity_decrease** : float, optional (default=0.)   
  A node will be split if this split induces a decrease of the impurity greater than or equal to this value. So this hyperparameter represents the threshold value e in the algorithm. Recall that in (2.2.1), Gini decrease = Gini_before - Gini_after, however in real practice, Gini decrease is calculated using the below formula:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cfrac%7BN_t%7D%7BN%7D%20*%20%28impurity%20-%20%5Cfrac%7BN_%7BtR%7D%7D%7BN_t%7D%20*%20right%5C_impurity%20-%20%5Cfrac%7BN_%7BtL%7D%7D%7BN_t%7D%20*%20left%5C_impurity%29)

  where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child. Using the weighted Gini decrease doesn't affect picking the best features, it only influences the splitting decision. 

- **class_weight** : dict, list of dicts, “balanced” or None, default=None 
  Weights associated with classes in the form {class_label: weight}. Use this hyperparameter to adjust for importance of each class. **balanced** automatically adjust weights inversely to class frequencies in this formula. So less frequent classes are assigned higher weigths.

  ![img](https://latex.codecogs.com/svg.latex?%5Cfrac%7Bn%5C_samples%7D%7Bn%5C_class%20*%20np.bincount%28y%29%7D)
  
  If class_weight is passed, weighted sum is used for calculating the Nt, N, NtR, NtL in the min_impurity_decrease.

- **presort** : bool, optional (default=False)  
  Whether to presort the data to speed up the finding of best splits in fitting. For the default settings of a decision tree on large datasets, setting this to true may slow down the training process. When using either a smaller dataset or a restricted depth, this may speed up the training.



**Reference**  

1. Breiman L, Friedman J, Olshen R, et al. Classification and regression trees. Wadsworth Int[J]. Group, 1984, 37(15): 237-251.
2. Quinlan J R. C4. 5: programs for machine learning[M]. Elsevier, 2014.
3. Hang Li. Statistical Learning Method[M]. Tsinghua University Press, 2019. [Chinese]
4. Zhihua Zhou. Machine Learning[M]. Tsinghua University Press, 2018. [Chinese]
5. Wikipedia contributors. Decision tree learning. Wikipedia, The Free Encyclopedia. October 21, 2019, 22:08 UTC. Available at: https://en.wikipedia.org/w/index.php?title=Decision_tree_learning&oldid=922400627. Accessed November 11, 2019.
6. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier 
7. https://stackoverflow.com/questions/46756606/what-does-splitter-attribute-in-sklearns-decisiontreeclassifier-do 
8. https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3
9. https://discuss.analyticsvidhya.com/t/what-does-min-samples-split-means-in-decision-tree/6233/2
10. **How to set max_features**: https://stats.stackexchange.com/questions/324370/references-on-number-of-features-to-use-in-random-forest-regression 
11. https://medium.com/datadriveninvestor/decision-tree-adventures-2-explanation-of-decision-tree-classifier-parameters-84776f39a28
12. https://towardsdatascience.com/scikit-learn-decision-trees-explained-803f3812290d 
