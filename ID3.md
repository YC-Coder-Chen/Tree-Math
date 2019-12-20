Tree-Math
============
Machine learning study notes, contains Math behind all the mainstream tree-based machine learning models, covering basic decision tree models (ID3, C4.5, CART), boosted models (GBM, AdaBoost, Xgboost, LightGBM), bagging models (Bagging Trees, Random Forest, Extra Trees).  



Decision Tree
------------
**ID3 Model**
> One Sentence Summary:   
Splitting the dataset recurrently on the features that yields the maximum information gain.  

- **a. What is information gain**    

  We define entropy H(X) as a metric to reflect the uncertainty of a random variable. 
  Suppose that a random variable X follows the below dirtribution:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20P%28X%3Dx%29%3Dp_%7Bi%7D%2C%20i%20%3D%201%2C2%2C3%2C...%2Cn)  
  Then the entropy of X is as below. And high entropy value also means more uncertain variables.   

  ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20%24H%28X%29%20%3D%20-%5Csum_%7Bi%20%3D%201%7D%5E%7Bn%7Dp_%7Bi%7D%5Clog%28p_%7Bi%7D%29%24)  
 
  So the information gain g(D, A) of dataset D conditioned on feature A is as below:

  ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20g%28D%2CA%29%3DH%28D%29-H%28D%7CA%29) 

- **b. How to compute information gain**  

  Suppose that the dataset D has k categories, each category is C1, C2, ... , Ck.
Suppose that feature A can split the dataset into n subset D1, D2, D3,..., Dn.  
Suppose that Dik denotes the subset of the sample of category k in subset Di.  
    1. **compute H(D) of dataset D**   
    |A| means the number of sample in A  
    ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20H%28D%29%20%3D%20-%5Csum_%7Bk%3D1%7D%5E%7BK%7D%7B%5Cfrac%7B%7CC_%7Bk%7D%7C%7D%7B%7CD%7C%7D%7D%5Clog_%7B%20%7D%5Cfrac%7B%7CC_%7Bk%7D%7C%7D%7B%7CD%7C%7D)  
    2. **compute H(D|A) of dataset D conditioned on condition A**  
    suppose we split the dataset D on feature A into D1, D2, ..., Dn, totally n parts. Then the H(D|A) is as below:  
    ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20H%28D%7CA%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7DH%28D_%7Bi%7D%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7D%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cfrac%7B%7CD_%7Bik%7D%7C%7D%7B%7CD_i%7C%7D%5Clog%28%5Cfrac%7B%7CD_%7Bik%7D%7C%7D%7B%7CD_i%7C%7D%29)    

- **c. The actual ID3 model & algorithm**  
During each split, we select the feature that gives the maximum increase in information gain (g(D, A)).  

  Model Input: dataset D, feature set A, stopping threshold e.  
Model Output: ID3 decision tree.  
 
  **Recurrent Algorithm**:  
  (1) If every sample in D belongs to one class Ck, stop splitting at this node, predict every node in D as Ck.  
  
  (2.1) If A is empty, then we stop splitting at this node, predict D as the class that has the most samples.    
  (2.2) If A is not empty, then we loop through the feature set A, compute the information gain for each feature using the equations in a and b.
  Suppose among them the maximum information gain is ga, obtained by splitting on the featurea a.   
  (2.2.1) If ga > threshold e, then we split on feature a, and split dataset into mini subset {D1, D2, ... Di} based on different values of categorical feature a. For each subset in {D1, D2, ... Di}, treat Di as the new dataset D, and treat A-{a} as the new feature set A, repeat the splitting process.  
  (2.2.2) If ga <= threshold e, then we split stop splitting at this node, set this final node to be the class that has the most samples. 
  
  (3) The tree stops growing when there is no splitting in any of the subsets.

**Reference**  

1. Quinlan J R. Induction of Decision Trees. Mach. Learn[J]. 1986. 
2. Quinlan J R. C4. 5: programs for machine learning[M]. Elsevier, 2014.
3. Hang Li. Statistical Learning Method[M]. Tsinghua University Press, 2019. [Chinese]
4. Zhihua Zhou. Machine Learning[M]. Tsinghua University Press, 2018. [Chinese]
5. Wikipedia contributors. ID3 algorithm. Wikipedia, The Free Encyclopedia. October 23, 2019, 17:33 UTC. Available at: https://en.wikipedia.org/w/index.php?title=ID3_algorithm&oldid=922686642. Accessed November 11, 2019.
6. https://medium.com/machine-learning-guy/an-introduction-to-decision-tree-learning-id3-algorithm-54c74eb2ad55
7. https://towardsdatascience.com/decision-trees-introduction-id3-8447fd5213e9

