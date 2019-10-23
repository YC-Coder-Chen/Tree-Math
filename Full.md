Tree-Math
============
My study notes, contains Math behind all the mainstream tree-based machine learning models, covering basic decision tree models (ID3, C4.5, CART), boosted models (GBM, AdaBoost, Xgboost, LightGBM), bagging models (Random Forest).



Decision Tree
------------
**1. ID3 Model**
> One Sentence Summary:   
Using the information gain matrix to find the features in each split.

- **a. What is information gain**    

  We define entropy H(X) as a matrix to reflect the uncertainty of a random variable. 
  Suppose that a random variable X follows the below dirtribution:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20P%28X%3Dx%29%3Dp_%7Bi%7D%2C%20i%20%3D%201%2C2%2C3%2C...%2Cn)  
  Then the entropy of X is as below:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20%24H%28X%29%20%3D%20-%5Csum_%7Bi%20%3D%201%7D%5E%7Bn%7Dp_%7Bi%7D%5Clog%28p_%7Bi%7D%29%24)  
So the information gain g(D, A) of dataset D given feature A is as below:

  ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20g%28D%2CA%29%3DH%28D%29-H%28D%7CA%29) 

- **b. How to compute information gain**  

  Suppose that the dataset D has k category, each category is C1, c2, ... , ck.
Suppose that feature A can split the dataset into n subset D1, D2, D3,..., Dn.  
Suppose that Dik denotes the subset of the sample of category k in subset Di.  
    1. **compute H(D) of dataset D**   
    |A| means the number of sample in A  
    ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20H%28D%29%20%3D%20-%5Csum_%7Bk%3D1%7D%5E%7BK%7D%7B%5Cfrac%7B%7CC_%7Bk%7D%7C%7D%7B%7CD%7C%7D%7D%5Clog_%7B%20%7D%5Cfrac%7B%7CC_%7Bk%7D%7C%7D%7B%7CD%7C%7D)  
    2. **compute H(D|A) of dataset D given condition A**  
    suppose givne condition A, we split the area into D1, D2, ..., Dn, totally n parts. Then the H(D|A) is as below:  
    ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20H%28D%7CA%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7DH%28D_%7Bi%7D%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7D%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cfrac%7B%7CD_%7Bik%7D%7C%7D%7B%7CD_i%7C%7D%5Clog%28%5Cfrac%7B%7CD_%7Bik%7D%7C%7D%7B%7CD_i%7C%7D%29)    

- **c. The actual ID3 model & algorithm**  
During each split, we find the feature that gives the dataset the maximum increase in information gain (g(D, A)).  

  Model Input: dataset D, feature set A, stopping threshold e.  
Model Output: ID3 decision tree.  
 
  **Recurrent Algorithm**:  
  (1) If all the sample in D already only belongs one class Ck, stop splitting at this node, set this final node to be class Ck.  
  
  (2.1) If A is empty, then we stop splitting at this node, set this final node to be the class that has the most samples.    
  (2.2) If A is not empty, then we loop through all the features in A, compute the information gain for each of them using the above equation.
  Suppose among them the maximum information gain is ga, obtained by splitting on the featurea a.   
  (2.2.1) If ga > threshold e, then we split on feature a, and break dataset into separate subset Di based on different value of category a. For each subset dataset in {D1, D2, ... Di}, treat Di as the new dataset D, treat A-{a} as the new feature set A, recurrently continue this splitting process.  
  (2.2.2) If ga <= threshold e, then we split stop splitting at this node, set this final node to be the class that has the most samples.

**2. C4.5 Model**
> One Sentence Summary:   
Using the information gain ratio instead of information gain to find the features in each split.

- **a. What is information gain ratio**    

  The information gain ratio gr(D, A) of dataset D given feature A is as below:  
  ![img](https://latex.codecogs.com/svg.latex?g_r%28D%2CA%29%20%3D%20%5Cfrac%7Bg%28D%2C%20A%29%7D%7BH%28D%29%7D)      
  The entropy of dataset D is the same as discussed in ID3:  
  ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20H%28D%29%20%3D%20-%5Csum_%7Bk%3D1%7D%5E%7BK%7D%7B%5Cfrac%7B%7CC_%7Bk%7D%7C%7D%7B%7CD%7C%7D%7D%5Clog_%7B%20%7D%5Cfrac%7B%7CC_%7Bk%7D%7C%7D%7B%7CD%7C%7D)  

- **b. How to compute information gain ratio**  

  Suppose that the dataset D has k category, each category is C1, c2, ... , ck.
Suppose that feature A can split the dataset into n subset D1, D2, D3,..., Dn.  
Suppose that Dik denotes the subset of the sample of category k in subset Di.  
    1. **compute H(D) of dataset D**   
    |A| means the number of sample in A  
    ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20H%28D%29%20%3D%20-%5Csum_%7Bk%3D1%7D%5E%7BK%7D%7B%5Cfrac%7B%7CC_%7Bk%7D%7C%7D%7B%7CD%7C%7D%7D%5Clog_%7B%20%7D%5Cfrac%7B%7CC_%7Bk%7D%7C%7D%7B%7CD%7C%7D)  
    2. **compute H(D|A) of dataset D given condition A**  
    suppose givne condition A, we split the area into D1, D2, ..., Dn, totally n parts. Then the H(D|A) is as below:  
    ![img](https://latex.codecogs.com/svg.latex?%5Cbg_black%20H%28D%7CA%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7DH%28D_%7Bi%7D%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7D%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cfrac%7B%7CD_%7Bik%7D%7C%7D%7B%7CD_i%7C%7D%5Clog%28%5Cfrac%7B%7CD_%7Bik%7D%7C%7D%7B%7CD_i%7C%7D%29)  
    3. **compute information gain ratio gr(D, A) given condition A**  
    Below is the formula to compute the information gain    
    ![img](https://latex.codecogs.com/svg.latex?g_r%28D%2CA%29%20%3D%20%5Cfrac%7Bg%28D%2C%20A%29%7D%7BH%28D%29%7D%20%3D%20%5Cfrac%7BH%28D%29%20-%20H%28D%7CA%29%7D%7BH%28D%29%7D)  

- **c. The actual C4.5 model & algorithm**  
During each split, we find the feature that gives the dataset the maximum increase in information gain ratio (gr(D, A)).  

  Model Input: dataset D, feature set A, stopping threshold e.  
  Model Output: C4.5 decision tree.  
 
  **Recurrent Algorithm**:  (very similar to ID3)  
  (1) If all the sample in D already only belongs one class Ck, stop splitting at this node, set this final node to be class Ck.  
  
  (2.1) If A is empty, then we stop splitting at this node, set this final node to be the class that has the most samples.    
  (2.2) If A is not empty, then we loop through all the features in A, compute the information gain ratio for each of them using the above equation.
  Suppose among them the maximum information gain ratio is gra, obtained by splitting on the feature a.   
  (2.2.1) If gra > threshold e, then we split on feature a, and break dataset into separate subset Di based on different value of category a. For each subset dataset in {D1, D2, ... Di}, treat Di as the new dataset D, treat A-{a} as the new feature set A, recurrently continue this splitting process.  
  (2.2.2) If gra <= threshold e, then we split stop splitting at this node, set this final node to be the class that has the most samples.

**3. CART Tree Model**
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
  ![img](https://latex.codecogs.com/svg.latex?MSE_%7Bbefore%7D%20%3D%20%5Csum_%7Bx_i%20%5Cin%20D%7D%5E%7B%20%7D%7B%28y_i%20-%20C%29%7D%20%5E2%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20C%20%3D%20%5Cfrac%7B1%7D%7B%7CD%7C%7D%20%5Csum_%7BX_i%20%5Cin%20D%7D%5E%7B%20%7Dy_i)  
  
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

Boosted Decision Tree
------------
**1. Adaptive Boosting (AdaBoost)**
> One Sentence Summary:   
Continuously adding weak base learners to the existing model and adaptively adjusting the weight for weak base learnings.

- **a. What is Forward Stagewise Additive Modeling**  
Suppose b(x;rm) is a base learning controlled by parameter rm.  
Beta_m is the parameter controlled by how each weak base learner is added.  
Then the final model f(x) based on M weak base learners will be as below:  
 ![img](https://latex.codecogs.com/svg.latex?f%28x%29%20%3D%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%20%5Cbeta_%7Bm%7D%20*%20b%28x%3Br_m%29)  

  Sign(x) is the function to convert values into actual classes. And the output classifier will be as below:  

  ![img](https://latex.codecogs.com/svg.latex?G%28x%29%20%3D%20sign%28f%28x%29%29%20%3Dsign%28%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%20%5Cbeta_%7Bm%7D%20*%20b%28x%3Br_m%29%29)   

  Global Optimal for the dataset D with N samples is as below:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cunderset%7B%5Cbeta_m%20%2C%20r_m%7D%7BMin%7D%20%28%5Csum_%7Bi%3D1%7D%5E%7BN%7DLoss%28y_i%2C%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%5Cbeta_m%20*%20b%28x_i%3B%20r_m%29%29%29)

  For each step, our optimal target is just to find the beta and r for the current base learning so as to approximate the above global optimal:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cunderset%7B%5Cbeta%2C%20r%7D%7BMin%7D%20%28%5Csum_%7Bi%3D1%7D%5E%7BN%7DLoss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Cbeta%20*%20b%28x_i%3Br%29%29%29)  

  To sum up, the Forward Stagewise Additive Modeling Algorithm is as below:  

  *Inputs*: Dataset D = {(x1,y1), (x2,y2), ... , (xN,yN)}, loss function Loss(y, f(x)), base learner set {b(x, rm)}.  
  *Outputs*: final output function f(x).  
  *Algorithm*:
   - Initialize base learning f0 as f0(x) = 0
   - For m in {1,2,3,4,...,M}:  
   Minimize the below loss function:  
   ![img](https://latex.codecogs.com/svg.latex?%28%5Cbeta%20_m%2C%20r_m%29%20%3D%20%5Cunderset%7B%5Cbeta%20%2C%20r%7D%7BargMin%7D%20%28%5Csum_%7Bi%3D1%7D%5E%7BN%7DLoss%28y_i%2C%20f_%7Bm-1%7D&plus;%5Cbeta%20*%20b%28x_i%2C%20r%29%29%29)  
   Then update the function:  

     ![img](https://latex.codecogs.com/svg.latex?f_m%28x%29%20%3D%20f_%7Bm-1%7D%20&plus;%20%5Cbeta%20_m*b%28x%3Br_m%29)  

   - Then we have the final output model f(x) as below:  
   ![img](https://latex.codecogs.com/svg.latex?f%28x%29%20%3D%20f_M%28x%29%20%3D%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%5Cbeta_m%20*%20b%28x%3B%20r_m%29)  
- **b. What is exponential loss function and why we use it in AdaBoost**  
Suppose y belongs to {-1,1}, then the exponential loss function is as below:  

  ![img](https://latex.codecogs.com/svg.latex?Loss%28y%2C%20f%28x%29%29%20%3D%20E%28e%5E%7B-y*f%28x%29%7D%7Cx%29%20%3D%20%5Cmathbb%7BP%7D%28y%3D1%7Cx%29*e%5E%7B-f%28x%29%7D%20&plus;%20%5Cmathbb%7BP%7D%28y%3D-1%7Cx%29*e%5E%7Bf%28x%29%7D)  
  
  If we take the derivative and set it to 0, we will find out that when minimizing exponential loss function, we are actually like fitting a logistic regression, and we will reach the optimal Bayes error rate :
  
  ![img](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20E%28e%5E%7B-y*f%28x%29%7D%7Cx%29%20%7D%7B%5Cpartial%20%7Bf%28x%29%7D%7D%20%3D%20%5B-%5Cmathbb%7BP%7D%28y%3D1%7Cx%29*e%5E%7B-f%28x%29%7D%20&plus;%20%5Cmathbb%7BP%7D%28y%3D-1%7Cx%29*e%5E%7Bf%28x%29%7D%5D%20*%20f%28x%29%20%3D%200)  

  ![img](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20E%28e%5E%7B-y*f%28x%29%7D%7Cx%29%20%7D%7B%5Cpartial%20%7Bf%28x%29%7D%7D%20%3D%200%20%5CRightarrow%20f%28x%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20*%20log%5Cfrac%7B%5Cmathbb%7BP%7D%28y%3D1%7Cx%29%7D%7B%5Cmathbb%7BP%7D%28y%3D-1%7Cx%29%7D)  

  ![img](https://latex.codecogs.com/svg.latex?sign%28f%28x%29%29%20%3D%20sign%28%5Cfrac%7B1%7D%7B2%7D%20*%20log%5Cfrac%7B%5Cmathbb%7BP%7D%28y%3D1%7Cx%29%7D%7B%5Cmathbb%7BP%7D%28y%3D-1%7Cx%29%7D%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%201%5C%2C%20%5C%2C%20%5C%2C%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5Cmathbb%7BP%7D%28y%3D1%7Cx%29%20%3E%20%5Cmathbb%7BP%7D%28y%3D-1%7Cx%29%5C%5C%20-1%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5Cmathbb%7BP%7D%28y%3D1%7Cx%29%20%3C%20%5Cmathbb%7BP%7D%28y%3D-1%7Cx%29%20%5Cend%7Bmatrix%7D%5Cright.)  

  ![img](https://latex.codecogs.com/svg.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%201%5C%2C%20%5C%2C%20%5C%2C%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5Cmathbb%7BP%7D%28y%3D1%7Cx%29%20%3E%20%5Cmathbb%7BP%7D%28y%3D-1%7Cx%29%5C%5C%20-1%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5Cmathbb%7BP%7D%28y%3D1%7Cx%29%20%3C%20%5Cmathbb%7BP%7D%28y%3D-1%7Cx%29%20%5Cend%7Bmatrix%7D%5Cright.%20%3D%20%5Cunderset%7By%5Cin%20%5C%7B%201%2C-1%20%5C%7D%7D%7Bargmax%7D%20%5Cmathbb%7BP%7D%28y%7Cx%29)  

- **c. Math behind AdaBoost- how to compute the optimal parameters**  
Suppose that now we have finished m-1 iterations and successfully computed the f_{m-1} as below:  

  ![img](https://latex.codecogs.com/svg.latex?f_%7Bm-1%7D%28x%29%20%3D%20f_%7Bm-2%7D%28x%29%20&plus;%20%5Cbeta_%7Bm-1%7D%20*%20b%28x%3Br_%7Bm-1%7D%29%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bm-1%7D%5Cbeta_%7Bi%7D%20*%20b%28x%3Br_i%29)  

  Now we are at iteration m, and we want to find the optimal beta_m and b_m(x, r_m) (simplified as b_m(x)) to minimize our exponential loss. Attention: the output of b_m(x) belongs to {-1,1} instead of probability. Our target is as below:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%28%5Cbeta_%7Bm%7D%2C%20b_m%28x%29%29%20%3D%20%5Cunderset%7B%5Cbeta%20%2C%20b%28x%29%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%20&plus;%20%5Cbeta%20*%20b%28x_i%29%29%5C%5C%20%26%5CRightarrow%20%28%5Cbeta_%7Bm%7D%2C%20b_m%28x%29%29%20%3D%20%5Cunderset%7B%5Cbeta%20%2C%20b%28x%29%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20exp%28-y_i%20*%20f_%7Bm-1%7D%20-%20y_i%20*%20%5Cbeta%20*%20b%28x_i%29%20%29%5C%5C%20%26%5CRightarrow%20%28%5Cbeta_%7Bm%7D%2C%20b_m%28x%29%29%20%3D%20%5Cunderset%7B%5Cbeta%20%2C%20b%28x%29%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%20y_i%20*%20%5Cbeta%20*%20b%28x_i%29%20%29%20%5Cend%7Balign*%7D) 

  Where  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbar%7Bw_%7Bmi%7D%7D%20%3D%20exp%28-y_i%20*%20f_%7Bm-1%7D%28x_i%29%29)  

  *c.1. compute the optimal b_m(x)*
  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%20y_i%20*%20%5Cbeta%20*%20b%28x_i%29%29%20%5C%5C%20%26%5CRightarrow%20%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%28%5Csum_%7By%3Db%28x_i%29%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%20&plus;%20%5Csum_%7By%5Cneq%20b%28x_i%29%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28%5Cbeta%20%29%29%20%5C%5C%20%26%5CRightarrow%20%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%28%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28%5Cbeta%20%29%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29&plus;%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%20*%20%5Cmathbb%7BI%7D%28y_i%20%3D%20b%28x_i%29%29%29%20%5C%5C%20%26%5CRightarrow%20%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%28%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28%5Cbeta%20%29%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29&plus;%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%20*%20%281%20-%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29%20%29%29%20%5C%5C%20%26%5CRightarrow%20%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%28%28exp%28%5Cbeta%29%20-%20exp%28-%5Cbeta%20%29%29%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29&plus;%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%29%20%5Cend%7Balign*%7D)  

  Since beta > 0, so we will have:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%20%5Cunderset%7Bb%28x%29%20%7D%7Bargmin%7D%20%28%28exp%28%5Cbeta%29%20-%20exp%28-%5Cbeta%20%29%29%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29&plus;%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%29%5C%5C%20%26%20%5CRightarrow%20%5Cunderset%7Bb%28x%29%20%7D%7Bargmin%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29%5C%5C%20%26%20%5CRightarrow%20b_m%20%3D%20%5Cunderset%7Bb%28x%29%20%7D%7Bargmin%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%20%5Cend%7Balign*%7D)  

  
  *c.2. compute the optimal beta_m(x)*  
  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%20y_i%20*%20%5Cbeta%20*%20b%28x_i%29%29%20%5C%5C%20%26%5CRightarrow%20%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%28%5Csum_%7By%3Db%28x_i%29%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%20&plus;%20%5Csum_%7By%5Cneq%20b%28x_i%29%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28%5Cbeta%20%29%29%20%5C%5C%20%26%5CRightarrow%20%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%28%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28%5Cbeta%20%29%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29&plus;%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%20*%20%5Cmathbb%7BI%7D%28y_i%20%3D%20b%28x_i%29%29%29%20%5C%5C%20%26%5CRightarrow%20%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%28%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28%5Cbeta%20%29%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29&plus;%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%20*%20%281%20-%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29%20%29%29%20%5C%5C%20%26%5CRightarrow%20%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%28%28exp%28%5Cbeta%29%20-%20exp%28-%5Cbeta%20%29%29%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29&plus;%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%29%20%5Cend%7Balign*%7D)  

  So in order to find the optimal beta, we need set the derivative to 0:    
  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%20%5Cunderset%7B%5Cbeta%20%7D%7Bargmin%7D%20%28%28exp%28%5Cbeta%29%20-%20exp%28-%5Cbeta%20%29%29%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29&plus;%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%29%5C%5C%20%26%20%5CRightarrow%20%5Cfrac%7B%5Cpartial%20%28%28exp%28%5Cbeta%29%20-%20exp%28-%5Cbeta%20%29%29%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29&plus;%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20exp%28-%5Cbeta%20%29%29%7D%7B%5Cpartial%20%7B%5Cbeta%20%7D%7D%20%3D%200%5C%5C%20%26%20%5CRightarrow%20%28exp%28%5Cbeta%29%20&plus;%20exp%28-%5Cbeta%20%29%29%20*%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29%20-%20exp%28-%5Cbeta%20%29%20*%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cbar%7Bw_%7Bmi%7D%7D%3D%200%5C%5C%20%26%20%5CRightarrow%20%5Cfrac%7Bexp%28-%5Cbeta%29%7D%7Bexp%28%5Cbeta%29%20&plus;%20exp%28-%5Cbeta%20%29%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29%7D%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cbar%7Bw_%7Bmi%7D%7D%7D%20%5Cend%7Balign*%7D) 

  If we set the right hand side of the last equation to be e_m, then we will have:  
  
  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%20%5Cfrac%7Bexp%28-%5Cbeta%29%7D%7Bexp%28%5Cbeta%29%20&plus;%20exp%28-%5Cbeta%20%29%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29%7D%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cbar%7Bw_%7Bmi%7D%7D%7D%20%3D%20e_m%5C%5C%20%26%20%5CRightarrow%20%5Cfrac%7Bexp%28%5Cbeta%29%20&plus;%20exp%28-%5Cbeta%20%29%7D%20%7Bexp%28-%5Cbeta%29%7D%20%3D%20%5Cfrac%7B1%7D%7Be_m%7D%5C%5C%20%26%20%5CRightarrow%20exp%282*%5Cbeta%20%29%20&plus;%201%20%3D%20%5Cfrac%7B1%7D%7Be_m%7D%5C%5C%20%26%20%5CRightarrow%20exp%282*%5Cbeta%20%29%20%3D%20%5Cfrac%7B1%20-%20e_m%7D%7Be_m%7D%5C%5C%20%26%20%5CRightarrow%20%5Cbeta%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20*%20ln%28%5Cfrac%7B1-e_m%7D%7Be_m%7D%29%5C%5C%20%26%20%5CRightarrow%20%5Cbeta_m%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20*%20ln%28%5Cfrac%7B1-e_m%7D%7Be_m%7D%29%20%5Cend%7Balign*%7D)  


  *c.3. update the optimal w_{m+1, i}(x)* 

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%20%5Cbar%7Bw_%7Bm&plus;1%2C%20i%7D%7D%20%3D%20exp%28-y_i%20*%20f_m%28x_i%29%29%5C%5C%20%26%20%5CRightarrow%20%5Cbar%7Bw_%7Bm&plus;1%2C%20i%7D%7D%20%3D%20exp%28-y_i%20*%20f_%7Bm&plus;1%7D%28x%29%20-%20y_i%20*%20%5Cbeta_m%20*%20b_m%28x%29%29%5C%5C%20%26%20%5CRightarrow%20%5Cbar%7Bw_%7Bm&plus;1%2C%20i%7D%7D%20%3D%20exp%28-%20y_i%20*%20%5Cbeta_m%20*%20b_m%28x%29%29%20*%20%5Cbar%7Bw_%7Bm%2C%20i%7D%7D%20%5Cend%7Balign*%7D)  

  But we want to normalize this term to make w_{m, i}(x) into "probability":    

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%20Z_m%20%3D%20%5Csum_%7Bn%20%3D%201%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bm%2C%20i%7D%7D%20*%20exp%28-%5Cbeta_m%20*%20y_i%20*%20b_m%28x_i%29%29%5C%5C%20%26%20%5CRightarrow%20%5Cbar%7Bw_%7Bm&plus;1%2C%20i%7D%7D%20%3D%20%5Cfrac%7B%20exp%28-%20y_i%20*%20%5Cbeta_m%20*%20b_m%28x%29%29%20*%20%5Cbar%7Bw_%7Bm%2C%20i%7D%7D%20%7D%7BZ_m%7D%20%5Cend%7Balign*%7D)  

  And this will not affect the way we compute the beta_m & b_m(x) because this will not affect e_m:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26e_m%20%3D%20%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29%7D%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cbar%7Bw_%7Bmi%7D%7D%7D%20%5C%5C%20%26%3D%20%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cfrac%7B%5Cbar%7Bw_%7Bmi%7D%7D%7D%7BZ_m%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29%7D%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cfrac%7B%5Cbar%7Bw_%7Bmi%7D%7D%7D%7BZ_m%7D%7D%20%5C%5C%20%26%20%3D%20%5Cfrac%7B%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29%7D%7BZ_m%7D%7D%7B%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cbar%7Bw_%7Bmi%7D%7D%7D%7BZ_m%7D%7D%20%5Cend%7Balign*%7D)  

- **d. Actual Recurrent Algorithm for AdaBoost Tree**   
*Model Input*:  Dataset D: D = {(x1,y1), ..., (x_N, y_N), y_i belongs to {-1,1}}  
*Model Output*: Final classifier: G(x)  

  *Steps*:  
  
  (1) Initialize the weight T1:  
  ![img](https://latex.codecogs.com/svg.latex?T1%20%3D%20%28w_%7B11%7D%2C%20w_%7B12%7D%2C%20w_%7B13%7D%2C%20...%20%2C%20w_%7B1N%7D%29%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20w_%7B1%2Ci%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20i%3D1%2C2%2C3%2C4%2C...%2CN)  

  (2): for m = 1,2,3,4, ..., M (The final classifier is consists of M weak learners):  
  - use dataset D with weight T_m to train weak learner b_m:  
  ![img](https://latex.codecogs.com/svg.latex?b_m%28x%29%3A%20x%20%5Crightarrow%20%5C%7B-1%2C1%5C%7D)  
  
    ![img](https://latex.codecogs.com/svg.latex?b_m%20%3D%20%5Cunderset%7Bb%28x%29%20%7D%7Bargmin%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%5Cneq%20b%28x_i%29%29)  
  
  - compute the error rate e_m of b_m(x) on the dataset D:  

    ![img](https://latex.codecogs.com/svg.latex?e_m%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bmi%7D%7D%20*%20%5Cmathbb%7BI%7D%28y_i%20%5Cneq%20b_m%28x%29%29)  
  
  - compute the parameter of b_m(x):  
 
    ![img](https://latex.codecogs.com/svg.latex?%5Cbeta_m%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20*%20log%28%5Cfrac%7B1-e_m%7D%7Be_m%7D%29)  

  - update the weight W_{m+1} for the next weak learner:  

    ![img](https://latex.codecogs.com/svg.latex?T_%7Bm&plus;1%7D%20%3D%20%28w_%7Bm&plus;1%2C1%7D%2C%20w_%7Bm&plus;1%2C2%7D%2C%20w_%7Bm&plus;1%2C3%7D%2C%20...%20%2C%20w_%7Bm&plus;1%2CN%7D%29%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%20i%3D1%2C2%2C3%2C4%2C...%2CN)  
    
    ![img](https://latex.codecogs.com/svg.latex?w_%7Bm&plus;1%2Ci%7D%20%3D%20%5Cfrac%7Bw_%7Bm%2C1%7D%7D%7BZ_m%7D*%20exp%28-%5Cbeta_m%20*%20y_i%20*%20b_m%28x_i%29%29)

    ![img](https://latex.codecogs.com/svg.latex?Z_m%20%3D%20%5Csum_%7Bn%20%3D%201%7D%5E%7BN%7D%20%5Cbar%7Bw_%7Bm%2C%20i%7D%7D%20*%20exp%28-%5Cbeta_m%20*%20y_i%20*%20b_m%28x_i%29%29)  


  (3): Build up the final classifier:  

   ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26f%28x%29%20%3D%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%20%5Cbeta_m%20*%20b_m%28x%29%20%5C%5C%20%26%5CRightarrow%20G%28x%29%20%3D%20sign%28f%28x%29%29%20%3D%20sign%28%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%20%5Cbeta_m%20*%20b_m%28x%29%29%20%5Cend%7Balign*%7D)

- **e. A deeper look into how AdaBoost update the weights**  

  Remeber that:  

  ![img](https://latex.codecogs.com/svg.latex?w_%7Bm&plus;1%2Ci%7D%20%3D%20%5Cfrac%7Bw_%7Bm%2C1%7D%7D%7BZ_m%7D*%20exp%28-%5Cbeta_m%20*%20y_i%20*%20b_m%28x_i%29%29)  

  So for beta_m > 0, we will have:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20w_%7Bm&plus;1%2Ci%7D%20%3D%20w_m%20*%20exp%28-%5Cbeta_m%29%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20y_i%3Db_m%28x_i%29%20%5C%5C%20w_%7Bm&plus;1%2Ci%7D%20%3D%20w_m%20*%20exp%28%5Cbeta_m%29%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20y_i%20%5Cneq%20b_m%28x_i%29%20%5Cend%7Bmatrix%7D%5Cright.)  

  which means that, if the classification is correct, then the weight will decrease, but if the classification is wrong, then the weight will increase.  

**2. GBM (Gradient Boosting Machine)**

> One Sentence Summary:   
Continuously adding weak base learners to approximate the negative gradient so as to decrease total loss.  

- **a. Difference between AdaBoost & GBM**  
AdaBoost uses exponential loss and the exponential loss grows exponentially for negative values which makes it more sensitive to outliers. But GBM allows for using more robust loss functions as long as the loss function is continuously differentiable.  

  | Models      | Methods to correct previous errors    |
  | ---------- | :-----------:  |
  | AdaBoost     | Adding the weights for incorrectly classified samples, decreasing the weights for correctly classified samples.     | 
  | GBM     | Using the negative gradient as the indicator of the error that previous base learners made, fitting the next base learner to approximate the negative gradient of the previous base learners.  |  

- **b. Negative gradient in GBM**  

  Recall in the AdaBoost, we memtion the Forward Stagewise Additive Modeling. Suppose now we are in interation m:  

  ![img](https://latex.codecogs.com/svg.latex?f_m%28x%29%20%3D%20f_%7Bm-1%7D%20&plus;%20%5Cbeta_m%20*%20b_m%28x%29)  

  So in order to reduce loss (gradient descent):  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26Min%5C%2C%20%5C%2C%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_m%28x%29%29%5C%5C%20%26%5CRightarrow%20Min%5C%2C%20%5C%2C%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x%29%20&plus;%20b_m%28x%29%29%20%5C%5C%20%26%5CRightarrow%20b_m%28x%29%20%3D%20-%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x%29%29%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%5CRightarrow%20f_m%28x%29%20%3D%20f_%7Bm-1%7D%20-%20%5Cbeta_m%20*%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x%29%29%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5Cend%7Balign*%7D)  

- **c. GBM algorithm**  
*Model Input*:  Dataset D: D = {(x1,y1), ..., (x_N, y_N), y_i belongs to {-1,1}}  
*Model Output*: Final classifier/regressor: f_m(x) 

  *Steps*:  
  
  (1) Initialization:  

  ![img](https://latex.codecogs.com/svg.latex?f_0%28x%29%20%3D%20%5Cunderset%7B%5Cgamma%20%7D%7Bargmin%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20%5Cgamma%20%29)  

  (2) for m in 1,2,3,..., M:  

  - compute the negative gradient:  
  ![img](https://latex.codecogs.com/svg.latex?%5Ctilde%7By_i%7D%5E%7Bm%7D%20%3D%20-%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%29%20%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%2C%20%5C%2C%20%5C%2C%20%5C%2C%20%5C%2C%20i%20%3D%201%2C2%2C3%2C4%2C...%2CN)  

  - Fit a new tree by minimizing the square loss:  
  ![img](https://latex.codecogs.com/svg.latex?b_m%28x%29%20%3D%20%5Cunderset%7Bb%28x%29%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%20-%20b%28x%29%5D%5E2)  

  - Use linear search to find the best step (very similar to the learning rate concept in SGD):  
  ![img](https://latex.codecogs.com/svg.latex?%5Ceta_m%20%3D%20%5Cunderset%7B%20%5Ceta%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Ceta%20*%20b_m%28x_i%29%29)  

  - update the function f_m(x):  

    ![img](https://latex.codecogs.com/svg.latex?f_m%28x%29%20%3D%20f_%7Bm-1%7D%28x%29%20&plus;%20%5Ceta_m%20*%20b_m%28x%29)  

  (3) for m in 1,2,3,..., M:  

  ![img](https://latex.codecogs.com/svg.latex?f_M%28x%29%20%3D%20f_o%28x%29%20&plus;%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%20%5Ceta_m%20*%20b_m%28x%29)  

- **d. GBM regression tree algorithm**  

  *Model Input*:  Dataset D: D = {(x1,y1), ..., (x_N, y_N), y_i belongs to R}  
  *Model Output*: Final regressor: f_m(x)  
  *Loss Function*: Square Loss

  *Steps*:  
  
  (1) Initialization:  

  ![img](https://latex.codecogs.com/svg.latex?f_0%28x%29%20%3D%20%5Cunderset%7B%5Cgamma%20%7D%7Bargmin%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20%5Cgamma%20%29)  

  (2) for m in 1,2,3,..., M:  

  - compute the negative gradient:  
  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Ctilde%7By_i%7D%5E%7Bm%7D%20%3D%20-%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%29%20%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%3D%20-%20%5Cfrac%7B%5Cpartial%20%5By_i%20-%20f_%7Bm-1%7D%28x_i%29%5D%5E2%20%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%3D%20y_i%20-%20f_%7Bm-1%7D%28x%29%20%5Cend%7Balign*%7D)  

  - Fit a new CART tree by minimizing the square loss, suppose that the CART decision tree split the area into J different parts R_{j,m}:  
  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26b_m%28x%29%20%3D%20%5Cunderset%7Bb%28x%29%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%20-%20b%28x%29%5D%5E2%5C%5C%20%26%5CRightarrow%20%5C%7BR_%7Bj%2Cm%7D%5C%7D%20%3D%20%5Cunderset%7B%5C%7BR_%7Bj%2Cm%7D%5C%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%20-%20b_m%28x_i%2C%20%5C%7BR_%7Bj%2Cm%7D%5C%7D%29%5D%5E2%20%5Cend%7Balign*%7D)  

  - Instead of using linear search to find the optimal parameter for the whole tree, we decide to find the optimal parameters for each zone individually so as to achieve better results:  
  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Ceta_%7Bj%2Cm%7D%20%3D%20%5Cunderset%7B%5Ceta_%7Bj%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Ceta_%7Bj%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bjm%7D%29%29%5C%5C%20%26%5CRightarrow%20%5Ceta_%7Bj%2Cm%7D%20%3D%20%5Cunderset%7B%5Ceta_%7Bj%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5By_i%20-%20%28f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Ceta_%7Bj%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bjm%7D%29%29%5D%5E2%20%5Cend%7Balign*%7D)  

  - update the function f_m(x):  

    ![img](https://latex.codecogs.com/svg.latex?f_m%28x%29%20%3D%20f_%7Bm-1%7D%28x%29%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20%5Ceta_%7Bj%2Cm%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bj%2C%20m%7D%29)  

  (3) for m in 1,2,3,..., M:  

  ![img](https://latex.codecogs.com/svg.latex?f_M%28x%29%20%3D%20f_o%28x%29%20&plus;%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20%5Ceta_%7Bj%2Cm%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bj%2Cm%7D%29)  

- **e. GBM classification tree algorithm**  
  
  *Model Input*:  Dataset D: D = {(x1,y1), ..., (x_N, y_N), y_i belongs to {-1,1}}  
  *Model Output*: Final classifier: f_m(x)  
  *Loss Function*: Deviance Loss




