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
Continously adding weak base learnings to the existing model and adaptively adjusting the weight for weak base learnings.

- **a. What is Forward Stagewise Additive Modeling**  
Suppose b(x;rm) is a base learning controled by parameter rm.  
Beta_m is the parameter controled how each weak base learner is added.  
Then the final model f(x) based on M weak base learners will be as below: ![img](https://latex.codecogs.com/svg.latex?f%28x%29%20%3D%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%20%5Cbeta_%7Bm%7D%20*%20b%28x%3Br_m%29)  

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
  