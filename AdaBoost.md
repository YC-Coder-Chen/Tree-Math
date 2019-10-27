Tree-Math
============
My study notes, contains Math behind all the mainstream tree-based machine learning models, covering basic decision tree models (ID3, C4.5, CART), boosted models (GBM, AdaBoost, Xgboost, LightGBM), bagging models (Random Forest).



Boosted Decision Tree
------------
**Adaptive Boosting (AdaBoost)**
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