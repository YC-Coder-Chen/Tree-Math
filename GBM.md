Tree-Math
============
My study notes, contains Math behind all the mainstream tree-based machine learning models, covering basic decision tree models (ID3, C4.5, CART), boosted models (GBM, AdaBoost, Xgboost, LightGBM), bagging models (Random Forest).



Boosted Decision Tree
------------
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
  
    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Ctilde%7By_i%7D%5E%7Bm%7D%20%3D%20-%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%29%20%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%3D%20-%20%5Cfrac%7B%5Cpartial%20%5By_i%20-%20f_%7Bm-1%7D%28x_i%29%5D%5E2%20%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%3D%20y_i%20-%20f_%7Bm-1%7D%28x_i%29%20%5Cend%7Balign*%7D)  

  - Fit a new CART tree by minimizing the square loss, suppose that the CART decision tree split the area into J different parts R_{j,m}:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26b_m%28x%29%20%3D%20%5Cunderset%7Bb%28x%29%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%20-%20b%28x%29%5D%5E2%5C%5C%20%26%5CRightarrow%20%5C%7BR_%7Bj%2Cm%7D%5C%7D%20%3D%20%5Cunderset%7B%5C%7BR_%7Bj%2Cm%7D%5C%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%20-%20b_m%28x_i%2C%20%5C%7BR_%7Bj%2Cm%7D%5C%7D%29%5D%5E2%20%5Cend%7Balign*%7D)  

  - Instead of using linear search to find the optimal parameter for the whole tree, we decide to find the optimal parameters for each zone individually so as to achieve better results:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Ceta_%7Bj%2Cm%7D%20%3D%20%5Cunderset%7B%5Ceta_%7Bj%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Ceta_%7Bj%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bjm%7D%29%29%5C%5C%20%26%5CRightarrow%20%5Ceta_%7Bj%2Cm%7D%20%3D%20%5Cunderset%7B%5Ceta_%7Bj%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5By_i%20-%20%28f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Ceta_%7Bj%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bjm%7D%29%29%5D%5E2%20%5Cend%7Balign*%7D)  

  - update the function f_m(x):  

    ![img](https://latex.codecogs.com/svg.latex?f_m%28x%29%20%3D%20f_%7Bm-1%7D%28x%29%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20%5Ceta_%7Bj%2Cm%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bj%2C%20m%7D%29)  

  (3) So we will output our final model f_M(x):  

  ![img](https://latex.codecogs.com/svg.latex?f_M%28x%29%20%3D%20f_o%28x%29%20&plus;%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20%5Ceta_%7Bj%2Cm%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bj%2Cm%7D%29)  

- **e. GBM classification tree algorithm**  
  
  *Model Input*:  Dataset D: D = {(x1,y1), ..., (x_N, y_N), y_i belongs to {-1,1}}  
  *Model Output*: Final classifier: f_m(x)  
  *Loss Function*: Deviance Loss

  *Steps*:  
  
  (1) Initialization:  

  - What is the loss function (deviance loss function):  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26z_i%5Em%20%3D%20f_m%28x_i%29%20%3D%20f_%7Bm-1%7D%28x_i%29%20&plus;%20b_m%28x_i%29%20%5C%5C%20%26p_i%5Em%20%3D%20p_i%5Em%28y_i%3D1%20%7C%20x_i%29%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20exp%28-z_i%5Em%29%7D%20%5C%5C%20%26%5CRightarrow%20Loss%28p%5Em%2C%20y%29%20%3D%20-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28y_i%20*%20log%28p_i%5Em%29%20&plus;%20%281-y_i%29%20*%20log%281-p_i%5Em%29%29%20%5Cend%7Balign*%7D)  

  - So each time, we are using sigmoid(fm(x)) to proximate the probability, which means that we are using iteration to proximate the log(p/1-p).  

    Suppose now we are at time 0, and we want a constant f0(x) to minimize our loss function during the initialization.  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cunderset%7Bf_0%20%7D%7Bargmin%7D%20%5C%2C%20%5C%2C%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28p_i%5E0%2C%20y_i%29%20%5C%5C%20%26%5CRightarrow%20%5Cunderset%7Bf_0%20%7D%7Bargmin%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28%5Cfrac%7B1%7D%7B1&plus;exp%28-f_0%28x_i%29%29%7D%20%2C%20y_i%29%20%5Cend%7Balign*%7D)  

  - We find this f0 by setting the derivative to be 0:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cunderset%7Bf_0%20%7D%7Bargmin%7D%20%5C%2C%20%5C%2C%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28p_i%5E0%2C%20y_i%29%20%5C%5C%20%26%5CRightarrow%20%5Cfrac%7B%5Cpartial%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28p_i%5E0%2C%20y_i%29%20%7D%7B%5Cpartial%20%7Bf_%7B0%7D%28x%29%7D%7D%20%3D%200%20%5C%5C%20%26%5CRightarrow%20%5Cfrac%7B%5Cpartial%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28p_i%5E0%2C%20y_i%29%20%7D%7B%5Cpartial%20%7Bf_%7B0%7D%28x%29%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28p_i%5E0%2C%20y_i%29%20%7D%7B%5Cpartial%20%7Bp%5E0%7D%7D%20%5C%2C%20%5C%2C%20%5Cfrac%7B%5Cpartial%20%7Bp%5E0%7D%7D%7B%5Cpartial%20%7Bf_0%28x%29%7D%7D%20%3D%200%20%5C%5C%20%26%5CRightarrow%20-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28%5Cfrac%7By_i%7D%7Bp%5E0%7D%20-%20%5Cfrac%7B1-y_i%7D%7B1-p%5E0%7D%29%20*%20%28p%5E0%281-p%5E0%29%29%3D%200%20%5C%5C%20%26%5CRightarrow%20-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%281-p%5E0%29%20*%20y_i%20-%20p%5E0%20*%20%281-y_i%29%5D%3D%200%20%5C%5C%20%26%5CRightarrow%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28p%5E0%20-%20y_i%29%3D%200%20%5C%5C%20%26%5CRightarrow%20p%5E0%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7D%7Bm%7D%20%5Cend%7Balign*%7D)  

  - So after computing p0, we can compute the constant f0(x):  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26p%5E0%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7D%7Bm%7D%20%5C%5C%20%26%5CRightarrow%20%5Cfrac%7B1%7D%7B1&plus;exp%28-f_0%28x%29%29%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7D%7Bm%7D%20%5C%5C%20%26%5CRightarrow%201&plus;exp%28-f_0%28x%29%29%20%3D%20%5Cfrac%7Bm%7D%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%20%7D%20%5C%5C%20%26%5CRightarrow%20exp%28-f_0%28x%29%29%20%3D%20%5Cfrac%7Bm-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%7D%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%20%7D%20%5C%5C%20%26%5CRightarrow%20exp%28f_0%28x%29%29%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%20%7D%7Bm-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%7D%20%5C%5C%20%26%5CRightarrow%20f_o%28x%29%20%3D%20log%28%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%20%7D%7Bm-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%7D%29%20%5Cend%7Balign*%7D)  

   (2) for m in 1,2,3,..., M:  

     ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%20Z_i%5Em%20%3D%20f_m%28x_i%29%20%3D%20f_%7Bm-1%7D%28x_i%29%20&plus;%20b_m%28x_i%29%5C%5C%20%26%20p_i%5Em%20%3D%20%5Cfrac%7B1%7D%7B1&plus;exp%28-z_i%5Em%29%7D%20%5Cend%7Balign*%7D)  

  - compute the negative gradient:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Ctilde%7By_i%7D%5E%7Bm%7D%20%3D%20-%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%29%20%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%3D%20-%20%5Cfrac%7B%5Cpartial%20%28-%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28y_i%20*%20log%28p_i%5E%7Bm-1%7D%29%20&plus;%20%281-y_i%29%29*log%281-p_i%5E%7Bm-1%7D%29%20%29%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%3D%20-%20%5Cfrac%7B%5Cpartial%20%28-%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28y_i%20*%20log%28p_i%5E%7Bm-1%7D%29%20&plus;%20%281-y_i%29%29*log%281-p_i%5E%7Bm-1%7D%29%20%29%7D%7B%5Cpartial%20p%5E%7Bm-1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20p%5E%7Bm-1%7D%7D%7B%5Cpartial%20f_%7Bm-1%7D%28x%29%7D%20%5C%5C%20%26%3D%20-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28%5Cfrac%7By_i%7D%7Bp%5E%7Bm-1%7D%7D%20-%20%28%5Cfrac%7B1-y_i%7D%7B1-p%5E%7Bm-1%7D%7D%29%20%29%20*%20%28p%5Em*%281-p%5E%7Bm-1%7D%29%29%5C%5C%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28p%5E%7Bm-1%7D%20-%20y_i%29%20%5C%5C%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28%5Cfrac%7B1%7D%7B1&plus;exp%28-f_%7Bm-1%7D%28x%29%29%7D%20-%20y_i%29%20%5C%5C%20%5Cend%7Balign*%7D)  

  - Fit a new CART tree by minimizing the square loss, suppose that the CART decision tree split the area into J different parts R_{j,m}:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26b_m%28x%29%20%3D%20%5Cunderset%7Bb%28x%29%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%20-%20b%28x%29%5D%5E2%5C%5C%20%26%5CRightarrow%20%5C%7BR_%7Bj%2Cm%7D%5C%7D%20%3D%20%5Cunderset%7B%5C%7BR_%7Bj%2Cm%7D%5C%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%20-%20b_m%28x_i%2C%20%5C%7BR_%7Bj%2Cm%7D%5C%7D%29%5D%5E2%20%5Cend%7Balign*%7D)  

  - Instead of using linear search to find the optimal parameter for the whole tree, we decide to find the optimal parameters for each zone individually so as to achieve better results:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Ceta_%7Bj%2Cm%7D%20%3D%20%5Cunderset%7B%5Ceta_%7Bj%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Ceta_%7Bj%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bjm%7D%29%29%5C%5C%20%26%5CRightarrow%20%5Ceta_%7Bj%2Cm%7D%20%3D%20%5Cunderset%7B%5Ceta_%7Bj%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5By_i%20-%20%28f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Ceta_%7Bj%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bjm%7D%29%29%5D%5E2%20%5Cend%7Balign*%7D)  

  - update the function f_m(x):  

    ![img](https://latex.codecogs.com/svg.latex?f_m%28x%29%20%3D%20f_%7Bm-1%7D%28x%29%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20%5Ceta_%7Bj%2Cm%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bj%2C%20m%7D%29)  

  (3) So we will output our final model f_M(x) and final predicted probability p_M(x):  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26f_M%28x%29%20%3D%20f_o%28x%29%20&plus;%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20%5Ceta_%7Bj%2Cm%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bj%2Cm%7D%29%5C%5C%20%26%5CRightarrow%20p_M%28y_i%3D1%7Cx_i%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;exp%28-f_M%28x%29%29%7D%20%5C%5C%20%5Cend%7Balign*%7D)  