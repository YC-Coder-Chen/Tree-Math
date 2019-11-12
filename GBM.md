Tree-Math
============
My study notes, contains Math behind all the mainstream tree-based machine learning models, covering basic decision tree models (ID3, C4.5, CART), boosted models (GBM, AdaBoost, Xgboost, LightGBM), bagging models (Random Forest).



Boosted Decision Tree
------------
**GBM (Gradient Boosting Machine)**

> One Sentence Summary:   
Continuously adding weak base learners to approximate the negative gradient so as to decrease total loss.  

- **a. Difference between AdaBoost & GBM**  
AdaBoost uses exponential loss and the exponential loss grows exponentially for negative values which makes it more sensitive to outliers. But GBM allows for using more robust loss functions as long as the loss function is continuously differentiable.  

  | Models      | Methods to correct previous errors   |
  | ---------- | :-----------:  |
  | AdaBoost     | Adding the weights for incorrectly classified samples, decreasing the weights for correctly classified samples.     | 
  | GBM     | Using the negative gradient as the indicator of the error that previous base learners made, fitting the next base learner to approximate the negative gradient of the previous base learners.  |  

- **b. Negative gradient in GBM**  

  Recall in the AdaBoost, we memtion the Forward Stagewise Additive Modeling. Suppose now we are in interation m:  

  ![img](https://latex.codecogs.com/svg.latex?f_m%28x%29%20%3D%20f_%7Bm-1%7D%20&plus;%20%5Cbeta_m%20*%20b_m%28x%29)  

  We want to reduce loss similar to AdaBoost:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26Min%5C%2C%20%5C%2C%5C%2C%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_m%28x%29%29%5C%5C%20%26%5CRightarrow%20Min%5C%2C%20%5C%2C%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x%29%20&plus;%20%5Cbeta_m%20*%20b_m%28x%29%29%20%5C%5C%20%5Cend%7Balign*%7D)  

  But here the problem is different from the case in Adaboost. In AdaBoost we know exactly the loss function (exponential loss) so we can find the exact optimal bm(x). But here, in GBM, we want to be able to solve any loss function as long as it is differentiable. So we adopt an idea similar to gradient descent to find the optimal bm(x) by setting bm(x) to proximate the direction of negative gradient:  

  ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26Min%5C%2C%20%5C%2C%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x%29%20&plus;%20%5Cbeta_m%20*%20b_m%28x%29%29%20%5C%5C%20%26%5CRightarrow%20b_m%28x%29%20%3D%20a%5C%20constant%20*%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x%29%29%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%5CRightarrow%20f_m%28x%29%20%3D%20f_%7Bm-1%7D%20&plus;%20a%5C%20constant%20*%20%5Cbeta_m%20*%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x%29%29%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%5CRightarrow%20f_m%28x%29%20%3D%20f_%7Bm-1%7D%20&plus;%20%5Ceta_m%20*%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x%29%29%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5Cend%7Balign*%7D)  

  where ![img](https://latex.codecogs.com/svg.latex?%5Ceta_m) is a parameter similar to learning rate, but it is negative here.  

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

    ![img](https://latex.codecogs.com/svg.latex?%5Ceta_m%20%3D%20%5Cunderset%7B%20%5Ceta%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%20&plus;%5C%2C%20%5Ceta%20*%20b_m%28x_i%29%29)  

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
  
    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Ctilde%7By_i%7D%5E%7Bm%7D%20%3D%20-%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%29%20%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%3D%20-%20%5Cfrac%7B%5Cpartial%20%5By_i%20-%20f_%7Bm-1%7D%28x_i%29%5D%5E2%20%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%3D%202%20*%20%28y_i%20-%20f_%7Bm-1%7D%28x_i%29%29%20%5Cend%7Balign*%7D)  

  - Fit a new CART tree by minimizing the square loss, suppose that the CART decision tree split the area into J different parts R_{j,m}:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26b_m%28x%29%20%3D%20%5Cunderset%7Bb%28x%29%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%5Em%20-%20b%28x%29%5D%5E2%5C%5C%20%26%5CRightarrow%20%5C%2C%20%5C%7BR_%7Bj%2Cm%7D%5C%7D%20%3D%20%5Cunderset%7B%5C%7BR_%7Bj%2Cm%7D%5C%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%5Em%20-%20b_m%28x_i%2C%20%5C%7BR_%7Bj%2Cm%7D%5C%7D%29%5D%5E2%20%5Cend%7Balign*%7D)  

  - Instead of using linear search to find the optimal parameter ![img](https://latex.codecogs.com/svg.latex?%5Ceta_m) for the whole tree, we decide to find the optimal parameters ![img](https://latex.codecogs.com/svg.latex?%5Cgamma_%7Bj%2Cm%7D) for each zone inside the tree individually so as to achieve better results:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cgamma_%7Bj%2Cm%7D%20%3D%20%5Cunderset%7B%5Ceta_%7Bj%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Cgamma_%7Bj%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bjm%7D%29%29%5C%5C%20%26%5CRightarrow%20%5Cgamma_%7Bj%2Cm%7D%20%3D%20%5Cunderset%7B%5Ceta_%7Bj%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5By_i%20-%20%28f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Cgamma_%7Bj%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bjm%7D%29%29%5D%5E2%20%5Cend%7Balign*%7D)  

  - update the function f_m(x):  

    ![img](https://latex.codecogs.com/svg.latex?f_m%28x%29%20%3D%20f_%7Bm-1%7D%28x%29%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20%5Cgamma_%7Bj%2Cm%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bj%2C%20m%7D%29)  

  (3) So we will output our final model f_M(x):  

  ![img](https://latex.codecogs.com/svg.latex?f_M%28x%29%20%3D%20f_o%28x%29%20&plus;%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20%5Cgamma_%7Bj%2Cm%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bj%2Cm%7D%29)  

- **e. GBM classification tree algorithm**  
  
  *Model Input*:  Dataset D: D = {(x1,y1), ..., (x_N, y_N), y_i belongs to {-1,1}}  
  *Model Output*: Final classifier: f_m(x)  
  *Loss Function*: Deviance Loss

  *Steps*:  
  
  (1) Initialization:  

  - What is the loss function (deviance loss function):  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26p_i%5Em%20%3D%20p_i%5Em%28y_i%3D1%20%7C%20x_i%29%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20exp%28-f_m%28x_i%29%29%7D%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20exp%28-%20f_%7Bm-1%7D%28x_i%29%20-%20b_m%28x_i%29%29%7D%20%5C%5C%20%26%5CRightarrow%20Loss%28p%5Em%2C%20y%29%20%3D%20-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28y_i%20*%20log%28p_i%5Em%29%20&plus;%20%281-y_i%29%20*%20log%281-p_i%5Em%29%29%20%5Cend%7Balign*%7D)  

  - So each time, we are using sigmoid(fm(x)) to proximate the probability, which means that we are using iteration to proximate the log(p/1-p).  

    Suppose now we are at time 0, and we want a constant f0(x) to minimize our loss function during the initialization.  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cunderset%7Bf_0%20%7D%7Bargmin%7D%20%5C%2C%20%5C%2C%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28p_i%5E0%2C%20y_i%29%20%5C%5C%20%26%5CRightarrow%20%5Cunderset%7Bf_0%20%7D%7Bargmin%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28%5Cfrac%7B1%7D%7B1&plus;exp%28-f_0%28x_i%29%29%7D%20%2C%20y_i%29%20%5Cend%7Balign*%7D)  

  - We find this f0 by setting the derivative to be 0:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cunderset%7Bf_0%20%7D%7Bargmin%7D%20%5C%2C%20%5C%2C%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28p_i%5E0%2C%20y_i%29%20%5C%5C%20%26%5CRightarrow%20%5Cfrac%7B%5Cpartial%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28p_i%5E0%2C%20y_i%29%20%7D%7B%5Cpartial%20%7Bf_%7B0%7D%28x%29%7D%7D%20%3D%200%20%5C%5C%20%26%5CRightarrow%20%5Cfrac%7B%5Cpartial%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28p_i%5E0%2C%20y_i%29%20%7D%7B%5Cpartial%20%7Bf_%7B0%7D%28x%29%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28p_i%5E0%2C%20y_i%29%20%7D%7B%5Cpartial%20%7Bp%5E0%7D%7D%20%5C%2C%20%5C%2C%20%5Cfrac%7B%5Cpartial%20%7Bp%5E0%7D%7D%7B%5Cpartial%20%7Bf_0%28x%29%7D%7D%20%3D%200%20%5C%5C%20%26%5CRightarrow%20-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28%5Cfrac%7By_i%7D%7Bp%5E0%7D%20-%20%5Cfrac%7B1-y_i%7D%7B1-p%5E0%7D%29%20*%20%28p%5E0%281-p%5E0%29%29%3D%200%20%5C%5C%20%26%5CRightarrow%20-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%281-p%5E0%29%20*%20y_i%20-%20p%5E0%20*%20%281-y_i%29%5D%3D%200%20%5C%5C%20%26%5CRightarrow%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28p%5E0%20-%20y_i%29%3D%200%20%5C%5C%20%26%5CRightarrow%20p%5E0%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7Dy_i%7D%7Bn%7D%20%5Cend%7Balign*%7D)  

  - So after computing p0, we can compute the constant f0(x):  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26p%5E0%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7Dy_i%7D%7Bn%7D%20%5C%5C%20%26%5CRightarrow%20%5Cfrac%7B1%7D%7B1&plus;exp%28-f_0%28x%29%29%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7Dy_i%7D%7Bn%7D%20%5C%5C%20%26%5CRightarrow%201&plus;exp%28-f_0%28x%29%29%20%3D%20%5Cfrac%7Bn%7D%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%20%7D%20%5C%5C%20%26%5CRightarrow%20exp%28-f_0%28x%29%29%20%3D%20%5Cfrac%7Bn-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%7D%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%20%7D%20%5C%5C%20%26%5CRightarrow%20exp%28f_0%28x%29%29%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%20%7D%7Bn-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%7D%20%5C%5C%20%26%5CRightarrow%20f_o%28x%29%20%3D%20log%28%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%20%7D%7Bn-%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%7D%29%20%5Cend%7Balign*%7D)  

   (2) for m in 1,2,3,..., M:  

     ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26p_i%5Em%20%3D%20p_i%5Em%28y_i%3D1%20%7C%20x_i%29%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20exp%28-f_m%28x_i%29%29%7D%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20exp%28-%20f_%7Bm-1%7D%28x_i%29%20-%20b_m%28x_i%29%29%7D%20%5C%5C%20%5Cend%7Balign*%7D)  

  - compute the negative gradient:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Ctilde%7By_i%7D%5E%7Bm%7D%20%3D%20-%20%5Cfrac%7B%5Cpartial%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%29%20%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%3D%20-%20%5Cfrac%7B%5Cpartial%20%5By_i%20*%20log%28p_i%5E%7Bm-1%7D%29%20&plus;%20%281-y_i%29*log%281-p_i%5E%7Bm-1%7D%29%5D%20%7D%7B%5Cpartial%20%7Bf_%7Bm-1%7D%28x%29%7D%7D%20%5C%5C%20%26%3D%20-%20%5Cfrac%7B%5Cpartial%20%5B-y_i%20*%20log%28p_i%5E%7Bm-1%7D%29%20&plus;%20%281-y_i%29*log%281-p_i%5E%7Bm-1%7D%29%5D%29%7D%7B%5Cpartial%20p%5E%7Bm-1%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20p%5E%7Bm-1%7D%7D%7B%5Cpartial%20f_%7Bm-1%7D%28x%29%7D%20%5C%5C%20%26%3D%20-%20%5B%5Cfrac%7By_i%7D%7Bp%5E%7Bm-1%7D%7D%20-%20%28%5Cfrac%7B1-y_i%7D%7B1-p%5E%7Bm-1%7D%7D%29%5D%20*%20%28p%5Em*%281-p%5E%7Bm-1%7D%29%29%5C%5C%20%26%3D%20p%5E%7Bm-1%7D%20-%20y_i%20%5C%5C%20%26%3D%20%5Cfrac%7B1%7D%7B1&plus;exp%28-f_%7Bm-1%7D%28x%29%29%7D%20-%20y_i%20%5C%5C%20%5Cend%7Balign*%7D)  

  - Fit a new CART tree by minimizing the square loss, suppose that the CART decision tree split the area into J different parts R_{j,m}:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26b_m%28x%29%20%3D%20%5Cunderset%7Bb%28x%29%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%5Em%20-%20b%28x%29%5D%5E2%5C%5C%20%26%5CRightarrow%20%5C%2C%20%5C%7BR_%7Bj%2Cm%7D%5C%7D%20%3D%20%5Cunderset%7B%5C%7BR_%7Bj%2Cm%7D%5C%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5B%5Ctilde%7By_i%7D%5Em%20-%20b_m%28x_i%2C%20%5C%7BR_%7Bj%2Cm%7D%5C%7D%29%5D%5E2%20%5Cend%7Balign*%7D)   

  - Instead of using linear search to find the optimal parameter ![img](https://latex.codecogs.com/svg.latex?%5Ceta_m) for the whole tree, we decide to find the optimal parameters ![img](https://latex.codecogs.com/svg.latex?%5Cgamma_%7Bj%2Cm%7D) for each zone inside the tree individually so as to achieve better results:  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26%5Cgamma_%7Bj%2Cm%7D%20%3D%20%5Cunderset%7B%5Ceta_%7Bj%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20Loss%28y_i%2C%20f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Cgamma_%7Bj%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bjm%7D%29%29%5C%5C%20%26%5CRightarrow%20%5Cgamma_%7Bj%2Cm%7D%20%3D%20%5Cunderset%7B%5Ceta_%7Bj%7D%20%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5By_i%20-%20%28f_%7Bm-1%7D%28x_i%29%20&plus;%20%5Cgamma_%7Bj%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bjm%7D%29%29%5D%5E2%20%5Cend%7Balign*%7D)    

  - update the function f_m(x):  

    ![img](https://latex.codecogs.com/svg.latex?f_m%28x%29%20%3D%20f_%7Bm-1%7D%28x%29%20&plus;%20%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20%5Cgamma_%7Bj%2Cm%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bj%2C%20m%7D%29)  

  (3) So we will output our final model f_M(x) and final predicted probability p_M(x):  

    ![img](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%26f_M%28x%29%20%3D%20f_o%28x%29%20&plus;%20%5Csum_%7Bm%3D1%7D%5E%7BM%7D%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20%5Cgamma_%7Bj%2Cm%7D%20*%20%5Cmathbb%7BI%7D%28x_i%20%5Cin%20R_%7Bj%2Cm%7D%29%5C%5C%20%26%5CRightarrow%20p_M%28y_i%3D1%7Cx_i%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;exp%28-f_M%28x%29%29%7D%20%5C%5C%20%5Cend%7Balign*%7D)  

**Scikit-learn Application**
> **GradientBoostingRegressor**:   
****class**** sklearn.ensemble.GradientBoostingRegressor(loss=’ls’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)  

  The base learners of GBR are regression trees, which are discussed in the CART decision tree document. Therefore only hyperparameters of GBR itself are introduced below. Please refer to the previous document for more information about CART decision tree parameters.

- **loss** : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}, optional (default=’ls’)  
  loss function to be optimized. Default 'ls' refers to least squares regression, which is also used in our math derivation. Please refer to [Prince Grover](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0)'s blog for introduction of different loss functions. 

- **learning_rate** : float, optional (default=0.1)  
  Learning rate shrinks the contribution of each tree by learning_rate. Here we use the same example as AdaBoost, suppose previous prediction fm-1 = 1, learning rate = 0.1, and next tree correction = 0.5, then the updated prediction fm = 1 + 0.1 * 0.5 = 1.05. Lower learning rate sometimes makes the model generalize better, but also requires higher number of estimators.

- **n_estimators** : int (default=100)  
  Numbers of estimators in the model. It refers to M in the formular. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.

- **subsample** : loat, optional (default=1.0)
  The fraction of samples to be randomely selected and used in each weak base learner. Subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias. Generally ~0.8 works fine. 

- **init** : estimator or ‘zero’, optional (default=None)  
  An estimator object to compute the initial predictions f0(x). If ‘zero’, the initial raw predictions are set to zero. By default a DummyEstimator is used, predicting either the average target value (for loss=’ls’), or a quantile for the other losses.

- **alpha** : float (default=0.9)  
  The alpha-quantile of the huber loss function and the quantile loss function. Only if loss='huber' or loss='quantile'.

- **verbose** : int, default: 0
  Enable verbose output. Default doesn't generate output. If 1 then it prints progress and performance once in a while (the more trees the lower the frequency). If greater than 1 then it prints progress and performance for every tree.

- **warm_start** : bool, default: False  
  When warm_start is true, the existing fitted model attributes are used to initialise the new model in a subsequent call to fit. In other words, we can add more trees on a trained model.

- **validation_fraction** : float, optional, default 0.1  
  The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.

- **n_iter_no_change**  : int, default None  
  n_iter_no_change is used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous n_iter_no_change numbers of iterations.

- **tol**   : float, optional, default 1e-4  
  Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.
 


**Reference**  

1. Friedman J H. Greedy function approximation: a gradient boosting machine[J]. Annals of statistics, 2001: 1189-1232.
2. Friedman J H. Stochastic gradient boosting[J]. Computational statistics & data analysis, 2002, 38(4): 367-378.
3. Hang Li. Statistical Learning Method[M]. Tsinghua University Press, 2019. [Chinese]
4. Zhihua Zhou. Machine Learning[M]. Tsinghua University Press, 2018. [Chinese]
5. Mason L, Baxter J, Bartlett P L, et al. Boosting algorithms as gradient descent[C]//Advances in neural information processing systems. 2000: 512-518.
6. Wikipedia contributors. Gradient boosting. Wikipedia, The Free Encyclopedia. October 21, 2019, 23:33 UTC. Available at: https://en.wikipedia.org/w/index.php?title=Gradient_boosting&oldid=922411214. Accessed November 11, 2019.
7. https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab
8. https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d
9. https://zhuanlan.zhihu.com/p/38329631 [Chinese]
10. https://zhuanlan.zhihu.com/p/43940320 [Chinese]
11. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
12. https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/