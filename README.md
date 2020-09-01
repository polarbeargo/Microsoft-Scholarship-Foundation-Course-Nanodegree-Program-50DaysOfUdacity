# Microsoft Scholarship Foundation Course Nanodegree Program 50DaysOfUdacity


[image1]: ./images/score.png    
[image2]: ./images/evaluate.png  
[image3]: ./images/ApproachesToML.png
[image4]: ./images/import.png
[image5]: ./images/transformExport.png
[image6]: ./images/fixDtypeToString.png
[image7]: ./images/datasetsVersion.png
[image8]: ./images/TaipeiGoSmart.png  
[image9]: ./images/featureSelection.png  
[image10]: ./images/EvaluationMetricsforClassification.png 
[image11]: ./images/TrainEvaluate.png
[image12]: ./images/confusionMatrix.png
[image13]: ./images/ROCPrecisionRecallLiftCurve.png  
[image14]: ./images/TwoClassBoostedDecisionTree.png
[image15]: ./images/TwoClassDecisionTreeResultVis.png
[image16]: ./images/confusionMatrixTwoClassBoostedDtree.png  
[image17]: ./images/AutomatedML.png
[image18]: ./images/AutoMLMetrics.png
[image19]: ./images/AutoMLMetricsPlot.png
[image20]: ./images/AutoMLMetricsPlot2.png  
[image21]: ./images/twoclasslogistic.png
[image22]: ./images/twoClassBoostedDecisionTreePrecision.png
[image23]: ./images/AIDLMLDiagram.png  
[image24]: ./images/Eid.jpeg  
[image25]: ./images/specialCase.png
[image26]: ./images/trainingClassification.png
[image27]: ./images/PredictingClassification.png
[image28]: ./images/machineryMaintenance.png  
[image29]: ./images/MultiClassClassifiersPerformance.png
[image30]: ./images/classifierAutomatedML.png
[image31]: ./images/classificationWithCNN.png
[image32]: ./images/ImageSearchWithAutoencoder.png  
[image33]: ./images/MLpipelines.png
[image34]: ./images/AdvancedModeling.png  
[image35]: ./images/IMG_8345.jpeg
[image36]: ./images/IMG_8346.jpeg
[image37]: ./images/IMG_8347.jpeg
[image38]: ./images/IMG_8348.jpeg
[image39]: ./images/RegressorPerformance.png
[image40]: ./images/RegressorAutoML.png
[image41]: ./images/ReviewBestModelPerformance.png
[image42]: ./images/AutoMLRegressorMetrics.png
[image43]: ./images/RegressorPredictTrue.png
[image44]: ./images/DesignerView.png
[image45]: ./images/TrainClusterModel.png
[image46]: ./images/scoreNeuralNet.png
[image47]: ./images/evalNeuralNet.png
[image48]: ./images/scoreSVD.png
[image49]: ./images/evalRecommender.png
[image50]: ./images/recommender.png
[image51]: ./images/Kubernetes.png
[image52]: ./images/Production.png
[image53]: ./images/overtime.png
[image54]: ./images/TextClassifier.png
[image55]: ./images/VisualChallenge.png  
[image56]: ./images/forcast.png
[image57]: ./images/computeInstance.png
[image58]: ./images/ManageComputeInstanceJupyternotebook.png
[image59]: ./images/ManageNotebook.png
[image60]: ./images/deployModelAsWebservice.png
[image61]: ./images/trainDeployWithComputerInstance.png
[image62]: ./images/run.png
[image63]: ./images/Experiments.png
[image64]: ./images/ExperimentsMetrics.png
[image65]: ./images/retrainPkl.png
[image66]: ./images/modelExplain.png
[image67]: ./images/modelExplainPlot.png
[image68]: ./images/pong.gif  
[image69]: ./images/FiftyBird.gif 
[image70]: ./images/50daysofUdacityBadge.png

Participation in the Microsoft Scholarship Foundation course Nanodegree Program 50 days of Udacity challenge

Follow Udacity Git Commit Message Style Guide: https://udacity.github.io/git-styleguide/  :heart:   

![][image70] 

### Day 1 : 14/07/2020 
#### Polarbeargo 

* Reading Lesson 2 Introduction to Machine Learning section 20 Linear Regression.
* I very love the Text data section where I have chance to read how to process text data with normalization(Lemmatization, Tokenization) and Vectorization where 
  I got understand about:
    * Term Frequency-Inverse Document Frequency (TF-IDF) vectorization
    * Word embedding, as done with Word2vec or Global Vectors (GloVe)
* Section 8 Scaling data with Standardization and Normalization where I'm very glad that I have chance to read this section :)

### Day 2 : 15/07/2020 
#### Polarbeargo 

* Reanding Lesson 2 Introduction to Machine Learning section 20 Linear Regression grasp the following concepts to talk to:
  * "To train a linear regression model" means to learn the coefficients and bias that best fit the data. 
  * The process of finding the best model is essentially a process of finding the coefficients and bias that minimize this error. 
  * Preparing the Data:  
        * Linear assumption  
        * Remove collinearity  
        * Gaussian (normal) distribution  
        * Rescale data  
        * Remove noise    
  * Calculating the Coefficients: Choose a cost function (like RMSE) to calculate the error and then minimize that error in order to arrive at a line of best fit that models the training data and can be used to make predictions.  
* Writing Quiz "Linear Regression: Check Your Understanding".

### Day 3 : 16/07/2020 
#### Polarbeargo

* Reading Lesson 2 Introduction to Machine Learning section 23 lab Instruction and writing section 24 lab: Train a Linear Regression Model:  

Score Model      |  Evaluate Model
:-------------------------:|:-------------------------:
![][image1]                | ![][image2]

### Day 4 : 17/07/2020 
#### Polarbeargo

* Reading Lesson 2 Introduction to Machine Learning section 25 Learning function keypoints:  
  * Irreducible error in Learning function is caused by the data collection process—such as when we don't have enough data or don't have enough data features.  
  * Model error measures how much the prediction made by the model is different from the true output. The model error is generated from the model and can be reduced during the model learning process.  
  
* Reading section 26 parametric vs. Non-parametric keypoints:   
  * Parametric Machine Learning Algorithms: 
     * Making assumption about the mapping function and have a fixed number of parameters. 
     * No matter how much data is used to learn the model, this will not change how many parameters the algorithm has. 
     * With a parametric algorithm, we are selecting the form of the function and then learning its coefficients using the training data.
     
     Benefits:

     * Simpler and easier to understand; easier to interpret the results.
     * Faster when talking about learning from data.
     * Less training data required to learn the mapping function, working well even if the fit to data is not perfect.   
     
     Limitations:

      * Highly constrained to the specified form of the simplified function.
      * Limited complexity of the problems they are suitable for.
      * Poor fit in practice, unlikely to match the underlying mapping function.
  
  * None-Parametric Machine Learning Algorithms:  
      * Non-parametric algorithms do not make assumptions regarding the form of the mapping function between input data and output so they are free to learn any functional form from the training data such as KNN and Decision tree.  
      
      Benefits:

      * High flexibility, in the sense that they are capable of fitting a large number of functional forms.
      * Power by making weak or no assumptions on the underlying function.
      * High performance in the prediction models that are produced.  
      
    Limitations:

      * More training data is required to estimate the mapping function.
      * Slower to train, generally having far more parameters to train.
      * Overfitting the training data is a risk; overfitting makes it harder to explain the resulting predictions.
  
### Day 5 : 18/07/2020 
#### Polarbeargo  

* Reading Lesson 2 Introduction to Machine Learning section 27 Classical ML vs. Deep Learning and section 28 Approaches to Machine Learning.
* Grasp concepts:
![][image3]

### Day 6 : 19/07/2020 
#### Polarbeargo 

* Reading Lesson 2 Introduction to Machine Learning section Bias vs. Variance Trade-off.
* Keypoint concepts:
   * Bias : 
      * Measures how inaccurate the model prediction is in comparison with the true output. 
      * Due to erroneous assumptions made in the machine learning process to simplify the model and make the target function easier to learn. 
      * High model complexity tends to have a low bias.
      
   * Variance: 
      * Measures how much the target function will change if different training data is used. 
      * Variance can be caused by modeling the random noise in the training data. 
      * High model complexity tends to have a high variance. 
      
* As a general trend, parametric and linear algorithms often have high bias and low variance, whereas non-parametric and non-linear algorithms often have low bias and high variance.  
  * Low bias:  
     * Low bias means fewer assumptions about the target function.  
     * Examples of algorithms with low bias are KNN and decision trees.   
     * Having fewer assumptions can help generalize relevant relations between features and target outputs. In contrast, high bias means more assumptions about the target function. Linear regression would be a good example (e.g., it assumes a linear relationship).   
     * Having more assumptions can potentially miss important relations between features and outputs and cause underfitting.  
  * Low variance:    
     * indicates changes in training data would result in similar target functions. For example, linear regression usually has a low variance.   
     * High variance indicates changes in training data would result in very different target functions. For example, support vector machines usually have a high variance.   
     * High variance suggests that the algorithm learns the random noise instead of the output and causes overfitting.      
* Increasing model complexity would decrease bias error since the model has more capacity to learn from the training data. But the variance error would increase if the model complexity increases, as the model may begin to learn from noise in the training data.

* The goal of training machine learning models is to achieve low bias and low variance.

### Day 7 : 20/07/2020 
#### Polarbeargo  

* Writing lesson 3 Model Training Lab: Import, Transform, and Export Data:  

Import Data      |  Fix Dtype to String |  Transform, and Export Data
:-------------------------:|:-------------------------:|:-------------------------:
![][image4]                | ![][image6]               |![][image5]

### Day 8 : 21/07/2020 
#### Polarbeargo  
* Reading lesson 3 Model Training section 6 Managing Data and section 8 More about datasets.  
* Organize keypoint:  
  * Datastores offer a layer of abstraction over the supported Azure storage services. They store all the information needed to connect to a particular storage service. Datastores provide an access mechanism that is independent of the computer resource that is used to drive a machine learning process.  

  * Datasets are resources for exploring, transforming, and managing data in Azure ML. A dataset is essentially a reference that points to the data in storage. It is used to get specific data files in the datastores.  
  
* Steps of the data access workflow are:  

  1. Create a datastore so that you can access storage services in Azure.  
  2. Create a dataset, which you will subsequently use for model training in your machine learning experiment.  
  3. Create a dataset monitor to detect issues in the data, such as data drift.  
* We do versioning most typically when:  

  * New data is available for retraining.  
  * When you are applying different approaches to data preparation or feature engineering.

### Day 9 : 22/07/2020 
#### Polarbeargo  
* Writing and reading lesson 3 Model Training Lab: Create and Version a Dataset, section 11 Introducing Features and section 12 Feature Engineering.    
![][image7]  
* The columns in a table can be referred to as features, selecting the features process is called feature selection and dimensionality reduction is to decrease the number of features.  
* Feature Engineering methods:  

  * Flagging: Deriving a boolean (0/1 or True/False) value for each entity.
  * Aggregation: Getting a count, sum, average, mean, or median from a group of entities.
  * Part-of: Extracting the month from a date variable.
  * Binning: Grouping customers by age and then calculating average purchases within each group.

### Day 10 : 23/07/2020 
#### Polarbeargo  
* Reading lesson 3 Model Training section 14 Feature Selection keypoint concept:  
  * Many machine learning do not perform well when given a large number of variables or features. We can improve the situation of having too many features through dimensionality reduction:   
    * PCA (Principal Component Analysis): A linear dimensionality reduction technique based mostly on exact mathematical calculations.
    * t-SNE (t-Distributed Stochastic Neighboring Entities): Encodes a larger number of features into a smaller number of "super-features."
    * Feature embedding: A dimensionality reduction technique based on a probabilistic approach; useful for the visualization of multidimensional data.  
* Rereading AWS Machine Learning Foundations Course spending more time to sink into the lectures.
* Yesterday I know my Road Repair Vision System and Mosquitoes Vector Control Vision System for Taiwan and Australia government helped my previous company received [2020 Taipei Go Smart Conference Award Winner](https://www.citiesgosmart.org/news_content.htm?news=69), I would love to write a post say thank you to Udacity, Sir David J Malan, Sir Luigi Morelli and Kaggle that your student continue implement Artificial Intelligence system help a lot of people and humanity again. Thank you for allowing me continue enjoy studying here :blush:.   
 
2020 Taipei Go Smart Conference Award Winner|  
:-------------------------:|
![][image8]                | 

### Day 11 : 24/07/2020 
#### Polarbeargo  
* Writing lesson 3 Model Training Lab: Engineer and Select Features.  
![][image9]  
* Rereading AWS Machine Learning Foundations Course spending more time to sink into the lectures.
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)

### Day 12 : 25/07/2020 
#### Polarbeargo  
* Reading lesson 3 Model Training section 17 Data Drift, section 18 Model Training Basics, section 19 Model Training in Azure Machine Learning, section 20 Training Classifiers and section 21 Training Regressors keypoint:  
  * Data drift is change in the input data for a model over time which causes degradation in the model's performance, as the input data drifts farther and farther from the data on which the model was trained. Cause of data drift:  
      * Natural drift in the data: A change in customer behavior over time.  
      * Data quality issues: A sensor breaks and starts providing inaccurate readings.  
      * Covariate shift: Two features that used to be correlated are no longer correlated.  
      * Upstream process changes: A sensor is replaced, causing the units of measurement to change (e.g., from minutes to seconds).  
  * Major goal of model training is to learn the values of the model parameters. Some model parameters are not learned from the data. These are called hyperparameters and their values are set before training. Here are some examples of hyperparameters:  
      * The number of layers in a deep neural network.  
      * The number of clusters (such as in a k-means clustering algorithm).  
      * The learning rate of the model.  
  * Split our data into three parts:  

      * Training data: To learn the values for the parameters.  
      * Validation data: Check the model's performance on the validation data and tune the hyperparameters until the model performs well with the validation data.  
      * Test data: Do a final check of its performance on fresh test data that we did not use during the training process.    
  * Training Classifiers: In a classification problem, the outputs are categorical or discrete.  
  * Training Regressors: In a regression problem, the output is numerical or continuous.  
* Rereading AWS Machine Learning Foundations Course spending more time to sink into the lectures.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)

### Day 13 : 26/07/2020 
#### Polarbeargo  

* Reading lesson 3 Model Training:
  * Section 22 Evaluating Model Performance
  * Section 23 Confusion Matrices
  * Section 24 Evaluation Metrics for Classification:  
    ![][image10]
  * Section 26 Evaluation Metrics for Regression:  
      * R-Squared: How close the regression line is to the true values.  
      * RMSE: Square root of the squared differences between the predicted and actual values.  
      * MAE: Average of the absolute difference between each prediction and the true value.  
      * Spearman correlation: Strength and direction of the relationship between predicted and actual values.    
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
* Rereading AWS Machine Learning Foundations Course spending more time to sink into the lectures.  

### Day 14 : 27/07/2020 
#### Polarbeargo  
* Reading and writing lesson 3 Model Training Lab Train and Evaluate a Model enjoy the awesome Microsoft Azure Machine Learning Scholarship Foundation course Nanodegree Program materials :smile:!    
 ![][image11]  
 
 Confusion Metrics      |  Evaluate Model  
:-------------------------:|:-------------------------:
![][image12]                | ![][image13]  

* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete, discussion and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
* Rereading AWS Machine Learning Foundations Course spending more time to sink into the lectures :blush:.  

### Day 15 : 28/07/2020 
#### Polarbeargo  

* Spending more time reading Ensemble Learning and Automated ML in lesson 3 Model Training section 30 Strength in Numbers.
  * Ensemble Learning: 
    * Bagging or bootstrap aggregation:   
      * Helps reduce overfitting for models that tend to have high variance (such as decision trees).  
      * Uses random subsampling of the training data to produce a bag of trained models.  
      * The resulting trained models are homogeneous.  
      * The final prediction is an average prediction from individual models.    
    * Boosting:
      * Helps reduce bias for models.
      * In contrast to bagging, boosting uses the same input data to train multiple models using different hyperparameters.
      * Boosting trains model in sequence by training weak learners one by one, with each new learner correcting errors from previous learners.  
      * The final predictions are a weighted average from the individual models.    
    * Stacking:  
      * Trains a large number of completely different (heterogeneous) models.  
      * Combines the outputs of the individual models into a meta-model that yields more accurate predictions.
  * Automated ML:  
    * Automates many of the iterative, time-consuming, tasks involved in model development (such as selecting the best features, scaling features optimally, choosing the best algorithms, and tuning hyperparameters).
    * Allows data scientists, analysts, and developers to build models with greater scale, efficiency, and productivity—all while sustaining model quality.  
    
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete, discussion and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
* Rereading AWS Machine Learning Foundations Course spending more time to sink into the lectures. 

### Day 16 : 29/07/2020 
#### Polarbeargo 

* Reading and writing lesson 3 Model Training Lab: Train a Two-Class Boosted Decision Tree and Lab: Train a Simple Classifier with Automated ML:  
![][image14]  
 
 Confusion Metrics      |  Evaluate Model  
:-------------------------:|:-------------------------:
![][image16]                | ![][image15]  

Automated ML      |  Metrics
:-------------------------:|:-------------------------:
![][image17]                | ![][image18]  

Precision-recall & ROC      |  Calibration & lift 
:-------------------------:|:-------------------------:
![][image19]                | ![][image20]  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete, discussion and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
   * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)
* Rereading AWS Machine Learning Foundations Course spending more time to sink into the lectures. 
 
### Day 17 : 30/07/2020 
#### Polarbeargo 

* Reading and writing lesson 4 Supervised learning & Unsupervised learning to  section 18 Automate the Training of Regressors and Lab: Two-Class Classifiers Performance:

Two Class Logistic Regression Precision     |  Two Class Boosted Decision Tree Precision
:-------------------------:|:-------------------------:  
![][image21]                | ![][image22]  

* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete, discussion and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
   * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)
* Rereading AWS Machine Learning Foundations Course spending more time to sink into the lectures. 

### Day 18 : 31/07/2020 
#### Polarbeargo 

* Reading lesson 4 Supervised learning & Unsupervised learning:  
  * Section 7 Multi-Class Algorithms - chart and metrics evaluating results of a classification algorithm:  
    * ROC curve  
    * Confusion Matrix  
    * Recall  
  * Section 19 Automate the Training of Regressors:  
    * Automated Machine Learning gives users the option to automatically scale and normalize input features.   
    * It also gives users the ability to enable additional featurization, such as missing values imputation(impute with most frequent value), encoding, and transforms.  
  * Section 20 Unsupervised Learning:  
    * K-Mean Clustering  
    * PCA  
    * Autoencoder  
  * Section 21 Semi-supervised learning:   
    * Combines the supervised and unsupervised approaches which involves having small amounts of labeled data and large amounts of unlabeled data.  
  * Section 24 Clustering:   
    * Density base clustering: Groups members based on how closely they are packed together; can learn clusters of arbitrary shape.  
    * Hierarchical clustering: Builds a tree of clusters.  
    * Centroid base clustering: Groups members based on their distance from the center of the cluster.  
    * Distributional based clustering: Groups members based on the probability of a member belonging to a particular distribution.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete, discussion and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
   * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)

### Day 19 : 01/08/2020 
#### Polarbeargo 

* Reading lesson 5 Applications of Machine learning to section 4 Characteristics of Deep Learning.  
* Learning the diagram of the relation between Artificial Intelligence, Machine Learning and Deep Learning from [Deep Learning, by Ian Goodfellow, Yoshua Bengio, Aaron Courville](https://www.deeplearningbook.org/contents/intro.html).  
![][image23] 
* Reading Brenda.Udacity and Palak.Udacity their post and writing of #ThankfulThursday, #FocusFriday and AMA session. It's excellent can reading their post and inspired by their warm writing :star:! Thank you Brenda.Udacity and Palak.Udacity:heart:.
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete, discussion and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
   * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)
* Participate, enjoy and immerse myself in the time with Sir David J Malan and Brian Yu's lecture about What is cloud computing on zoom.  
* Catch up the reading, writing quiz and lab of [Architecting with Google Kubernetes Engine: Production
by Google Cloud](https://www.coursera.org/learn/deploying-secure-kubernetes-containers-in-production).

### Day 20 : 02/08/2020 
#### Polarbeargo 
  
* Reading lesson 5 Applications of Machine learning to section 6 Benefits and Applications of Deep Learning.
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete, discussion and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
   * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started) 
* Catch up the reading, writing quiz and lab of [Architecting with Google Kubernetes Engine: Production
by Google Cloud](https://www.coursera.org/learn/deploying-secure-kubernetes-containers-in-production).
* Sunday morning church time, Happy Eid Al Adha Mubarak :smile:!  
![][image24]

### Day 21 : 03/08/2020 
#### Polarbeargo  

* Reading lesson 5 Applications of Machine learning:  
    * Section 9 Specialized Cases of Model Training  
    ![][image25]  
    * Section 11 Similarity Learning  
        * Similarity Learning as classification: The similarity function map pair of entities to a finite number of similarity level(between 0-1).  
        * Similarity Learning as regression: The similarity function map pair of entities to numberical values.  
    * Section 15 Text Classification  

Training a classification Model with Text|
:-------------------------:|
![][image26]               |     

Predicting a classification from text|
:-------------------------:|
![][image27]               |
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete, discussion and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
   * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started) 
* Catch up the reading, writing quiz and lab of [Architecting with Google Kubernetes Engine: Production
by Google Cloud](https://www.coursera.org/learn/deploying-secure-kubernetes-containers-in-production).

### Day 22 : 04/08/2020 
#### Polarbeargo 

* Reading lesson 5 Applications of Machine learning:  
    * Section 18 Feature Learning    
        * Supervised feature learning: New feature are learned by labled data.  
        Examples:  
          * Image classification  
          * Data set that has multiple categorical features with high cardinality  
        * Unsupervised feature learning: Based on the learning the new features without having labled input data. Clustering = a form of feature learning.   
        Other algorithms:  
           * PCA   
           * Independent component analysis  
           * Autoencoder(deep learning)  
           * Matrix factorization  
    
    * Section 20 Anomaly Detection  
        * Can be done in both supervised and unsupervised ways.   
        * The anomaly and normal data are highly imbalanced.  
        * Machinery maintenance 
        ![][image28]  
        
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete, discussion and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
   * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started) 
* Catch up the reading, writing quiz and lab of [Architecting with Google Kubernetes Engine: Production
by Google Cloud](https://www.coursera.org/learn/deploying-secure-kubernetes-containers-in-production).
* After participated in Sir David J Malan and Brian Yu's lecture about what is cloud computing on zoom, writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Cloud Computing assignment. 

### Day 23 : 05/08/2020 
#### Polarbeargo 

* Writing lesson 4 Lab: Multi-Class Classifiers Performance and Lab: Train a Classifier Using Automated Machine Learning :star::   

Multi-Class Classifiers Performance     |  Train a Classifier Using Automated Machine Learning   
:-------------------------:|:-------------------------:  
![][image29]                | ![][image30]  

* Reading lesson 5 Applications of Machine learning:  
    * Section 19 Applications of Feature Learning:  
    Application of Feature learning:  
        * Image Classification
        * Image Search
        * Feature embedding  
    Image Classification with Convolutional Neural Networks (CNNs):  
    ![][image31]  
    Image Search with Autoencoders:  
    ![][image32]  
    
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
* Compete, discussion and learn in Kaggle:
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
   * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
* Writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) web development and Technology Stacks assignments. 

### Day 24 : 06/08/2020 
#### Polarbeargo 

* Reading Lesson 6 Managed Services for Machine Learning to section 17 Operationalizing Models.  
  Keypoint grasp:   
  
Advanced Modeling      |  Machine Learning Pipelines
:-------------------------:|:-------------------------:  
![][image33]                | ![][image34]  

* Enjoying the time playing Student Story Challenge :smile: !   

![][image35]                | ![][image36]  
:-------------------------:|:-------------------------:  
![][image37]                | ![][image38]   

* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
   * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
* Writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) web development assignment.   
* Writing quiz and lab of [Architecting with Google Kubernetes Engine: Production
by Google Cloud](https://www.coursera.org/learn/deploying-secure-kubernetes-containers-in-production).

### Day 25 : 07/08/2020 
#### Polarbeargo   

* Watching record LIVE AMA with Microsoft Azure Experts Thank you Brenda.Udacity and Palak.Udacity. Your writings and posts are lovely and brightful :star:! Many gratitude and great memories with Sir Sebastian Thrun and Udacity.
* It's a half way of the Scholarship spending more time writing lesson 4 Supervised learning & Unsupervised learning Labs:   
    * Regressors Performance   
    ![][image39]  
    * Train a Regressor using Automated Machine Learning  
    
Model     |  Compare best model performance   
:-------------------------:|:-------------------------:  
![][image40]                | ![][image41] 

Metrics     |  Predic_true  
:-------------------------:|:-------------------------:  
![][image42]                | ![][image43]  
    
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
   * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
* Finish writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) web development assignment. During implement web development assignment my memory float back to 2014 when I'm writing CS50 intro final project I miss those memories with lovely, touched and excitement in 2014. Writing assignments helped me not only reskill or advanced my knowledge but also recover stronger through these process. Thank you Sir David J Malan :heart: and Brian Yu still be there with a lot of students.  
* Writing quiz and lab of [Architecting with Google Kubernetes Engine: Production
by Google Cloud](https://www.coursera.org/learn/deploying-secure-kubernetes-containers-in-production).

### Day 26 : 08/08/2020 
#### Polarbeargo   

* Writing lesson 4 Supervised learning & Unsupervised learning Lab: Train a Simple Clustering Algorithm  
    
K Mean Clustering     |  Result  
:-------------------------:|:-------------------------:  
![][image44]                | ![][image45]  

* Reading Lesson 6 Managed Services for Machine Learning to section 2 Programmatically Accessing Managed Services via the Azure Machine Learning SDK for Python.
* Think of what part I would love to draw for the Visual Challenge Exhibition.  

* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started) 
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)
* Writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Internet Technologies assignment.
* Writing quiz and lab of [Architecting with Google Kubernetes Engine: Production
by Google Cloud](https://www.coursera.org/learn/deploying-secure-kubernetes-containers-in-production).

### Day 27 : 09/08/2020 
#### Polarbeargo   

* Writing lesson 5 Applications of Machine learning Lab: Train a Simple Neural Net.  

Score Model      |  Evaluate Model
:-------------------------:|:-------------------------:
![][image47]                | ![][image46]
* Saturday I pick and immersed myself in a horrible movie yesterday.  

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/pX-_0FFzgTI/0.jpg)](https://youtu.be/pX-_0FFzgTI)   
* Drawing the Visual Challenge :smile:!
* Reading Kaggle:    
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started) 
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)
* Finish writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Internet Technologies assignment. Listen to Programming Languages Lecture.
* Writing quiz and lab of [Architecting with Google Kubernetes Engine: Production
by Google Cloud](https://www.coursera.org/learn/deploying-secure-kubernetes-containers-in-production).

### Day 28 : 10/08/2020 
#### Polarbeargo   

* Writing lesson 5 Applications of Machine learning Lab: Train a Simple Recommender.  

Score Model      |  Evaluate Model
:-------------------------:|:-------------------------:
![][image48]                | ![][image49]  
![][image50]
* Drawing the Visual Challenge :star:!
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started) 
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)
* Watching [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Program Language Lecture.
* Finish writing quiz and lab of [Architecting with Google Kubernetes Engine: Production
by Google Cloud](https://www.coursera.org/learn/deploying-secure-kubernetes-containers-in-production). In the Securing Google Kubernetes Engine with Cloud IAM and Pod Security Policies Lab I stuck for a while and talk to Coursera course helper Annu, I'm very appreciate the time debug this lab with Annu who send me a lovely message see you on the cloud in the time we said goodbye to each other. This lecture with Sir David J Malan talks about what is the cloud computing on zoom stronger the load balancer fundamental knowledge in my mind. After finish the quiz Access Control and Security in Kubernetes and GKE, a small window pop out "Congratulation you finish the most difficult quiz in this course!!" I feel kind warm inside my heart:blush:.

Architecting with Google Kubernetes Engine      |  Architecting with Google Kubernetes Engine: Production
:-------------------------:|:-------------------------:
![][image51]                | ![][image52]  

### Day 29 : 11/08/2020 
#### Polarbeargo   

* Writing lesson 5 Applications of Machine learning Lab: Train a Simple Text Classifier stuck in Train model module 

Text Classifier      |  Over time
:-------------------------:|:-------------------------:
![][image54]                | ![][image53] 
* Submit the Visual Challenge :star:!  
![][image55]  
* Plan spend more time finish watch all lectures first.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started) 
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)
* Writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Program Language assignment.

### Day 30 : 12/08/2020 
#### Polarbeargo  

* Finished watching all lectures.  
* Focus on implement the rest Labs. 
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started) 
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
* Finished writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Program Language assignment and  watching Lecture Computational Thinking.
* Reviewing [Architecting with Google Kubernetes Engine: Production
by Google Cloud](https://www.coursera.org/learn/deploying-secure-kubernetes-containers-in-production).

### Day 31 : 13/08/2020 
#### Polarbeargo 

* Finished writing lesson 5 Applications of Machine learning and Lesson 6 Managed Services for Machine Learning Labs:
  * Lab:Forecasting
  * Lab: Managing Compute
  * Lab: Managed Notebook Environments 
  * Lab: Deploy a Model as a Webservice
  * Lab: Training and Deploying a Model from a Notebook Running in a Compute Instance 
  
Forecasting     |  Managing Compute   
:-------------------------:|:-------------------------:  
![][image56]                | ![][image57] 

Managing Compute     |  Managed Notebook Environments  
:-------------------------:|:-------------------------:  
![][image58]                | ![][image59]  
    
Deploy a Model as a Webservice     |  Training and Deploying a Model from a Notebook Running in a Compute Instance  
:-------------------------:|:-------------------------:  
![][image60]                | ![][image61]  

* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)      

### Day 32 : 14/08/2020 
#### Polarbeargo 

* Finished writing Lesson 6 Managed Services for Machine Learning Labs: Explore Experiments and Runs

![][image62]                | ![][image63]  
:-------------------------:|:-------------------------:  
![][image64]                | ![][image65]   

* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)    
* Writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Computational Thinking assignment and fixing Technology Stacks assignments. 
* Recently start reviewing C++ Nanodegree Program and Robotics Software Engineer Nanodegree Program.

### Day 33 : 15/08/2020 
#### Polarbeargo 

* Finished writing Lesson 7 Responsible AI Lab: Model Explainability:  

![][image66]  | ![][image67]  

* Start reviewing Microsoft Scholarship Foundation course Nanodegree Program and my 50days of Udacity course note :star:!
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)    
* Writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Computational Thinking assignment, re watching lecture Technology Stacks carefully, fixed and resubmitted Technology Stacks assignment. 
* Reviewing C++ Nanodegree Program and Robotics Software Engineer Nanodegree Program.

### Day 34 : 16/08/2020 
#### Polarbeargo 

* Finished reviewing Lesson 7 Responsible AI.
* Reviewing my 50days of Udacity course note :star:!
* Reviewing C++ Nanodegree Program and Robotics Software Engineer Nanodegree Program.
* Writing [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Computational Thinking assignment, spending more time recall lecture Technology Stacks carefully in my mind and fixing Programming Languages assignment. 
* Review CS50 Quiz show quiz replay the zoom session in my mind :scream:!
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson) 

### Day 35 : 17/08/2020 
#### Polarbeargo 

* Reviewing Lesson 6 Managed Services for Machine Learning.
* Reviewing my 50days of Udacity course note :star:!  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
* Reviewing C++ Nanodegree Program and Robotics Software Engineer Nanodegree Program.  
* Fixed and resubmitted [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Programming Languages assignment and writing Computational Thinking assignment.  

### Day 36 : 18/08/2020 
#### Polarbeargo 

* Reviewing Lesson 6 Managed Services for Machine Learning.
* Reviewing my 50days of Udacity course note :star:! 
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
* Reviewing C++ Nanodegree Program and Robotics Software Engineer Nanodegree Program.  
* Finished writing and submitted [HarvardX: CS50B CS50's Computer Science for Business Professionals](https://online-learning.harvard.edu/course/cs50s-computer-science-business-professionals) Computational Thinking assignment. Spending some time sink into the memories of lectures, zoom and assignments within these time.

### Day 37 : 19/08/2020 
#### Polarbeargo   
* Reviewing Lesson 5 Applications of Machine learning.  
* Treasure hunt in the free course found Madam Katie and Sir Sebastian Thrun's lecture [Intro to Machine Learning](https://classroom.udacity.com/courses/ud120) and [Artificial Intelligence for Robotics](https://classroom.udacity.com/courses/cs373) listen to their gentle voice talk about the great story of Enron email dataset.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
* Reviewing C++ Nanodegree Program and Robotics Software Engineer Nanodegree Program.  
* Reading [CS50’s Introduction to Game Development Final Project](https://cs50.harvard.edu/games/2018/assignments/final/)  
* Reading [Probabilistic Graphical Models Specialization](https://www.coursera.org/specializations/probabilistic-graphical-models) and [Sensor Fusion Engineer Nanodgree Program](https://www.udacity.com/course/sensor-fusion-engineer-nanodegree--nd313) think about in my mind when will be the perfect time for enroll?

### Day 38 : 20/08/2020 
#### Polarbeargo  

* Reviewing Lesson 5 Applications of Machine learning.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
* Reviewing C++ Nanodegree Program and Robotics Software Engineer Nanodegree Program.  

### Day 39 : 21/08/2020 
#### Polarbeargo  

* Reviewing Lesson 5 Applications of Machine learning.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
* Reviewing C++ Nanodegree Program and Robotics Software Engineer Nanodegree Program.  
* Watching the free course Madam Katie and Sir Sebastian Thrun's lecture [Intro to Machine Learning](https://classroom.udacity.com/courses/ud120) and [Artificial Intelligence for Robotics](https://classroom.udacity.com/courses/cs373) because their lovely, cute and intelligence interaction inside lectures.
* Writing [CS50’s Introduction to Game Development Assignment 0: “Pong, The AI Update”](https://cs50.harvard.edu/games/2018/assignments/0/) 

### Day 40 : 22/08/2020 
#### Polarbeargo  

* Reviewing Lesson 5 Applications of Machine learning.
* Next AMA will host on Zoom looking forward to it:star:!
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
* Reviewing C++ Nanodegree Program and Robotics Software Engineer Nanodegree Program. 
* Writing [CS50’s Introduction to Game Development Assignment 0: “Pong, The AI Update”](https://cs50.harvard.edu/games/2018/assignments/0/) 

### Day 41 : 23/08/2020 
#### Polarbeargo  

* Reviewing Lesson 5 Applications of Machine learning.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
* Finished Writing [CS50’s Introduction to Game Development Assignment 0: “Pong, The AI Update”](https://cs50.harvard.edu/games/2018/assignments/0/) and submitted it.  
![][image68]  
[Link](https://youtu.be/7igLgibis0Q)

### Day 42 : 24/08/2020 
#### Polarbeargo  

* Finish review Lesson 5 Applications of Machine learning and start reviewing Lesson 4 Supervised learning & Unsupervised learning.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
* Writing [CS50’s Introduction to Game Development Assignment 2: “Breakout, The Powerup Update”](https://cs50.harvard.edu/games/2018/assignments/2/).

### Day 43 : 25/08/2020 
#### Polarbeargo  

* Reviewing Lesson 4 Supervised learning & Unsupervised learning.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
  * [Lyft Motion Prediction for Autonomous Vehicles
Build motion prediction models for self-driving vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles)  
* Finished writing and submitted [CS50’s Introduction to Game Development Assignment 1: “Flappy Bird, The Reward Update”](https://cs50.harvard.edu/games/2018/assignments/1/).  
![][image69]  
[Link](https://youtu.be/37-S2wT1cPU)

### Day 44 : 26/08/2020 
#### Polarbeargo  

* Reviewing Lesson 4 Supervised learning & Unsupervised learning and writing Lesson 2 Scaling data quiz.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
  * [Lyft Motion Prediction for Autonomous Vehicles
Build motion prediction models for self-driving vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles)  
* Writing [CS50’s Introduction to Game Development Assignment 2: “Breakout, The Powerup Update”](https://cs50.harvard.edu/games/2018/assignments/2/).  

### Day 45 : 27/08/2020 
#### Polarbeargo  

* Reviewing Lesson 4 Supervised learning & Unsupervised learning and finshed writing Lesson 2 Scaling data quiz.  
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
  * [Lyft Motion Prediction for Autonomous Vehicles
Build motion prediction models for self-driving vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles)  
* Writing [CS50’s Introduction to Game Development Assignment 2: “Breakout, The Powerup Update”](https://cs50.harvard.edu/games/2018/assignments/2/).  

### Day 46 : 28/08/2020 
#### Polarbeargo  

* Reviewing Lesson 3 Model training.  
* Think of Project Showcase.
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
  * [Lyft Motion Prediction for Autonomous Vehicles
Build motion prediction models for self-driving vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles)  
* Writing [CS50’s Introduction to Game Development Assignment 2: “Breakout, The Powerup Update”](https://cs50.harvard.edu/games/2018/assignments/2/).  

### Day 47 : 29/08/2020 
#### Polarbeargo  
 
* Searching, reading paper references and design implement flow for Project Showcase.
* Reading Kaggle:  
  * [Prostate cANcer graDe Assessment (PANDA) Challenge
Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion)  
  * [ALASKA2 Image Steganalysis
Detect secret data hidden within digital images](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion)  
  * [Jigsaw Multilingual Toxic Comment Classification
Use TPUs to identify toxicity comments across multiple languages](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion)  
  * [Global Wheat Detection
Can you help identify wheat heads using image analysis?](https://www.kaggle.com/c/global-wheat-detection/overview)  
  * [SIIM-ISIC Melanoma Classification
Identify melanoma in lesion images](https://www.kaggle.com/c/siim-isic-melanoma-classification)  

* Compete, discussion and learn in Kaggle:
  * [Cornell Birdcall Identification
Build tools for bird population monitoring](https://www.kaggle.com/c/birdsong-recognition)  
  * [Petals to the Metal: Flower Classification on TPU
Getting Started with TPUs on Kaggle!](https://www.kaggle.com/c/tpu-getting-started)  
  * [Contradictory, My Dear Watson
Detecting contradiction and entailment in multilingual text using TPUs](https://www.kaggle.com/c/contradictory-my-dear-watson)  
  * [Lyft Motion Prediction for Autonomous Vehicles
Build motion prediction models for self-driving vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles)  
* Debugging [CS50’s Introduction to Game Development Assignment 2: “Breakout, The Powerup Update”](https://cs50.harvard.edu/games/2018/assignments/2/). 

### Day 48 : 30/08/2020 
#### Polarbeargo

* Reading paper references, think and design implement flow for Project Showcase.

### Day 49 : 31/08/2020 
#### Polarbeargo

* Reading project_idea channel, exploring datasets start implementing Showcase Project.

### Day 50 : 01/09/2020 
#### Polarbeargo

* Thinking and writing Showcase Project.
* Today I recieved the lovely Machine Learning Scholarship Program for Microsoft Azure badge :star:!
![][image70]
