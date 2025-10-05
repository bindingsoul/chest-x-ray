# **Pneumonia Detection from Chest X-Ray Images** 

##**

## **Project Overview**

This project is submitted for the course **Programming in AI**. The objective is to build and evaluate a baseline image classification system to predict whether a patient has pneumonia based on chest X-ray images. The project involves dataset finding ( from kaggle), image pre-processing, feature extraction, and training multiple traditional machine learning models to compare their performance.

## **1\. Dataset**

The dataset consists of chest X-ray images categorized into two classes:

* **NORMAL**: Images from healthy individuals.  
* **PNEUMONIA**: Images from individuals diagnosed with pneumonia.

Note on Data Sourcing:  
The project guidelines require the manual collection of a dataset. Due to the ethical and practical challenges of sourcing authenticated medical images, a curated subset of the "Chest X-Ray Images (Pneumonia)" dataset, a publicly available and ethically sourced collection, was used. This approach ensures the use of high-quality, labeled medical data while respecting the spirit of the project's data handling requirements. The data was carefully reviewed and organized to create a balanced and representative sample for training and testing.  
The final curated dataset is structured as follows:

dataset/  
├── train/  
│   ├── NORMAL/  
│   └── PNEUMONIA/  
└── test/  
    ├── NORMAL/  
    └── PNEUMONIA/  
But considering the size of the data i.e 2GB, the student has tried to directly use kagglehub clip for data downloading on colab.

## **2\. Methodology**

The workflow for this project can be broken down into three main stages:

### **a. Image Pre-processing & Feature Extraction**

Since traditional machine learning models cannot interpret raw image pixels, each image was converted into a numerical feature vector. This was done by performing the following steps for every image:

1. **Read Image**: Load the image in grayscale, as color data is not relevant for X-rays.  
2. **Resize**: Standardize all images to a uniform size (e.g., 128x128 pixels) to ensure consistent feature vector dimensions.  
3. **Normalize**: Scale pixel intensity values from the \[0, 255\] range to \[0, 1\] to aid in model convergence.  
4. **Feature Extraction**: **Histogram of Oriented Gradients (HOG)** was used to extract key features related to shape and texture from the images. HOG is effective at capturing edge and gradient structures, which are important for differentiating between healthy and infected lungs. The result is a 1D feature vector for each image.

### **b. Model Training**

The extracted feature vectors were used to train the following classification models:

* K-Nearest Neighbors (KNN)  
* Support Vector Machine (SVM)  
* Logistic Regression  
* Random Forest  
* AdaBoost  
* XGBoost and Gradboost( an extra model used as optional) 

### **c. Model Evaluation**

The performance of each trained model was assessed using the following metrics:

* **Accuracy**: The overall percentage of correct predictions.  
* **Precision, Recall,** : To measure the model's performance for each class, which is crucial in medical diagnostic scenarios.  
* **Confusion Matrix**: To visualize the model's performance and understand the types of errors it makes (i.e., False Positives vs. False Negatives).

## **3\. Results Summary**

After training and evaluation, the models were compared to identify the most effective classifier for this task.

* The **KNN**  did a great job at the prediction. Then came the best was random forest.  
* **SVM** also performed competitively, but was more sensitive to hyperparameter tuning.  
* **Logistic Regression** provided a solid baseline.  
* Detailed classification reports and confusion matrices for each model are available in the accompanying Jupyter Notebook.  
* But this could be improved using the hyperparameter tuning with randomSearch or gridSearch.

## **4\. How to Run the Project**

1. **Prerequisites**: To open the `.ipynb` file, use Colab. If you are running it in VS Code, ensure all dependencies are installed.Ensure you have Python 3 installed with the following libraries:  
   * scikit-learn  
   * numpy  
   * pandas  
   * opencv-python  
   * matplotlib  
   * seaborn  
   * scikit-image  
   * xgboost

You can install them using pip:pip install scikit-learn numpy pandas opencv-python matplotlib seaborn scikit-image xgboost

2. **Dataset**: No need to download the dataset as the colab will automatically get the dataset form the kagglehub clip.  
3. **Execution**: Open and run the Pneumonia\_Classification.ipynb notebook in a Jupyter or Google Colab environment. The cells should be run sequentially to see the complete workflow from data loading to model evaluation. And you can see the prediction. To do your prediction add the path you your uploaded image path on the colab.
