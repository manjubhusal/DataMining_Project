import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from datetime import datetime

# Function to implement naive bayes classification on the dataset
def naive_bayes_classification(x_train_classification, x_test_classification, y_train_classification, y_test_classification):
      print("\n---Naive Bayes Classification------")
      start_time = datetime.now()
      bayes_model = GaussianNB()
      bayes_model.fit(x_train_classification, y_train_classification)
      nb_predicted_y = bayes_model.predict(x_test_classification)
      execution_time(start_time, "Naive Bayes")
      classification_evaluation(y_test_classification, nb_predicted_y,"Naive Bayes")

# Function to implement random forest classification on the dataset
def random_forest_classification(x_train_classification, x_test_classification, y_train_classification, y_test_classification):
      print("\n-------Random Forest Classification--------")
      start_time = datetime.now()
      random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
      random_forest.fit(x_train_classification, y_train_classification)
      rf_predicted_y = random_forest.predict(x_test_classification)
      execution_time(start_time, "Random Forest")
      classification_evaluation(y_test_classification, rf_predicted_y, "Random Forest")

# Function to evaluate the performance of classification algorithms
def classification_evaluation(y_test_classification, predicted_y, classifier):
      model_confusion_matrix = confusion_matrix(y_test_classification, predicted_y)
      confusion_matrix_display = ConfusionMatrixDisplay(model_confusion_matrix)
      confusion_matrix_display.plot()
      plt.title(f"Confusion Matrix of {classifier} classifier")
      plt.show()

      print(f"Accuracy obtained by {classifier} classifier is: ", accuracy_score(y_test_classification, predicted_y))
      print(f"Precision obtained by {classifier} classifier is: ", precision_score(y_test_classification, predicted_y, average='micro'))
      print(f"Recall obtained by {classifier} classifier is: ", recall_score(y_test_classification, predicted_y, average='micro'))

# Function to keep track of the execution time
def execution_time(start_time, classifier):
      end_time = datetime.now()
      total_time = (end_time - start_time).total_seconds()
      print(f"Total execution time for {classifier} classifier is: ", total_time)