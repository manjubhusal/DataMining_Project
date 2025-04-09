import pandas as pd
from DataProcessing import time_calories_processing_weekday, time_calories_processing_weekend, time_calories_processing_merged, preprocess_data, plot_boxplot
from sklearn.model_selection import train_test_split
from Classification import naive_bayes_classification, random_forest_classification
from Regression import gradient_boosting_regression, polynomial_regression
from ThreeMealsProcessing import meals_processing

def main():
    #  This is the prompt for the user to choose the time interval and the dataset on which they want to perform the analysis
    print("Choose the case number you want to visit:")

    try:
        case_number = int(input("Enter one case number between 1 to 4."
                                "\n1 has 3 intervals"
                                "\n2 has 4 intervals"
                                "\n3 has 5 intervals"
                                "\n4 is related to maximum calorie consumption : "))
        data_analysis = int(input("Enter which dataset to analyse (1 for weekday, 2 for weekend, 3 for merged): "))
    except ValueError:
        print("Wrong number. Please enter the correct number")
        return

    if case_number == 1:
        intervals = [(4,12), (12,18)]
    elif case_number == 2:
        intervals = [(4,10),(10,14), (14,18)]
    elif case_number == 3:
        intervals = [(4,9), (9,12), (12,16),(16,20)]
    elif case_number == 4:
        print("This case is handled in analysis part directly.")
    else:
        print("Invalid case number. Please enter correct number")
        return
    if case_number != 4:
        print(f"Running case {case_number} with intervals:{intervals}")

    # Load the datatsets
    # Case for weekday dataset only
    if data_analysis == 1:
        weekday_dataset = pd.read_csv("dataset/RAC_TDP_Goal_1_weekday.csv")
        if case_number == 4:
            dataset = weekday_dataset
            calories = meals_processing(weekday_dataset, "weekday")
        else:
            dataset, calories, time_columns = time_calories_processing_weekday(weekday_dataset, intervals)
    # Case for weekend dataset only
    elif data_analysis == 2:
        weekend_dataset = pd.read_csv("dataset/RAC_TDP_Goal_1_weekend_day.csv")
        if case_number == 4:
            dataset = weekend_dataset
            calories = meals_processing(weekend_dataset, "weekend")
        else:
            dataset, calories, time_columns = time_calories_processing_weekend(weekend_dataset, intervals)
    # Case for merged dataset
    elif data_analysis == 3:
        weekday_dataset = pd.read_csv("dataset/RAC_TDP_Goal_1_weekday.csv")
        weekend_dataset = pd.read_csv("dataset/RAC_TDP_Goal_1_weekend_day.csv")
        if case_number == 4:
            weekday_calories = meals_processing(weekday_dataset, "weekday")
            weekend_calories = meals_processing(weekend_dataset, "weekend")
            calories = pd.merge(weekday_calories, weekend_calories, on="SEQN", suffixes=("_weekday", "_weekend"))
            dataset = weekday_dataset
        else:
            dataset, calories, time_columns = time_calories_processing_merged(weekday_dataset, weekend_dataset, intervals)

    else:
        print("Invalid number. Please enter correct number")
        return

    if case_number != 4:
        if data_analysis != 3:
            # Execute the boxplot function
            plot_boxplot(dataset, intervals, time_columns)
    # Data Preprocessing
    x, y_classification, y_regression = preprocess_data(dataset, calories, classification_target_col = "BMI_outcome", regression_target_col = "BMXBMI")
    # Split dataset into 80% and 20% for training and testing
    x_train_regression, x_test_regression, y_train_regression, y_test_regression = train_test_split(x, y_regression, test_size=0.2, random_state=42)
    x_train_classification, x_test_classification, y_train_classification, y_test_classification = train_test_split(x, y_classification, test_size=0.2, random_state=42)
    return x_train_regression, x_test_regression, y_train_regression, y_test_regression, x_train_classification, x_test_classification, y_train_classification, y_test_classification


# Running the main file
if __name__=="__main__":
    x_train_regression, x_test_regression, y_train_regression, y_test_regression, x_train_classification, x_test_classification, y_train_classification, y_test_classification = main()
    # Classification algorithms execution
    print("Classification started")
    naive_bayes_classification(x_train_classification, x_test_classification, y_train_classification, y_test_classification)
    random_forest_classification(x_train_classification, x_test_classification, y_train_classification, y_test_classification)
    # Regression algorithms execution
    print("Regression started")
    gradient_boosting_regression(x_train_regression, x_test_regression, y_train_regression, y_test_regression)
    polynomial_regression(x_train_regression, x_test_regression, y_train_regression, y_test_regression)