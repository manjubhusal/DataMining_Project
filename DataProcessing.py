import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Function to dynamically segment timeframes
def time_segregation(time, intervals):
    time_col = pd.to_datetime(time)
    for i, interval in enumerate(intervals):
        if interval[0] <= time_col.hour < interval[1]:
            return f"Timerange{i+1}"
    return f"Timerange{len(intervals)+1}"

# Function to determine the sum, frequency and variance of calories based on timeframes
def consumption_statistics(row, intervals, time_columns, calorie_columns):
    statistics = {f"Timerange{i+1}_count": 0 for i in range(len(intervals)+1)}
    statistics.update({f"Timerange{i + 1}_sum": 0 for i in range(len(intervals) + 1)})
    statistics.update({f"Timerange{i + 1}_variance":[] for i in range(len(intervals) + 1)})

    for time, calories in zip(row[time_columns], row[calorie_columns]):
        timeframe = time_segregation(time, intervals)
        statistics[f"{timeframe}_count"] += 1
        statistics[f"{timeframe}_sum"] += calories
        statistics[f"{timeframe}_variance"].append(calories)
    for key in list(statistics.keys()):
        if "_var" in key:
            statistics[key] = np.var(statistics[key]) if statistics[key] else 0
    return pd.Series(statistics)


# Function to process time and calories pairs for weekday dataset
def time_calories_processing_weekday(weekday_dataset, intervals):
    column_names = weekday_dataset.columns
    time_columns = []
    calorie_columns = []
    for column in column_names:
        if "WKDY_020" in column:
            time_columns.append(column)
        elif "WKDYikcal" in column:
            calorie_columns.append(column)
            weekday_dataset[column] = weekday_dataset[column].fillna(0)

    case_calories = weekday_dataset.apply(consumption_statistics, args = (intervals, time_columns, calorie_columns,), axis=1)
    return weekday_dataset, case_calories, time_columns

# Function to process time and calories pairs for weekend dataset
def time_calories_processing_weekend(weekend_dataset, intervals):
    column_names = weekend_dataset.columns
    time_columns = []
    calorie_columns = []
    for column in column_names:
        if "WKEND_020" in column:
            time_columns.append(column)
        elif "WKENDikcal" in column:
            calorie_columns.append(column)
            weekend_dataset[column] = weekend_dataset[column].fillna(0)

    case_calories = weekend_dataset.apply(consumption_statistics, args = (intervals, time_columns, calorie_columns, ), axis=1)
    return weekend_dataset, case_calories, time_columns

# Function to process the time and calorie pairs for weekday and weekend and merge them into a single dataframe
def time_calories_processing_merged(weekday_dataset,weekend_dataset, intervals):
    weekday_dataset, case_calories,time_columns = time_calories_processing_weekday(weekday_dataset, intervals)
    weekday_columns = pd.concat([weekday_dataset["SEQN"], case_calories],axis=1)
    weekend_dataset, case_calories, time_columns = time_calories_processing_weekend(weekend_dataset, intervals)
    weekend_columns = pd.concat([weekend_dataset["SEQN"], case_calories], axis=1)
    merged_columns = pd.merge(weekday_columns, weekend_columns, on="SEQN", suffixes=("_weekday", "_weekend"))
    return weekday_dataset, merged_columns, time_columns

# Function to process the data for creating the boxplot
def box_plot_processing(row, intervals, time_columns):
    time_values = []
    timeframe_values = []
    for time in row[time_columns]:
        timeframe = time_segregation(time, intervals)
        if not pd.isna(time):
            if time not in time_values:
                timeframe_values.append(timeframe)
                time_col = pd.to_datetime(time)
                time_hour = time_col.hour
                if 0 <= time_hour < 4:
                    time_hour = time_hour + 24
                time_values.append(time_hour)
    return pd.Series({"Timeframe": timeframe_values, "Time": time_values})

# Function to create boxplot of time and timeframe
def plot_boxplot(dataset, intervals, time_columns):
    box_plot_data = dataset.apply(box_plot_processing, args = (intervals, time_columns,), axis=1)
    box_plot_data_exploded = box_plot_data.explode(['Timeframe','Time'],ignore_index=True)
    box_plot_data_exploded.dropna(subset=['Time'], inplace=True)
    time_boxplot = sns.boxplot(
         x='Timeframe'
        ,y='Time'
        ,data=box_plot_data_exploded
        ,order=[f"Timerange{n+1}" for n in range(len(intervals)+1)]
    )
    plt.xlabel("Timeranges")
    plt.ylabel("Time in Hours")
    plt.title("Box Plot of Timerange and Hours")
    plt.show()


# Function to preprocess data
def preprocess_data(dataset, calories, classification_target_col = "BMI_outcome", regression_target_col = "BMXBMI"):
    # Use the line below for analysis with calories plus body statistics
    grouped_case = pd.concat([dataset[["RIAGENDR"
                                    , "RIDAGEYR"
                                    , "BMXHT"
                                    , "BMXWAIST"
                                    , "WTINT2YR"
                                ]], calories, dataset[[classification_target_col, regression_target_col]]], axis=1)
    # Use the line below for analysis with calories only
    # grouped_case = pd.concat([calories, dataset[[classification_target_col, regression_target_col]]], axis=1)

    print("\nTotal number of duplicated rows: ", grouped_case.duplicated().sum())
    print(grouped_case.info())

    # Check if data is balanced for BMI_outcome
    plt.figure(figsize=(10, 6))
    sns.countplot(data=grouped_case, x=classification_target_col)
    plt.title("Data Distribution of BMI_outcome Label")
    plt.xlabel("BMI_outcome")
    plt.ylabel("Number of samples")
    plt.show()

    x = grouped_case.drop(columns=[classification_target_col, regression_target_col])
    y_classification = grouped_case[classification_target_col]
    y_regression = grouped_case[regression_target_col]
    return x, y_classification, y_regression

