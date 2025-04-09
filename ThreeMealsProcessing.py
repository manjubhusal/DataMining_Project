import pandas as pd

# Function to sort the calories and determine the maximum three calorie intakes
def three_maximum_calories(row, formatted_time_columns):
    time_calorie = []
    for i in range(1,len(formatted_time_columns)):
        if pd.notna(row[f"calories_sum_{i}"]):
            time_calorie.append((row[f"timestamp_{i}"],row[f"calories_sum_{i}"],row[f"calories_count_{i}"],row[f"calories_var_{i}"]))
    sorted_time_calorie = sorted(time_calorie, key = lambda x:x[1],reverse=True)
    return sorted_time_calorie[:3]

# Function to aggregate the time and calorie pairs based on the maximum three calorie intakes
def meals_processing(dataset, data_case):
    selected_columns = [0] + list(range(21,143)) + [205]
    dataset_subset = dataset.iloc[:,selected_columns]
    consumption_timestamp = []
    consumed_calories = []
    SEQN_list = []

    for i in range(1,62):
        SEQN_list.append(dataset_subset['SEQN'])
        if data_case == "weekday":
            consumption_timestamp.append(dataset_subset[f'WKDY_020{i}'])
            consumed_calories.append(dataset_subset[f'WKDYikcal{i}'])
        elif data_case == "weekend":
            consumption_timestamp.append(dataset_subset[f'WKEND_020{i}'])
            consumed_calories.append(dataset_subset[f'WKENDikcal{i}'])
    agg_dataset = pd.DataFrame({'SEQN':pd.concat(SEQN_list),'timestamp':pd.concat(consumption_timestamp),'calories':pd.concat(consumed_calories)})
    timestamp_agg_dataset = agg_dataset.groupby(['SEQN','timestamp'],as_index=False)['calories'].agg(['sum','count','var'])
    timestamp_agg_dataset['var'] = timestamp_agg_dataset['var'].fillna(0)

    reformatted_dataset = []

    unique_seqn = timestamp_agg_dataset['SEQN'].unique()
    for seqn in unique_seqn:
        temp_value = {'SEQN' : seqn}
        seqn_filtered_data = timestamp_agg_dataset[timestamp_agg_dataset['SEQN'] == seqn]
        seqn_filtered_data['timestamp'] = pd.to_datetime(seqn_filtered_data['timestamp'])
        seqn_filtered_data = seqn_filtered_data.sort_values(by = 'timestamp')
        for index, row in enumerate(seqn_filtered_data.itertuples(),1):
            temp_value[f'timestamp_{index}'] = row.timestamp
            temp_value[f'calories_sum_{index}'] = row.sum
            temp_value[f'calories_count_{index}'] = row.count
            temp_value[f'calories_var_{index}'] = row.var
        reformatted_dataset.append(temp_value)
    transpose_reformatted_dataset = pd.DataFrame(reformatted_dataset)

    formatted_time_columns = []
    for column in transpose_reformatted_dataset.columns:
        if "timestamp" in column:
            formatted_time_columns.append(column)

    max_3_calories = transpose_reformatted_dataset.apply(three_maximum_calories, args = (formatted_time_columns, ),axis=1)
    sorted_max_3_calories = max_3_calories.apply(lambda row: sorted(row , key = lambda x:x[0]))

    for i in range(len(sorted_max_3_calories)):
        if len(sorted_max_3_calories[i]) == 2:
            new_row_value = []
            time1 = pd.to_datetime(sorted_max_3_calories[i][0][0])
            calorie1 = pd.to_datetime(sorted_max_3_calories[i][0][1])
            time2 = pd.to_datetime(sorted_max_3_calories[i][1][0])
            calorie2 = pd.to_datetime(sorted_max_3_calories[i][1][1])
            if (0 < time1.hour < 10) and (0 < time2.hour < 10):
                if calorie1 > calorie2:
                    sorted_max_3_calories[i] = [(sorted_max_3_calories[i][0])]
                else:
                    sorted_max_3_calories[i] = [(sorted_max_3_calories[i][1])]
                continue
            elif (10 <= time1.hour < 16) and (10 <= time2.hour < 16):
                if calorie1 > calorie2:
                    sorted_max_3_calories[i] = [(sorted_max_3_calories[i][0])]
                else:
                    sorted_max_3_calories[i] = [(sorted_max_3_calories[i][1])]
                continue
            elif (16 <= time1.hour < 24) and (16 <= time2.hour < 24):
                if calorie1 > calorie2:
                    sorted_max_3_calories[i] = [(sorted_max_3_calories[i][0])]
                else:
                    sorted_max_3_calories[i] = [(sorted_max_3_calories[i][1])]
                continue
            for tuples in sorted_max_3_calories[i]:
                tim = pd.to_datetime(tuples[0])
                if 0 < tim.hour < 10:
                    new_row_value.append(tuples)
                elif (10 <= tim.hour) and (sorted_max_3_calories[i].index(tuples) == 0):
                    new_row_value.append((0, 0, 0, 0))
                if 10 <= tim.hour < 16:
                    new_row_value.append(tuples)
                elif (16 <= tim.hour) and (sorted_max_3_calories[i].index(tuples) == 1) and (len(new_row_value) in (0, 1)):
                    new_row_value.append((10, 0, 0, 0))
                if 16 <= tim.hour < 24:
                    new_row_value.append(tuples)
                elif (tim.hour < 16) and (sorted_max_3_calories[i].index(tuples) == 2):
                    new_row_value.append((16, 0, 0, 0))
            if len(new_row_value) == 2:
                new_row_value.append((16, 0, 0, 0))
            sorted_max_3_calories[i] = new_row_value

    for i in range(len(sorted_max_3_calories)):
        if len(sorted_max_3_calories[i]) == 1:
            new_row_value = []
            for tuples in sorted_max_3_calories[i]:
                tim = pd.to_datetime(tuples[0])
                if 0 < tim.hour < 10:
                    new_row_value.append(tuples)
                    new_row_value.append((10, 0, 0, 0))
                    new_row_value.append((16, 0, 0, 0))
                elif 10 <= tim.hour < 16:
                    new_row_value.append((0, 0, 0, 0))
                    new_row_value.append(tuples)
                    new_row_value.append((16, 0, 0, 0))
                else:
                    new_row_value.append((0, 0, 0, 0))
                    new_row_value.append((10, 0, 0, 0))
                    new_row_value.append(tuples)
            sorted_max_3_calories[i] = new_row_value

    new_dataset = []
    for row in sorted_max_3_calories:
        tuple_value_list = []
        for row_tuple in row:
            tuple_value_list.append(row_tuple[1])
            tuple_value_list.append(row_tuple[2])
            tuple_value_list.append(row_tuple[3])
        new_dataset.append(tuple_value_list)

    new_dataset_df = pd.DataFrame(new_dataset, columns=['First_Meal_Sum', 'First_Meal_Count', 'First_Meal_Variance', 'Second_Meal_Sum', 'Second_Meal_Count','Second_Meal_Variance', 'Third_Meal_Sum', 'Third_Meal_Count', 'Third_Meal_Variance'])
    concatenated_dataset = pd.concat([dataset["SEQN"], new_dataset_df], axis=1)
    return concatenated_dataset
