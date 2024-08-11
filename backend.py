import pandas as pd
import json
import re
from icdmappings import Mapper

import warnings
warnings.filterwarnings("ignore")

class backend:
    def __init__(self):
        print(True)
    
    @classmethod    
    def read_json(self, filename):
        with open(filename) as file:
            data = json.load(file)
            file.close()
        return data

    @classmethod
    def read_file(self, filename, extension, convert_col=None, convert_type=None):
        if extension == "excel":
            if convert_col is None:
                df = pd.read_excel(filename)
            else:
                df = pd.read_excel(filename, converters={convert_col: convert_type})
        elif extension == "csv":
            if convert_col is None:
                df = pd.read_csv(filename)
            else:
                df = pd.read_csv(filename, converters={convert_col: convert_type})
        else:
            return print("The file extension is not supported.")
        
        return df
    
    @classmethod            
    def get_zip3_df(self, df, json_data, cols, update_cols):
        # Get necessary columns
        df = df[json_data[cols]]
        # Change columns names
        df.columns = [json_data[update_cols]]
        # Get zip3 values
        df["zip3"] = df["zip"].apply(lambda x:x.str.slice(0,3))
        df.drop(columns=["zip"], inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df
      
    @classmethod
    def zip_to_zip3(self, zip_county_df):
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        zip_county_df = zip_county_df.copy()
        # Rename the "zip" column to "zip3"
        zip_county_df.rename(columns={"ZIP": "ZIP3"}, inplace=True)
        # Keep only the "ZIP", "COUNTY NAME", and "STATE" columns
        zip_county_df = zip_county_df[["ZIP3", "COUNTYNAME", "STATE"]]
        # Keep only the first three digits of the "zip" column
        zip_county_df['ZIP3'] = zip_county_df['ZIP3'].astype(str).str[:3]
        # Remove duplicate rows
        zip_county_df.drop_duplicates(inplace=True)
        # Reset the index
        zip_county_df.reset_index(drop=True, inplace=True)
        # Remove ' County', ' Borough', ' Area', and 'Parish' from the 'COUNTYNAME' column in zip_county_df
        zip_county_df['COUNTYNAME'] = zip_county_df['COUNTYNAME'].str.replace(' County', '')
        zip_county_df['COUNTYNAME'] = zip_county_df['COUNTYNAME'].str.replace(' Borough', '')
        zip_county_df['COUNTYNAME'] = zip_county_df['COUNTYNAME'].str.replace(' Area', '')
        zip_county_df["COUNTYNAME"] = zip_county_df["COUNTYNAME"].str.replace(" Parish", '')
        zip_county_df["COUNTYNAME"] = zip_county_df["COUNTYNAME"].str.replace(" Municipio", '')
        zip_county_df["COUNTYNAME"] = zip_county_df["COUNTYNAME"].str.replace(" Census Area", '')
        zip_county_df["COUNTYNAME"] = zip_county_df["COUNTYNAME"].str.replace(" city", '')
        
        # COUNTY_STATE = COUNTYNAME + STATE
        zip_county_df['COUNTY_STATE'] = zip_county_df['COUNTYNAME'] + ', ' + zip_county_df['STATE']
        return zip_county_df

    
    @classmethod
    def merge_dataframes(self, aqi_county_dfs, state_abbreviations, df, data):
        print("shape of df: ", df.shape)
        for year in aqi_county_dfs:
            aqi_county_dfs[year] = aqi_county_dfs[year].loc[:, data["merged_keep_cols"]]
        # Loop through each year
        for year in range(2013, 2019):
            # Apply the mapping to the "STATE" column, State2 -> state_abbreviations
            aqi_county_dfs[year]['State2'] = aqi_county_dfs[year]['State'].map(state_abbreviations)
            # Reorder the columns to place 'State2' and 'COUNTY_STATE' next to 'State'
            columns_order = ['County', 'State', 'State2'] + [col for col in aqi_county_dfs[year].columns
                                                                             if col not in ['County', 'State', 'State2']]
            aqi_county_dfs[year] = aqi_county_dfs[year][columns_order]
        # Concatenate all merged DataFrames into a single DataFrame
        final_merged_df = pd.concat(aqi_county_dfs, ignore_index=True)
        print("shape of df: ", final_merged_df.shape)
        final_merged_df.loc[(final_merged_df["County"] == "Washington") & (final_merged_df["State"] == "Maryland"), "State2"] = "DC"
        # Calculate the average AQI across all years for each state
        average_aqi_bystate = final_merged_df.groupby(['State2'], as_index=False).mean(numeric_only=True)
        # Drop the 'Year' column from the resulting DataFrame
        average_aqi_bystate.drop(columns=['Year'], inplace=True)
        # Rename each column by adding 'avbystate' at the end of their names
        average_aqi_bystate.columns = [col + '(avbystate)' if col not in [ 'State2'] else col for col in average_aqi_bystate.columns]
        # Print the resulting Series with the average AQI values by state
        print(average_aqi_bystate)
        print("shape of df: ", df.shape)
        # Merge df and final_merged_df on the state
        merged_final_df = pd.merge(df, average_aqi_bystate, left_on='state', right_on='State2',
                                        how='inner')
        print("shape of df: ", merged_final_df.shape)
        # Remove the "State2" column from the DataFrame
        merged_final_df = merged_final_df.drop(columns=['State2'])
        return merged_final_df

    @classmethod
    def calculate_monthly_avg_temperature(self, df, data, num_months):
        # Generate the sorted list of months and years
        sorted_lst = []
        for aMonth in range(1, num_months + 1):
            for aYear in data["year_lst"]:
                sorted_lst.append(str(aMonth) + aYear)

        # Update the temporary list
        updated_tmp_list = []
        for i in range(0, len(sorted_lst), 6):
            updated_tmp_list.append(sorted_lst[i:i + 6])

        # Select temperature columns
        temperature_columns = [col for col in df.columns if 'Average of' in col]
        # Melt the DataFrame to get months, years, and temperatures in a more manageable form
        temperature_data = df.melt(id_vars=['patient_zip3'], value_vars=temperature_columns,
                                         var_name='Month_Year', value_name='Temperature')
        # Extract just the month name to average across all years
        temperature_data['Month'] = temperature_data['Month_Year'].apply(lambda x: x.split(' ')[2][:3])
        # Group by 'Zip3' and 'Month', and calculate the average temperature across all years
        monthly_average_temperatures = temperature_data.groupby(['patient_zip3', 'Month'])[
            'Temperature'].mean().reset_index()
        # List of all months
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        # Pivot the table to have months as columns and zip codes as rows
        pivot_table_month = monthly_average_temperatures.pivot(index='patient_zip3', columns=['Month'],
                                                               values='Temperature')
        # Reindex the columns to include all months
        pivot_table_month = pivot_table_month.reindex(columns=months)
        # Rename the columns to match the format 'xxx_temp'
        pivot_table_month.columns = [f'temp_{month.lower()}' for month in pivot_table_month.columns]
        # Reset the index to turn the zip codes into a column
        pivot_table_month.reset_index(inplace=True)

        # calculate month average
        i = 0
        for zip_code in df['patient_zip3'].unique():
            # Filter df_zip3 to include only rows with the current zip_code
            df_filtered = df[df['patient_zip3'] == zip_code]
            pivot_table_month.at[i, "patient_zip3"] = zip_code
            for month_data in updated_tmp_list:
                # Calculate average temperature
                monthly_sum = df_filtered[month_data].sum(axis=1)
                monthly_avg = monthly_sum / len(data["year_lst"])
                # Get the index of the month
                month_index = int(month_data[0].split('_')[0]) - 1
                # Get the month string
                monthstr = f'temp_{months[month_index]}'
                # Assign the average temperature to the corresponding cell in the pivot_table_month
                pivot_table_month.at[i, monthstr] = float(monthly_avg.iloc[0])
            # Increment i for the next row
            i += 1

        # Merge the pivot_table_month with the original DataFrame df based on patient_zip3
        df = df.merge(pivot_table_month, on='patient_zip3', how='left')

        return df

    @classmethod
    def calculate_yearly_avg_temperature(self, df, data, num_months):
        # Generate the sorted list of months and years
        sorted_year_lst = []
        for aYear in data["year_lst"]:
            for aMonth in range(1, num_months + 1):
                sorted_year_lst.append(str(aMonth) + aYear)
        print(sorted_year_lst)

        # Update the temporary list
        updated_tmp_year_list = []
        for i in range(0, len(sorted_year_lst), 12):
            updated_tmp_year_list.append(sorted_year_lst[i:i + 12])

        # Select temperature columns
        temperature_columns = [col for col in df.columns if 'Average of' in col]
        # Melt the DataFrame to get months, years, and temperatures in a more manageable form
        temperature_data = df.melt(id_vars=['patient_zip3'], value_vars=temperature_columns,
                                         var_name='Month_Year', value_name='Temperature')
        # Extract just the month name to average across all years
        temperature_data['Month'] = temperature_data['Month_Year'].apply(lambda x: x.split(' ')[2][:3])
        # Group by 'Zip3' and 'Month', and calculate the average temperature across all years
        years_average_temperatures = temperature_data.groupby(['patient_zip3', 'Month'])[
            'Temperature'].mean().reset_index()
        # List of all years
        years = ['13', '14', '15', '16', '17', '18']
        # Pivot the table to have years as columns and zip codes as rows
        pivot_table_year = years_average_temperatures.pivot(index='patient_zip3', columns=['Month'],
                                                            values='Temperature')
        # Reindex the columns to include all years
        pivot_table_year = pivot_table_year.reindex(columns=years)
        # Rename the columns to match the format 'xxx_temp'
        pivot_table_year.columns = [f'temp_{year.lower()}' for year in pivot_table_year.columns]
        # Reset the index to turn the zip codes into a column
        pivot_table_year.reset_index(inplace=True)

        # year average
        i = 0
        for zip_code in df['patient_zip3'].unique():
            # Filter df_zip3 to include only rows with the current zip_code
            print(zip_code)
            df_filtered = df[df['patient_zip3'] == zip_code]
            pivot_table_year.at[i, "patient_zip3"] = zip_code
            for year_data in updated_tmp_year_list:
                print(year_data)
                # Calculate average temperature
                yearly_sum = df_filtered[year_data].sum(axis=1)
                # print(yearly_sum)
                yearly_avg = yearly_sum / 12
                print(yearly_avg)
                # Get the index of the month
                year = "temp_" + year_data[0].split('_')[1]
                print(year)
                # Assign the average temperature to the corresponding cell in the pivot_table_year
                pivot_table_year.at[i, year] = float(yearly_avg.iloc[0])

            # Increment i for the next row
            i += 1

        # Merge the pivot_table_month with the original DataFrame df based on patient_zip3
        df = df.merge(pivot_table_year, on='patient_zip3', how='left')

        return df
    
    @classmethod
    def fill_missing_values_category(self, df, col_name, fill_value):
        df[col_name] = df[col_name].fillna(fill_value)
        df[col_name] = df[col_name].str.lower()
        return df
    
    @classmethod
    def remove_nested_parentheses(self, s):
    # Regular expression to find the outermost parentheses
        while '(' in s and ')' in s:
            s = re.sub(r'\([^()]*\)', '', s)
        return s.strip()
    
    @classmethod
    def get_disaster_df(self, df, data):
        new_df = df[data["disaster_cols"]]
        # Convert date to datetime
        new_df["declarationDate"] = pd.to_datetime(new_df["declarationDate"]).dt.strftime("%Y%m%d")
        # Remove unnecessary columns
        final_df = new_df[data["disaster_update_cols"]]
        # Rename column names
        final_df.rename(columns={"incidentType": "incident_type",
                               "declarationDate": "declaration_date",
                               "designatedArea": "county"},
                      inplace=True)
        # Get county state names
        # final_df["COUNTY_STATE"] = final_df["county"] + "," + final_df["state"]
        # Merge zip_county to disaster
        # merged_df = final_df.merge(zip_county_df, on="COUNTY_STATE")
        final_df = final_df.drop(columns=["incident_type"])
        
        # Get year as a new column and convert it to integer
        final_df["year"] = final_df["declaration_date"].str.extract(r"(\d{4})")
        final_df["year"] = final_df["year"].astype(int)
        final_df = final_df[(final_df["year"] >= 2013) & (final_df["year"] <= 2018)]
        # Create a new column for pivot table aggfunction
        final_df["count"] = 1
        disaster_final = pd.pivot_table(final_df, values="count", index="state", columns=["year"], aggfunc="sum", fill_value=0)
        # Convert the pivot table to dataframe
        disaster_final_df = pd.DataFrame(disaster_final.to_records())
        disaster_final_df.rename(columns={
            "2013": "disaster_num_2013",
            "2014": "disaster_num_2014",
            "2015": "disaster_num_2015",
            "2016": "disaster_num_2016",
            "2017": "disaster_num_2017",
            "2018": "disaster_num_2018"
        }, inplace=True)
        return disaster_final_df
    
    @classmethod
    def get_nri_df(self, df, data):
        # Remove unnecessary columns
        df = df[data["nri_cols"]]
        # Get COUNTY_STATE by combining county and stateabbrv
        df["COUNTY_STATE"] = df["COUNTY"] + ", " + df["STATEABBRV"]
        # Remove unnecessary columns
        df = df[data["nri_update_cols"]]
        # Remove missing values
        df = df[df["RISK_RATNG"] != "Insufficient Data"]
        # Get duplicate data and use mean value
        duplicated_state_county = df.duplicated(subset=["COUNTY_STATE"])
        nri_duplicated_list = []
        if duplicated_state_county.any():
            nri_duplicated_list = df.loc[duplicated_state_county]["COUNTY_STATE"].to_list()
        tmp_df = pd.DataFrame(columns=data["nri_update_cols"])
        for aCounty in nri_duplicated_list:
            update_df = df[df["COUNTY_STATE"] == aCounty]
            score_sum = 0
            for i in range(len(update_df)):
                score_sum += update_df["RISK_SCORE"].iloc[i]
            avg_score = score_sum / len(update_df)
            risk_rating = df[(df["RISK_SCORE"] < avg_score + 0.5) & (df["RISK_SCORE"] > avg_score - 0.5)]["RISK_RATNG"].iloc[0]
            tmp_df.loc[len(tmp_df)] = [aCounty, avg_score, risk_rating]
        
        for aCounty in nri_duplicated_list:
            df.loc[df["COUNTY_STATE"] == aCounty, "RISK_SCORE"] = tmp_df[tmp_df["COUNTY_STATE"] == aCounty]["RISK_SCORE"].iloc[0]
            df.loc[df["COUNTY_STATE"] == aCounty, "RISK_RATNG"] = tmp_df[tmp_df["COUNTY_STATE"] == aCounty]["RISK_RATNG"].iloc[0]
        # Remove duplicate data
        df = df.drop_duplicates()
        return df
    
    @classmethod
    def starts_with_letter(self, text):
        return bool(re.match(r'^[a-zA-Z]', text))
    
    @classmethod
    def get_icd10(self, df):
        mapper = Mapper()
        df["breast_cancer_diagnosis_icd10"] = None
        icd9code = None
        for i in range(len(df)):
            if df["breast_cancer_diagnosis_code_isIcd"][i]:
                df["breast_cancer_diagnosis_icd10"][i] = df["breast_cancer_diagnosis_code"][i]
            else:
                icd9code = df["breast_cancer_diagnosis_code"][i]
                # print(icd9code)
                df["breast_cancer_diagnosis_icd10"][i] = mapper.map(icd9code, source="icd9", target="icd10")
        df["breast_cancer_diagnosis_icd10"] = df["breast_cancer_diagnosis_icd10"].str.lower()
        return df
    
    @classmethod
    def check_cold_conditions(self, row, column_averages):
        conditions_met = 0
        for col in row.index:
            if row[col] < 32 or row[col] < column_averages[col] - 10:
                conditions_met += 1
        return conditions_met
 
    
    @classmethod
    def get_extremem_temp(self, df, data):
        avg_temps = df[data["tmp_columns"]].mean()
        df["days_of_extreme_heat"] = df[data["tmp_columns"]].apply(lambda x: x>=90).sum(axis=1)
        df["days_of_extreme_cold"] = df[data["tmp_columns"]].apply(self.check_cold_conditions, axis=1, args=(avg_temps,))
        return df
    
    @classmethod
    def temp_agg(self, df, data):
        df["temp_std_dev"] = df[data["tmp_columns"]].std(axis=1)
        df["temp_range"] = df[data["tmp_columns"]].max(axis=1) - df[data["tmp_columns"]].min(axis=1)
        df["winter_avg_2013"] = df[data["tmp_win_2013"]].mean(axis=1)
        df["winter_avg_2014"] = df[data["tmp_win_2014"]].mean(axis=1)
        df["winter_avg_2015"] = df[data["tmp_win_2015"]].mean(axis=1)
        df["winter_avg_2016"] = df[data["tmp_win_2016"]].mean(axis=1)
        df["winter_avg_2017"] = df[data["tmp_win_2017"]].mean(axis=1)
        df["winter_avg_2018"] = df[data["tmp_win_2018"]].mean(axis=1)
        df["summer_avg_2013"] = df[data["tmp_summer_2013"]].mean(axis=1)
        df["summer_avg_2014"] = df[data["tmp_summer_2014"]].mean(axis=1)
        df["summer_avg_2015"] = df[data["tmp_summer_2015"]].mean(axis=1)
        df["summer_avg_2016"] = df[data["tmp_summer_2016"]].mean(axis=1)
        df["summer_avg_2017"] = df[data["tmp_summer_2017"]].mean(axis=1)
        df["summer_avg_2018"] = df[data["tmp_summer_2018"]].mean(axis=1)
        df["fall_avg_2013"] = df[data["tmp_fall_2013"]].mean(axis=1)
        df["fall_avg_2014"] = df[data["tmp_fall_2014"]].mean(axis=1)
        df["fall_avg_2015"] = df[data["tmp_fall_2015"]].mean(axis=1)
        df["fall_avg_2016"] = df[data["tmp_fall_2017"]].mean(axis=1)
        df["fall_avg_2017"] = df[data["tmp_fall_2017"]].mean(axis=1)
        df["fall_avg_2018"] = df[data["tmp_fall_2018"]].mean(axis=1)
        df["spring_avg_2013"] = df[data["tmp_spring_2013"]].mean(axis=1)
        df["spring_avg_2014"] = df[data["tmp_spring_2014"]].mean(axis=1)
        df["spring_avg_2015"] = df[data["tmp_spring_2015"]].mean(axis=1)
        df["spring_avg_2016"] = df[data["tmp_spring_2016"]].mean(axis=1)
        df["spring_avg_2017"] = df[data["tmp_spring_2017"]].mean(axis=1)
        df["spring_avg_2018"] = df[data["tmp_spring_2018"]].mean(axis=1)
        
        return df

    def add_obesity_column(df, obesity_threshold=30):
        # Update 'obese' column based on BMI values and threshold
        df['obese'] = 0  # Initialize the column
        df.loc[df["bmi"].notnull() & (df["bmi"] >= obesity_threshold), 'obese'] = 2
        df.loc[df["bmi"].notnull() & (df["bmi"] < obesity_threshold), 'obese'] = 1
        # Drop the 'bmi' column
        df = df.drop(columns=["bmi"])

        return df

