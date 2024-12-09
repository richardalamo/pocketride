import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, app, jsonify, url_for, render_template
import os
import pyodbc
from dotenv import load_dotenv
import requests

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Fetch the environment variables ( Yet to include .env. Currently hardcoded the credentials)
server = os.getenv('AZURE_SQL_SERVER')
database = os.getenv('AZURE_SQL_DATABASE')
username = os.getenv('AZURE_SQL_USERNAME')
password = os.getenv('AZURE_SQL_PASSWORD')

# Connection string
connection_string = (
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=server;'
    'DATABASE=database;'
    'UID=username;'
    'PWD=password'
)

# Power BI Push Dataset URL
powerbi_push_url = "https://app.powerbi.com/view?r=eyJrIjoiZWEzNjllYTctNWFkMi00MWE3LTljYTItNzljM2IyZGM4NDAzIiwidCI6ImQwOTVkYmRkLWVhOWMtNDM5YS1iYWNmLTQyY2FmMTJiYTEzYiJ9"

def push_to_powerbi(data):
    """Push calculated data to Power BI Push Dataset."""
    headers = {'Content-Type': 'application/json'}
    response = requests.post(powerbi_push_url, json=data, headers=headers)
    if response.status_code == 200:
        print("Data pushed successfully to Power BI")
    else:
        print(f"Failed to push data: {response.status_code}, {response.text}")

def save_to_sql(dataframe):
    """Save DataFrame to Azure SQL Database."""
    conn = None
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        
        delete_query = "TRUNCATE TABLE all_inputs"
        cursor.execute(delete_query)

        for _, row in dataframe.iterrows():
            cursor.execute(
                """
                INSERT INTO all_inputs (zone, month, year, date, hour, farePerMile, avgFlowSpeed_mph,
                 weather_category, n_minus_1_hour_trips, n_minus_2_hour_trips, n_minus_3_hour_trips, weather_input, zone_name )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row['zone'], row['month'], row['year'], row['date'],row['hour'], row['farePerMile'],
                row['avgFlowSpeed_mph'],row['weather_category'],row['n_minus_1_hour_trips'],
                row['n_minus_2_hour_trips'], row['n_minus_3_hour_trips'], row['weather_input'], row['zone_name']
            )
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


# Load your model
model = pickle.load(open("lgbm_model.pkl", "rb"))

# Loading final dataframe to merge fare and flow speed values
df = pd.read_csv('final_df.csv')
df['month'] = df['month'].astype(int) # making sure the data types to be same in the dataframe and the extracted values using flask
df['zone'] = df['zone'].astype(int)
df['hour'] = df['hour'].astype(int)


# n-1,2,3 hour values to be based on average for a given zone, for a given hour for the entire data period
# Modify date from entry to selection of a particular date from the calendar
# power Bi visuals to include two dashboards - one to include map of hot zones, based on demand current hour, 
# Second power BI dashboard, to have filters to select month, or date, and the trends to apply

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict_demand", methods=["GET", "POST"])
def predict_demand():
    form_data = {}
    try:
        # Retrieve input data from the form
        form_data['year'] = request.form.get("year")
        form_data['month'] = int(request.form.get("month"))
        form_data['date'] = request.form.get("date")
        form_data['hour'] = int(request.form.get("hour"))

        
        # Zone mapping
        zone_mapping = {'Bridgeport': 67,
                        'Lincoln Park': 68,
                        'Little Village': 81,
                        'Hyde Park': 58,
                        'Downtown': 87,
                        'China Town': 77,
                        'Cicero': 39,
                        'Irving Park': 57,
                        'Oak Park': 86,
                        'Lincoln Square': 66,
                        'South Shore': 35}
        zone_category = request.form.get("zone")
        if zone_category not in zone_mapping:
            return render_template('index.html', error=f"Invalid zone category: {weather_category}")
        form_data['zone'] = int(zone_mapping[zone_category])
        
        # Extracting fare per mile value based on average for the selected month
        filtered_data = df[df['month']==form_data['month']]
        grouped_data = filtered_data.groupby('month')['farePerMile'].mean().reset_index()
        form_data['farePerMile'] = float(grouped_data.loc[grouped_data['month'] == form_data['month'],'farePerMile'].values[0])

        # Extracting fare per mile value based on average for the selected month
        grouped_data = filtered_data.groupby('month')['avgFlowSpeed_mph'].mean().reset_index()
        form_data['avgFlowSpeed_mph'] = float(grouped_data.loc[grouped_data['month'] == form_data['month'],'avgFlowSpeed_mph'].values[0])

        
        # Mapping weather fields to integer values
        weather_mapping = {
            "cloudy": 0,
            "overcast": 1,
            "rainy": 2,
            "drizzle": 3,
            "snow": 4,
            "sunny": 5,
        }
        weather_category = request.form.get("weather_category")
        if weather_category not in weather_mapping:
            return render_template('index.html', error=f"Invalid weather category: {weather_category}")
        form_data['weather_category'] = int(weather_mapping[weather_category])

        # Extracting n-1, 2, and 3 hour values based on average for that hour in that month just for the purpose of MVP
        # In the actual app, these values should be extracted from the database, where the actual values of demand in previous 
        # hours is recorded and stored
        filtered_data = df[(df['zone']==form_data['zone'])*(df['month']==form_data['month'])]
        grouped_data = filtered_data.groupby(['hour'])[['n_minus_1_hour_trips','n_minus_2_hour_trips','n_minus_3_hour_trips']].mean().reset_index()
        form_data['n_minus_1_hour_trips'] = float(grouped_data.loc[grouped_data['hour'] == form_data['hour'],'n_minus_1_hour_trips'].values[0])
        form_data['n_minus_2_hour_trips'] = float(grouped_data.loc[grouped_data['hour'] == form_data['hour'],'n_minus_2_hour_trips'].values[0])
        form_data['n_minus_3_hour_trips'] = float(grouped_data.loc[grouped_data['hour'] == form_data['hour'],'n_minus_3_hour_trips'].values[0])

        # Pushing input and other data to SQL Database and pushing that data to power BI
        dataframe = pd.DataFrame([form_data])
        dataframe['weather_input'] = weather_category
        dataframe['zone_name'] = zone_category
        save_to_sql(dataframe)
        push_data = dataframe.to_dict(orient='records')
        push_to_powerbi(push_data)

        # Prepare input data for the model
        fields = [
            'zone', 'year', 'month', 'date', 'hour', 
            'farePerMile', 'avgFlowSpeed_mph', 'weather_category', 
            'n_minus_1_hour_trips', 'n_minus_2_hour_trips', 'n_minus_3_hour_trips'
        ]

        # Check for missing or empty fields
        missing_fields = [field for field, value in form_data.items() if value is None]
        if missing_fields:
            return render_template(
                'index.html',
                error=f"Missing values for fields: {', '.join(missing_fields)}"
            )

        # Convert all fields to float
        features = [float(form_data[field]) for field in fields]

        # Use features for prediction (example)
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]

        return render_template('index.html', prediction=round(prediction))

    except Exception as e:
        return render_template("index.html", error=f"{str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
