from flask import Flask, render_template, url_for, request, redirect
from pycaret.classification import load_model,predict_model
import joblib
import pandas as pd
import pycaret
import folium
from folium.plugins import HeatMap
import math
import itertools
import numpy as np

app = Flask(__name__)

# model = joblib.load('tuned_dt_wildfire.pkl')
model = load_model("tuned_dt_wildfire")
df_org = pd.read_csv('https://raw.githubusercontent.com/hugorpereira/wildfire-prediction/refs/heads/main/data_analysis/df_wildfire_cleaned.csv') #nrows=5000
df_org.drop(columns=['fire_year','general_cause_desc','fire_type','weather_conditions_over_fire','fuel_type'], axis=1, inplace=True)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/understanding')
def understanding():
    return render_template('understanding.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/prediction', methods=['GET', 'POST'] )
def prediction():
    if request.method == 'GET':
        
        severity_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        df_org["weight"] = df_org["size_class"].map(severity_map)
        heat_data = df_org[["fire_location_latitude", "fire_location_longitude", "weight"]].values.tolist()

        folium_map = folium.Map(location=[df_org.fire_location_latitude.mean(), df_org.fire_location_longitude.mean()], zoom_start=5)
        HeatMap(heat_data, radius=20, blur=15).add_to(folium_map)
        folium_map = folium_map._repr_html_()

        return render_template('prediction.html', folium_map=folium_map)
    
    elif request.method == "POST":
        # Get form data
        data = request.form.to_dict()

        try:            
            # Create a dataframe from form data
            df = pd.DataFrame([data])

            df['fire_lat_sin'] = np.sin(np.radians(float(df['fire_location_latitude'])))
            df['fire_lat_cos'] = np.cos(np.radians(float(df['fire_location_latitude'])))
            df['fire_lon_sin'] = np.sin(np.radians(float(df['fire_location_longitude'])))
            df['fire_lon_cos'] = np.cos(np.radians(float(df['fire_location_longitude'])))

            df.drop(columns=['fire_location_latitude','fire_location_longitude'], axis=1, inplace=True)

            # Execute prediction
            # prediction = model.predict(df)
            prediction_df = predict_model(model, data=df)
            prediction = prediction_df["prediction_label"]

            # Get class data based on prediction
            classData = getClassData(prediction[0])

            # probabilities = model.predict_proba(df)
            # confidence = max(probabilities[0]) * 100

            # Process variants
            variation_dict = {
                "fire_origin": ['DND','Indian Reservation','Metis Settlement','Private Land','Provincial Land','Provincial Park','Saskatchewan'],
                "fuel_type": ['C1','C2','C3','C4','C6','C7','D1','M1','M2','M3','M4','O1a','O1b','S1','S2'],
                "weather_conditions_over_fire": ['CB Dry','CB Wet','Clear','Cloudy','Rainshowers'],
                "general_cause_desc": ['Forest Industry','Incendiary','Lightning','Miscellaneous Known','Oil & Gas Industry','Other Industry','Power Line Industry','Prescribed Fire','Railroad','Recreation','Resident','Restart','Undetermined'],
                "fire_type": ['Surface', 'Crown', 'Ground']
            }

            dfs = []
            for col, values in variation_dict.items():
                expanded = pd.concat(
                    [df.assign(**{col: val}) for val in values],
                    ignore_index=True
                )
                expanded["varied_feature"] = col
                dfs.append(expanded)
            
            complement_df = pd.concat(dfs, ignore_index=True)
            result = predict_model(model, complement_df)
            result = result[result['prediction_label'] != prediction[0]]

            filtered_results = {}
            if not result.empty:
                grouped_results = {
                    feature: df_group
                    for feature, df_group in result.groupby("varied_feature")
                }
                
                for feature, df_group in grouped_results.items():
                    filtered_results[feature] = df_group[[feature, "prediction_label"]]

            # Process critical cases
            variation_dict_reduced = {
                "fire_origin": ['Indian Reservation','Provincial Land'],
                "fuel_type": ['C1','C2'],
                "weather_conditions_over_fire": ['CB Dry','Clear','Cloudy'],
                "general_cause_desc": ['Lightning','Power Line Industry','Resident'],
                "fire_type": ['Surface', 'Crown']
            }

            all_combinations = list(itertools.product(*variation_dict_reduced.values()))
            df_variations = pd.DataFrame(all_combinations, columns=variation_dict_reduced.keys())
            # df_variations["fire_location_latitude"] = data['fire_location_latitude']
            # df_variations["fire_location_longitude"] = data['fire_location_longitude']
            # df_variations["fire_year"] = data['fire_year']

            df_variations['fire_lat_sin'] = np.sin(np.radians(float(data['fire_location_latitude'])))
            df_variations['fire_lat_cos'] = np.cos(np.radians(float(data['fire_location_latitude'])))
            df_variations['fire_lon_sin'] = np.sin(np.radians(float(data['fire_location_longitude'])))
            df_variations['fire_lon_cos'] = np.cos(np.radians(float(data['fire_location_longitude'])))


            result_complement = predict_model(model, df_variations)
            critical_sorted = {}
            if not result_complement.empty:
                critical = result_complement[result_complement["prediction_label"].isin(["E", "D"])]

                severity_order = {"E": 0, "D": 1}
                critical["severity_rank"] = critical["prediction_label"].map(severity_order)
                critical_sorted = critical.sort_values("severity_rank").drop(columns="severity_rank")

                critical_sorted.drop(columns=['prediction_score'], inplace=True)
                critical_list = critical_sorted.to_dict(orient="records")

            # Create map
            zoom_size = 12 if classData["fireClass"] == 'E' or classData["fireClass"] == 'D' else 16
            folium_map = folium.Map(location=[data['fire_location_latitude'], data['fire_location_longitude']], zoom_start=zoom_size)
            folium.Marker([data['fire_location_latitude'], data['fire_location_longitude']], popup="Location").add_to(folium_map)
            folium.Circle([data['fire_location_latitude'], data['fire_location_longitude']], radius=classData['radius'], color="red", fill=True, fill_opacity=0.3).add_to(folium_map)

            # Add Heatmap
            severity_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
            df_org["weight"] = df_org["size_class"].map(severity_map)
            heat_data = df_org[["fire_location_latitude", "fire_location_longitude", "weight"]].values.tolist()
            HeatMap(heat_data, radius=20, blur=15).add_to(folium_map)
            
            # Get map html
            folium_map = folium_map._repr_html_()
            
            return render_template('prediction.html', 
                                     prediction=classData,
                                     form_data=data,
                                     folium_map=folium_map,
                                     variants=filtered_results,
                                     critical_sorted=critical_list)

        except Exception as e:
            return render_template('prediction.html',
                                    error=str(e),
                                    form_data=data,)

def getClassData(prediction):
    if not prediction:
        raise ValueError("Invalid prediction!")
    
    fire_classes = {
        "A": {
            "fireClass": "A",
            "rangeHa": "0.01 - 0.1 ha",
            "radius": math.sqrt(0.1 * 10000 / math.pi),
            "severity": "Incipient Fire"
        },
        "B": {
            "fireClass": "B", 
            "rangeHa": "0.11 - 4.0 ha",
            "radius": math.sqrt(4.0 * 10000 / math.pi),
            "severity": "Small Fire"
        },
        "C": {
            "fireClass": "C",
            "rangeHa": "4.1 - 40.0 ha", 
            "radius": math.sqrt(40.0 * 10000 / math.pi),
            "severity": "Moderate Fire"
        },
        "D": {
            "fireClass": "D",
            "rangeHa": "40.1 - 200.0 ha",
            "radius": math.sqrt(200.0 * 10000 / math.pi),
            "severity": "Large Fire"
        },
        "E": {
            "fireClass": "E",
            "rangeHa": "200.1+ ha",
            "radius": math.sqrt(8000.0 * 10000 / math.pi),
            "severity": "Extreme Fire"
        }
    }

    if prediction not in fire_classes:
        raise ValueError(f"Invalid prediction class: '{prediction}'")
    
    return fire_classes[prediction]

if __name__ == '__main__':
    app.run(debug=True)