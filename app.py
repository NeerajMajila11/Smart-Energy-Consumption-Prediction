from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import joblib

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PREDICTION_FOLDER = "predictions"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PREDICTION_FOLDER"] = PREDICTION_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Load the trained Random Forest model
model = joblib.load("models/saved_models/random_forest.pkl")


@app.route("/")
def home():
    return render_template("index.html")

print("13. Starting prediction")
@app.route("/predict", methods=["POST"])

def predict():

    print("1. Predict function started")

    file = request.files["file"]
    print("2. File received")

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)
    print("3. File saved")

    df = pd.read_csv(filepath)
    print("4. CSV loaded", df.shape)

# Remove target column if present
    if "Appliances" in df.columns:
        df = df.drop(columns=["Appliances"])

# Remove date column if present
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    print("5. Dataset ready for prediction", df.shape)
    expected_columns = [
    'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4',
    'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
    'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out',
    'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2',
    'Hour', 'Day', 'Week', 'Year', 'Weekend',
    'Lag_1', 'Lag_2', 'Lag_6', 'Lag_12', 'Lag_24',
    'Rolling_Mean_6', 'Rolling_Mean_12',
    'Rolling_STD_6', 'Rolling_STD_12',
    'DayOfWeek_Monday', 'DayOfWeek_Saturday',
    'DayOfWeek_Sunday', 'DayOfWeek_Thursday',
    'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday',
    'Month_February', 'Month_January',
    'Month_March', 'Month_May'
]

    # Add missing columns with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Remove extra columns and arrange in correct order
    df = df.reindex(columns=expected_columns, fill_value=0)
    # Predict
    predictions = model.predict(df)

    # Save predictions
    result = df.copy()
    result["Predicted_Appliances"] = predictions

    output_path = os.path.join(
        app.config["PREDICTION_FOLDER"],
        "predictions.csv"
    )

    result.to_csv(output_path, index=False)

    return render_template(
    "result.html",
    tables=[result.head(10).to_html(classes="table table-bordered table-hover", index=False)],
    total=len(result),
    avg=round(result["Predicted_Appliances"].mean(),2),
    maximum=round(result["Predicted_Appliances"].max(),2),
    minimum=round(result["Predicted_Appliances"].min(),2),
    predictions=result["Predicted_Appliances"].head(50).tolist(),

    mae=21.39,
    rmse=45.68,
    r2=0.8033
)

@app.route("/download")
def download():

    output_path = os.path.join(
        app.config["PREDICTION_FOLDER"],
        "predictions.csv"
    )

    return send_file(
        output_path,
        as_attachment=True
    )
if __name__ == "__main__":
    app.run(debug=True)