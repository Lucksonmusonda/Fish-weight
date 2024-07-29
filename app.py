from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('fish_weight_model.pkl')
encoder = joblib.load('species_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    species = request.form['species']
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[species, length1, length2, length3, height, width]],
                              columns=['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width'])

    # Encode the species
    encoded_species = encoder.transform(input_data[['Species']])
    encoded_species_df = pd.DataFrame(encoded_species, columns=encoder.get_feature_names_out(['Species']))

    # Combine the encoded species with the other input data
    input_data_encoded = pd.concat([input_data.drop(columns=['Species']), encoded_species_df], axis=1)

    # Run inference with the loaded model
    prediction = model.predict(input_data_encoded)

    # Render the result page
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
