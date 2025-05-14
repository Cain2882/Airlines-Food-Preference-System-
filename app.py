from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files

# Initialize combined dataset variable to None
app.combined_dataset_df = None
app.clf = None
app.vectorizer = None

# Process uploaded files, combine datasets, and train model
def process_and_train(csv_airline, csv_food_menu):
    try:
        df1 = pd.read_csv(csv_airline)
        df2 = pd.read_csv(csv_food_menu)

        combined_dataset = []
        for index1, row1 in df1.iterrows():
            serial_number = row1.get("S_No")
            person_name = row1.get("Name")
            comfort_foods = row1.get("comfort_food", "").split(", ")
            identified_foods = []
            food_list = {}

            for category in ["Candy", "Liquid_Food", "Dessert", "Fast_Food", "Snacks"]:
                if pd.notna(df2[category].iloc[0]):
                    food_data = df2[category].iloc[0].split(", ")
                    if len(food_data) >= 2:
                        food_name = food_data[0]
                        quantity = int(food_data[1])
                        food_list[category] = f"{food_name} (Available: {quantity})"

                        # Check if comfort food matches the food menu item and there is quantity available
                        for comfort_food in comfort_foods:
                            if comfort_food == food_name and quantity > 0:
                                identified_foods.append(comfort_food)
                                new_quantity = quantity - 1
                                df2.at[0, category] = f"{food_name}, {new_quantity}"

            combined_dataset.append({
                "Serial Number": serial_number,
                "Name": person_name,
                "Comfort Foods": ", ".join(comfort_foods),
                "Identified Foods": ", ".join(identified_foods) if identified_foods else "None",
                **food_list  # Add the food menu details to the dataset
            })

        combined_dataset_df = pd.DataFrame(combined_dataset)

        # For machine learning purposes (you can keep or remove based on requirements)
        X = combined_dataset_df["Identified Foods"]
        y = combined_dataset_df["Name"]

        vectorizer = CountVectorizer()
        X_vectorized = vectorizer.fit_transform(X)
        clf = RandomForestClassifier()
        clf.fit(X_vectorized, y)

        return clf, vectorizer, combined_dataset_df

    except Exception as e:
        print(f"Error in processing and training: {e}")
        return None, None, None

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file uploads and process data
@app.route('/upload', methods=['POST'])
def upload_files():
    if 'airline_file' not in request.files or 'food_menu_file' not in request.files:
        return jsonify({'response': 'Please upload both files!'})

    airline_file = request.files['airline_file']
    food_menu_file = request.files['food_menu_file']

    # Save the uploaded files
    airline_path = os.path.join(app.config['UPLOAD_FOLDER'], airline_file.filename)
    food_menu_path = os.path.join(app.config['UPLOAD_FOLDER'], food_menu_file.filename)
    
    try:
        airline_file.save(airline_path)
        food_menu_file.save(food_menu_path)
    except Exception as e:
        return jsonify({'response': f'Error saving files: {e}'})

    # Train the model with the uploaded datasets
    clf, vectorizer, combined_dataset_df = process_and_train(airline_path, food_menu_path)

    if clf is None or vectorizer is None or combined_dataset_df is None:
        return jsonify({'response': 'Error processing and training the model. Please check the dataset format.'})

    # Store the model and vectorizer in global variables
    app.clf = clf
    app.vectorizer = vectorizer
    app.combined_dataset_df = combined_dataset_df  # Store the combined dataset for later use

    return jsonify({'response': 'Model trained successfully with the uploaded datasets!'})

# Route for chatbot interaction (Optional)
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input or not hasattr(app, 'clf') or app.clf is None:
        return jsonify({'response': 'Please upload datasets first or provide a valid input.'})

    input_vector = app.vectorizer.transform([user_input])
    prediction = app.clf.predict(input_vector)
    
    response = f"The identified foods for your input: {prediction[0]}"
    return jsonify({'response': response})

# Route to display the combined dataset in a well-formatted table
@app.route('/details', methods=['GET'])
def show_details():
    if app.combined_dataset_df is None:
        return jsonify({'response': 'No data available. Please upload datasets first.'})

    # Render the combined data on an HTML page
    dataset_dict = app.combined_dataset_df.to_dict(orient='records')
    return render_template('templates.html', dataset=dataset_dict)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)