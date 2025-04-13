from flask import Flask, request, jsonify, render_template, send_from_directory
from feature_extraction import generate_feature_vector, calculate_distance, draw_landmark_rectangles, extract_facial_features
import datetime
from dotenv import load_dotenv
import cv2
from pymongo import MongoClient
import os
import numpy as np
import uuid

app = Flask(__name__, template_folder='../templates', static_folder='../static')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home', methods=['GET'])
def index():
    return render_template('index.html') 

@app.route('/upload', methods=['POST'])
def upload():
    if 'imageU' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['imageU']
    image_path = f"/tmp/{image.filename}"
    image.save(image_path)

    name = request.form['name']

    try:
        features = generate_feature_vector(image_path)

        load_dotenv()
        client = MongoClient("mongodb+srv://jj1057:Fo7NByhT@aihackfest2025.pkixofv.mongodb.net/?retryWrites=true&w=majority&appName=AIHackFest2025")
        db = client['FaceStorage']
        collection = db['Faces']

        feature_vector = features.tolist()  
        timestamp = datetime.datetime.now().isoformat()
        
        collection.insert_one({
            'id': str(uuid.uuid4()),
            'feature_vector': feature_vector,
            'timestamp': timestamp,
            'name': name
        })

        return render_template(
            'upload.html',
            imageU_url=f"/tmp/{image.filename}",
            name=name,
            timestamp=timestamp
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500  

@app.route('/compare', methods=['POST'])
def compare():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Please upload both images.'}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    image1_path = f"/tmp/{image1.filename}"
    image2_path = f"/tmp/{image2.filename}"

    image1.save(image1_path)
    image2.save(image2_path)

    try:
        img1, landmarks1 = extract_facial_features(image1_path)
        img2, landmarks2 = extract_facial_features(image2_path)

        if img1 is None or img2 is None:
            return jsonify({'error': 'Could not read one or both images.'}), 400
        
        if landmarks1 is None or landmarks2 is None:
            return jsonify({'error': 'Could not detect faces in one or both images.'}), 400

        distance = calculate_distance(landmarks1, landmarks2)
    
        indices_to_highlight = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        img1_annotated = draw_landmark_rectangles(img1, landmarks1, indices_to_highlight)
        img2_annotated = draw_landmark_rectangles(img2, landmarks2, indices_to_highlight)

        annotated_image1_path = f"/tmp/annotated_{image1.filename}"
        annotated_image2_path = f"/tmp/annotated_{image2.filename}"
        cv2.imwrite(annotated_image1_path, img1_annotated)
        cv2.imwrite(annotated_image2_path, img2_annotated)

        return render_template(
            'compare_result.html',
            image1_url=f"/tmp/annotated_{image1.filename}",
            image2_url=f"/tmp/annotated_{image2.filename}",
            similarity_score=distance
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # finally:
    #     if os.path.exists(image1_path):
    #         os.remove(image1_path)
    #     if os.path.exists(image2_path):
    #         os.remove(image2_path)

@app.route('/names', methods=['GET'])
def get_names():
    load_dotenv()
    client = MongoClient("mongodb+srv://jj1057:Fo7NByhT@aihackfest2025.pkixofv.mongodb.net/?retryWrites=true&w=majority&appName=AIHackFest2025")
    db = client['FaceStorage']
    collection = db['Faces']

    names = collection.distinct('name')
    return render_template('names.html', names=names)

@app.route('/compare_with_name', methods=['POST'])
def compare_with_name():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Please select a name and upload an image.'}), 400

    name = request.form['name']
    image = request.files['image']
    image_url = f"/tmp/{image.filename}"
    image.save(image_url)

    try:
        _, uploaded_features = extract_facial_features(image_url)
        if uploaded_features is None:
            return jsonify({'error': 'Could not detect a face in the uploaded image.'}), 400

        load_dotenv()
        client = MongoClient("mongodb+srv://jj1057:Fo7NByhT@aihackfest2025.pkixofv.mongodb.net/?retryWrites=true&w=majority&appName=AIHackFest2025")
        db = client['FaceStorage']
        collection = db['Faces']

        record = collection.find_one({'name': name})
        if not record:
            return jsonify({'error': f'No record found for name: {name}'}), 404

        database_features = np.array(record['feature_vector'])

        distance = calculate_distance(uploaded_features, database_features)

        return render_template(
            'compare_result.html',
            image_url=image_url,
            database_image_url=f"/tmp/{record['id']}.jpg",
            similarity_score=distance
        )

    except Exception as e:  
        return jsonify({'error': str(e)}), 500

@app.route('/tmp/<filename>')
def serve_temp_file(filename):
    return send_from_directory('/tmp', filename)

if __name__ == "__main__":
    app.run(port=5000, debug=True)