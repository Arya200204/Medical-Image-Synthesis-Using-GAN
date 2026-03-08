from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file 
import mysql.connector
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = '59dd11478de1189f8a736f37dddbbb88401b57994deaf854a71b9ba638c300de'

# Load pre-trained GAN model (.h5)
generator = load_model('generator_cgan_sn.h5', compile=False) 

# MySQL DB connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password='123',
    database="medxgan"
)
cursor = conn.cursor(dictionary=True)

# === IMAGE PREPROCESSING: Grayscale for X-ray / MRI ===
def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((256, 256))
    array = np.array(image).astype(np.float32)
    normalized = (array / 127.5) - 1.0
    array = np.expand_dims(normalized, axis=-1)
    return np.expand_dims(array, axis=0)

# === POSTPROCESSING: Convert model output to grayscale PIL ===
def postprocess_image(output_array):
    output_array = output_array[0, :, :, 0]
    output_array = ((output_array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output_array, mode='L')

# === File validation ===
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        phone = request.form['phone']
        role = request.form['role']

        role_map = {'Admin': 1, 'Doctor': 2, 'Patient': 3}
        role_id = role_map.get(role)

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Email already registered. Please login.")
            return redirect(url_for('login'))

        cursor.execute("""
            INSERT INTO users (name, email, password, phone, role_id)
            VALUES (%s, %s, %s, %s, %s)
        """, (name, email, password, phone, role_id))
        conn.commit()

        flash("Registration successful. Please login.")
        return redirect(url_for('login'))

    return render_template("user_register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
        user = cursor.fetchone()
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['role_id'] = user['role_id']
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password")
            return redirect(url_for('login'))
    return render_template("user_login.html")

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("user_dashboard.html", name=session['user_name'])

@app.route('/select_image_type', methods=['POST'])
def select_image_type():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    image_type = request.form.get('image_type')
    if image_type:
        session['image_type'] = image_type
        return redirect(url_for('upload'))
    else:
        flash("Please select an image type.")
        return redirect(url_for('dashboard'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('image')
        image_type = session.get('image_type')  # from session

        if not file or not allowed_file(file.filename):
            flash("Please upload a valid image file (JPG, JPEG, PNG).")
            return redirect(url_for('upload'))

        if not image_type:
            flash("Image type not selected.")
            return redirect(url_for('upload'))

        try:
            print("🔍 Received file:", file.filename)

            # Load and preprocess
            image = Image.open(file.stream).convert("RGB")
            print("📷 Image loaded. Size:", image.size)

            input_tensor = preprocess_image(image)
            print("📐 Preprocessed image shape:", input_tensor.shape)

            # Generate output
            output = generator.predict([input_tensor_1, input_tensor_2])
            print("🎯 Prediction done. Output shape:", output.shape)

            generated_image = postprocess_image(output)
            print("🖼 Postprocessing complete.")

            # Save original and generated images
            user_folder = os.path.join("static", "outputs", f"user_{session['user_id']}")
            os.makedirs(user_folder, exist_ok=True)

            original_filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            original_path = os.path.join(user_folder, f"{timestamp}_{original_filename}")
            generated_path = os.path.join(user_folder, f"gen_{timestamp}_{original_filename}")

            image.save(original_path)
            generated_image.save(generated_path)
            print(f"✅ Saved original at {original_path}")
            print(f"✅ Saved generated at {generated_path}")

            # Insert into DB
            cursor.execute(
                "INSERT INTO generated_images (user_id, image_name, image_type) VALUES (%s, %s, %s)",
                (session['user_id'], f"gen_{timestamp}_{original_filename}", image_type)
            )

            cursor.execute(
                "INSERT INTO uploaded_images (user_id, image_type, filename, filepath, uploaded_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (session['user_id'], image_type, original_filename, original_path, datetime.now())
            )

            conn.commit()
            print("📦 Database updated")

            return redirect(url_for('result', filename=f"gen_{timestamp}_{original_filename}"))

        except Exception as e:
            import traceback
            print("❌ Upload error:", e)
            traceback.print_exc()
            flash("Error processing the image.")
            return redirect(url_for('upload'))

    return render_template("upload.html")

@app.route('/result/<filename>')
def result(filename):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user_folder = f"static/outputs/user_{user_id}"
    real_image_path = f"{user_folder}/{filename.replace('gen_', '')}"
    generated_image_path = f"{user_folder}/{filename}"

    return render_template("result.html",
                           real_image_path=real_image_path,
                           image_path=generated_image_path)

@app.route('/download/<image_name>')
def download_image(image_name):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_folder = f"static/outputs/user_{session['user_id']}"
    image_path = os.path.join(user_folder, image_name)

    if os.path.exists(image_path):
        return send_file(image_path, as_attachment=True)
    else:
        flash("Image not found.")
        return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)