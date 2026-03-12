# Medical-Image-Synthesis-Using-GAN
This project is a web-based application that generates synthetic medical images using a Conditional Generative Adversarial Network (CGAN). The system allows users to upload medical images such as Chest X-ray or MRI scans, and the trained GAN model generates synthetic medical images based on the selected image type.

The application is built using Flask for the backend, MySQL for database management, and TensorFlow/Keras for deep learning model integration. It also includes user authentication and role-based access control for Admin, Doctor, and Patient users.

The primary goal of the project is to demonstrate how Generative AI can be used for medical image augmentation, which can support research, training datasets, and medical imaging studies.

# Objectives
* To generate synthetic medical images using Conditional GAN architecture.
* To provide a web-based interface for uploading and generating images.
* To manage users using role-based authentication.
* To store uploaded and generated images in a structured database system.
* To demonstrate the application of Generative AI in healthcare imaging.

# Methodology
1. Data Preprocessing

* Medical images are resized and normalized.
* Data augmentation techniques are applied to increase dataset diversity.

2. Model Training

* A Conditional GAN (CGAN) model is trained using labeled medical images.
* The generator produces synthetic images using random noise and class labels.
* The discriminator evaluates whether images are real or generated.

3. Model Integration

* The trained generator model is saved and loaded into the Flask application.

4. Image Upload & Processing
* Users upload medical images through the web interface.
* Images are preprocessed before being passed to the GAN generator.

5. Image Generation

* The model generates synthetic medical images based on the selected image type.

6. Result Display

*Both the original and generated images are displayed to the user.
* Generated images can be downloaded.
