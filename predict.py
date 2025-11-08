import tensorflow as tf
import numpy as np
import os
xximport random

# Load your trained model
model = tf.keras.models.load_model("caries_model.h5")

# Define class labels
class_names = ["NoEnamel_Caries", "EarlyStageEnamel_Caries", "AdvanceEnamel_Caries"]

# Pick a random test image
test_folder = r"C:\Users\yazhi\Downloads\Caries_Dataset\dataset\test"
selected_class = random.choice(os.listdir(test_folder))
img_name = random.choice(os.listdir(os.path.join(test_folder, selected_class)))
img_path = os.path.join(test_folder, selected_class, img_name)

print(f"Selected image: {img_path}")

# Load and preprocess image (resize to model input size)
img_size = (224, 224)  # must match training size
img = tf.keras.utils.load_img(img_path, target_size=img_size)
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict probabilities
prediction = model.predict(img_array)

# Step 1: Classify (argmax)
predicted_class_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_class_index]

print("\n🔹 Step 1: Classification Result")
print(f"Image classified as: {predicted_class}")

# Step 2: Show prediction probabilities
print("\n🔹 Step 2: Prediction Probabilities")
for i, prob in enumerate(prediction[0]):
    print(f"{class_names[i]}: {prob:.4f}")
