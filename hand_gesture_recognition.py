import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt

model_path = 'hand_recognition_model.h5'
model = load_model(model_path)

alphabet_mapper = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0
    return img_normalized

def predict_hand_gesture(img_processed):
    img_expanded = np.expand_dims(img_processed, axis=(0, -1))
    output = model.predict(img_expanded)
    predicted_index = np.argmax(output)
    return predicted_index

cap = cv2.VideoCapture(0)

last_predicted_index = None
confidence_threshold = 0.8

plt.ion()  # Activer le mode interactif pour matplotlib

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    x, y, w, h = 800, 100, 300, 300
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    roi = frame[y:y + h, x:x + w]
    img_processed = process_image(roi)
    predicted_index = predict_hand_gesture(img_processed)
    predicted_letter = alphabet_mapper[predicted_index]

    cv2.putText(frame, f"Prediction: {predicted_letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Conversion en RGB et affichage avec matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.pause(0.05)
    plt.draw()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
plt.close()  # Fermer la fenêtre matplotlib
