import cv2
import torch
import numpy as np
from cnn_model import CNN, load_model
import time
import torch.nn.functional as F
from cnn_model import CNN

# Charger le modèle entraîné
model_path = 'model.pth'
model = load_model(model_path)

alphabet_mapper = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


# Fonction pour convertir l'image en niveaux de gris et normaliser
def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0
    return img_normalized


# Fonction pour prédire le geste de la main à partir de l'image traitée
def predict_hand_gesture(img_processed):
    img_expanded = np.expand_dims(img_processed, axis=(0, 1))
    input_tensor = torch.tensor(img_expanded, dtype=torch.float32)
    output = model(input_tensor)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()


cap = cv2.VideoCapture(0)

last_predicted_index = None
confidence_threshold = 0.8
last_predicted_index = None
confidence_threshold = 0.8  # Vous pouvez ajuster cette valeur selon vos besoins

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    x, y, w, h = 800, 100, 300, 300
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    roi = frame[y:y + h, x:x + w]
    img = cv2.resize(roi, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_normalized = img / 255.0

    img_expanded = np.expand_dims(img_normalized, axis=(0, 1))
    img_expanded = torch.tensor(img_expanded, dtype=torch.float32)

    outputs = model(img_expanded)
    probabilities = F.softmax(outputs, dim=1)
    predicted_index = torch.argmax(outputs).item()
    predicted_probability = probabilities[0, predicted_index].item()

    if predicted_probability > confidence_threshold and predicted_index != last_predicted_index:
        predicted_letter = alphabet_mapper[predicted_index]
        print("Predicted letter:", predicted_letter)
        last_predicted_index = predicted_index

    cv2.putText(frame, f"Prediction: {alphabet_mapper[predicted_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(1)


cap.release()
cv2.destroyAllWindows()
