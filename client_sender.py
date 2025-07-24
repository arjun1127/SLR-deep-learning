import cv2
import requests
import time

SERVER_URL = "http://127.0.0.1:5000/predict"

cap = cv2.VideoCapture(0)
last_sent = 0
send_interval = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Sending frame to server...", frame)

    # Send every 2 seconds
    if time.time() - last_sent > send_interval:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(
            SERVER_URL,
            files={"frame": img_encoded.tobytes()}
        )
        try:
            prediction = response.json().get("prediction", "None")
            print("Prediction from server:", prediction)
        except Exception as e:
            print("Error:", e)

        last_sent = time.time()

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
