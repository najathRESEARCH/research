import cv2
import os
import pandas as pd

# Initialize the video capture object(cam is open)
cap = cv2.VideoCapture(0)

# Load the Haar cascade classifier for face detection(this is pre trained xml file for human face)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the name, mobile number, and payment details from the user
name = input("Enter the name: ")
mobile = input("Enter the mobile number: ")
payment = input("Enter the payment details: ")

# Create an empty DataFrame to store the details
df = pd.DataFrame(columns=['Name', 'Mobile', 'Payment'])

# Create a directory to save the images
if not os.path.exists('facedataset'):
    os.makedirs('facedataset')

count = 0
while count < 200:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, f"Name: {name}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Mobile: {mobile}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Payment: {payment}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Count: {count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Press 'c' to capture the current frame
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        # Create a new DataFrame with the details
        new_df = pd.DataFrame({'Name': [name], 'Mobile': [mobile], 'Payment': [payment]})

        # Concatenate the new DataFrame with the existing one
        df = pd.concat([df, new_df], ignore_index=True)

        # Save the current frame as an image
        cv2.imwrite(f'facedataset/{name}_{count}.jpg', gray[y:y + h, x:x + w])
        count += 1

cap.release()
cv2.destroyAllWindows()
df.to_csv('facedataset.csv', index=False)





