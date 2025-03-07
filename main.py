from ultralytics import YOLO
import cv2
from openvino.runtime import Core
import  numpy as np

core = Core()

#face detect
face_model_xml = r"intel\face-detection-retail-0004\FP16-INT8\face-detection-retail-0004.xml"  # You'll need to download this from OpenVINO Model Zoo
try:
    face_model = core.read_model(face_model_xml)
    face_compiled_model = core.compile_model(face_model)
    face_output_layer = face_compiled_model.output(0)
except Exception as e:
    print(f"Error loading OpenVINO model: {e}")
    print("Please download the face-detection-retail-0004 model from OpenVINO Model Zoo")
    print("Alternative: You can use face-detection-adas-0001 or another compatible face detection model")
    exit()
    
# Load the OpenVINO age-gender recognition model
age_gender_model_xml = r"intel\age-gender-recognition-retail-0013\FP16-INT8\age-gender-recognition-retail-0013.xml"
try:
    age_gender_model = core.read_model(age_gender_model_xml)
    age_gender_compiled_model = core.compile_model(age_gender_model)
    # The model has two outputs: age and gender
    age_output = age_gender_compiled_model.output("age_conv3")
    gender_output = age_gender_compiled_model.output("prob")
except Exception as e:
    print(f"Error loading OpenVINO age-gender model: {e}")
    print("Please ensure age-gender-recognition-retail-0013 model is downloaded correctly")
    exit()
    
# Load the OpenVINO emotion recognition model
emotion_model_xml = r"intel\emotions-recognition-retail-0003\FP16-INT8\emotions-recognition-retail-0003.xml"
try:
    emotion_model = core.read_model(emotion_model_xml)
    emotion_compiled_model = core.compile_model(emotion_model)
    emotion_output = emotion_compiled_model.output(0)
except Exception as e:
    print(f"Error loading OpenVINO emotion model: {e}")
    print("Please ensure emotions-recognition-retail-0003 model is downloaded correctly")
    exit()


# Load a pretrained YOLO model
model = YOLO("yolo11n_int8_openvino_model")

# Open the camera feed
camera_url = 0  # Use 0 for the default camera
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Error: Could not open camera feed.")
    exit()

# Frame skipping counter
frame_skip = 20  # Process every 5th frame
frame_count = 0

n, c, h, w = face_model.inputs[0].shape
n, c, h_age_gender, w_age_gender = age_gender_model.inputs[0].shape
n, c, h_emotion, w_emotion = emotion_model.inputs[0].shape  # Typically 64x64


# Define gender labels
gender_labels = ['Female', 'Male']

# Define emotion labels
emotion_labels = ['neutral', 'happy', 'sad', 'surprise', 'anger']

# Loop to process each frame from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Increment frame counter
    frame_count += 1

    # Skip frames if necessary
    if frame_count % frame_skip != 0:
        continue

    # Run the YOLO model on the frame
    results = model(frame)

    # Access the first result (since results is a list)
    result = results[0]

    # Loop through the detected objects and draw bounding boxes only for "person" (class ID 0)
    for box in result.boxes:
        class_id = int(box.cls)  # Get the class ID
        confidence = float(box.conf)  # Get the confidence score and convert it to a float

        # Check if the class is "person" and confidence is greater than 80%
        if class_id == 0 and confidence > 0.80:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to integers

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

            # Optionally, add a label
            label = f"Person {confidence:.2f}"  # Label with confidence score
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Crop the person region from the frame
            person_region = frame[y1:y2, x1:x2]
            
            if person_region.size == 0:  # Skip if the region is empty
                continue
            
            person_region_resized = cv2.resize(person_region, (w, h))
            # Change data layout from HWC to CHW
            person_region_input = np.expand_dims(person_region_resized.transpose(2, 0, 1), 0)


            faces = face_compiled_model([person_region_input])[face_output_layer]

            # Process detected faces
            for face_detection in faces[0][0]:
                # Each detection has format [image_id, label, conf, x_min, y_min, x_max, y_max]
                face_confidence = float(face_detection[2])
                
                if face_confidence > 0.6:  # Confidence threshold for face detection
                    # Scale detection from normalized coordinates to person region coordinates
                    fx_rel = int(face_detection[3] * person_region.shape[1])
                    fy_rel = int(face_detection[4] * person_region.shape[0])
                    fw_rel = int(face_detection[5] * person_region.shape[1]) - fx_rel
                    fh_rel = int(face_detection[6] * person_region.shape[0]) - fy_rel
                    
                    # Convert face coordinates back to the original frame coordinates
                    fx_abs = x1 + fx_rel
                    fy_abs = y1 + fy_rel
                    fw_abs = fw_rel
                    fh_abs = fh_rel
                    
                    # Extract face for age-gender recognition
                    face_image = frame[fy_abs:fy_abs+fh_abs, fx_abs:fx_abs+fw_abs]
                    if face_image.size == 0:  # Skip if the region is empty
                        continue
                    
                    # Preprocess face for age-gender model
                    face_image_ag = cv2.resize(face_image, (w_age_gender, h_age_gender))
                    face_image_ag = np.expand_dims(face_image_ag.transpose(2, 0, 1), 0)
                    
                    # Run age-gender inference
                    age_result = age_gender_compiled_model([face_image_ag])[age_output]
                    gender_result = age_gender_compiled_model([face_image_ag])[gender_output]
                    
                    # Process results
                    age = age_result[0][0][0][0] * 100  # The model output is normalized (0-1), scale to years
                    gender_id = np.argmax(gender_result[0])
                    gender = gender_labels[gender_id]
                    gender_confidence = gender_result[0][gender_id]
                    
                    # Preprocess face for emotion model
                    face_image_em = cv2.resize(face_image, (w_emotion, h_emotion))
                    face_image_em = np.expand_dims(face_image_em.transpose(2, 0, 1), 0)
                    
                    # Run emotion inference
                    emotion_result = emotion_compiled_model([face_image_em])[emotion_output]
                    
                    # Process results for emotion
                    emotion_id = np.argmax(emotion_result[0])
                    emotion = emotion_labels[emotion_id]
                    emotion_confidence = emotion_result[0][emotion_id]
                    
                    
                    # Draw the face bounding box
                    cv2.rectangle(frame, (fx_abs, fy_abs), (fx_abs + fw_abs, fy_abs + fh_abs), (255, 0, 0), 2)  # Blue box for face
                    
                    # Add face info labels
                    face_label = f"Face: {face_confidence:.2f}"
                    cv2.putText(frame, face_label, (fx_abs, fy_abs - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Add age label
                    age_label = f"Age: {age:.1f}"
                    cv2.putText(frame, age_label, (fx_abs, fy_abs - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Add gender label
                    cv2.putText(frame, gender, (fx_abs, fy_abs + fh_abs + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Add emotion label
                    cv2.putText(frame, emotion, (fx_abs, fy_abs + fh_abs + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Live Camera - Detected Persons and Faces", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()