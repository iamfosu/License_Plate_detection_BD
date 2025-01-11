from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from utils import get_car, write_csv, segment_characters, read_results_from_csv
from sort.sort import Sort

vcl_tracker = Sort()
results = {}
# Load models
coco_model = YOLO("yolov8m.pt")
LPD_model = YOLO(r"E:\license capture\Projects\LPD4kYOLOv8mALPR\YOLOv8mTM\detect\train\weights\best.pt")

# Initialize EasyOCR reader for Bangla
reader = easyocr.Reader(['bn'], gpu=False)

scale_factor = 3
# Load video
cap = cv2.VideoCapture(r"E:\license capture\Projects\LPD4kYOLOv8mALPR\video\3.mp4")
vehicles = [2, 3, 5, 7]


# # Allowed characters (Bangla characters, Bangla digits, and English alphabet)
# allowed_characters = set(
#     "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়১২৩৪৫৬৭৮৯০1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# Define the class letters
bengali_classes = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"

# Read frames
frame_num = -1
ret = True
while ret:
    frame_num += 1
    ret, frame = cap.read()

    if not ret:
        break

    results[frame_num] = {}
    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])
            # Draw bounding box for visualization
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    if len(detections_) == 0:
        print(f"Frame {frame_num}: No vehicles detected.")
        continue

    # Track vehicles
    track_ids = vcl_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = LPD_model(frame)[0]
    if len(license_plates.boxes.data) == 0:
        print(f"Frame {frame_num}: No license plates detected.")
        continue

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Assign plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        if car_id is None:
            continue

        # Crop license plate
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        # Preprocess license plate for OCR
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 21, 4)

        # Enhance contrast
        enhanced_image = cv2.equalizeHist(adaptive_thresh)

        # Edge detection
        edges = cv2.Canny(enhanced_image, 50, 150)

        # Save the preprocessed image (for debug purposes, can be removed later)
        cv2.imwrite(f'frame_{frame_num}_car_{car_id}_edges.jpg', edges)

        # Segment characters
        characters = segment_characters(edges)

        # Read license plate number using EasyOCR
        license_plate_text = ''
        for char_img in characters:
            result = reader.readtext(char_img)
            if result:
                license_plate_text += ''.join([text for _, text, _ in result])
                # filtered_text = ''.join([char for char in license_plate_text if char in allowed_characters])
                # license_plate_text += filtered_text

        if license_plate_text:
            results[frame_num][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'text': license_plate_text,
                    'bbox_score': score,
                    'text_score': None
                }
            }

        # Display each preprocessed image
        aa = cv2.resize(edges, None, fy=scale_factor, fx=scale_factor)
        cv2.imshow(f"edges_frame_{frame_num}_car_{car_id}", aa)

    # Display the frame with detected vehicles
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Write results to CSV
write_csv(results, './test.csv')

print('Detected text: ', license_plate_text)
