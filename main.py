import cv2
import numpy as np
import onnxruntime as ort
import math

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "yolov8n.onnx"
CONFIDENCE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45

# ==========================================
# COCO CLASS NAMES
# ==========================================

classes = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# ==========================================
# LOAD ONNX MODEL
# ==========================================

print("Loading YOLO model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("YOLO model loaded successfully.")

# ==========================================
# SIMPLE OBJECT TRACKER
# ==========================================

next_id = 0
tracked_objects = {}

def calculate_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )

def update_tracker(detections):

    global next_id
    global tracked_objects

    new_objects = {}

    for detection in detections:

        x1, y1, x2, y2, class_id, confidence = detection

        center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

        best_id = None
        best_distance = 80

        # Find nearest existing object
        for object_id, old_object in tracked_objects.items():

            old_center = old_object["center"]

            distance = calculate_distance(
                center,
                old_center
            )

            if distance < best_distance:

                best_distance = distance
                best_id = object_id

        # If no matching object found, create new ID
        if best_id is None:

            best_id = next_id
            next_id += 1

        new_objects[best_id] = {
            "center": center,
            "class_id": class_id
        }

        detection.append(best_id)

    tracked_objects = new_objects

    return detections


# ==========================================
# OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    print("Please check your camera permission.")
    exit()

print()
print("==========================================")
print("OBJECT DETECTION AND TRACKING STARTED")
print("==========================================")
print("Press Q to stop the program.")
print()

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:

        print("Could not read camera frame.")
        break

    frame_height, frame_width = frame.shape[:2]

    # --------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------

    image = cv2.resize(frame, (640, 640))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    # --------------------------------------
    # YOLO INFERENCE
    # --------------------------------------

    outputs = session.run(
        None,
        {input_name: image}
    )

    predictions = outputs[0]

    # YOLOv8 output:
    # (1, 84, 8400)

    predictions = np.squeeze(predictions)

    # Convert to:
    # (8400, 84)

    predictions = predictions.T

    boxes = []
    confidences = []
    class_ids = []

    # --------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------

    for detection in predictions:

        class_scores = detection[4:]

        class_id = np.argmax(class_scores)

        confidence = float(
            class_scores[class_id]
        )

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x_center = float(detection[0])
        y_center = float(detection[1])

        box_width = float(detection[2])
        box_height = float(detection[3])

        # Scale to original frame
        x_center *= frame_width / 640
        y_center *= frame_height / 640

        box_width *= frame_width / 640
        box_height *= frame_height / 640

        x1 = int(
            x_center - box_width / 2
        )

        y1 = int(
            y_center - box_height / 2
        )

        width = int(box_width)
        height = int(box_height)

        boxes.append([
            x1,
            y1,
            width,
            height
        ])

        confidences.append(confidence)

        class_ids.append(
            int(class_id)
        )

    # --------------------------------------
    # NON-MAXIMUM SUPPRESSION
    # --------------------------------------

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD
    )

    detections = []

    if len(indices) > 0:

        for index in indices:

            if isinstance(index, (list, tuple, np.ndarray)):

                index = int(index[0])

            else:

                index = int(index)

            x, y, width, height = boxes[index]

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(
                frame_width,
                x + width
            )

            y2 = min(
                frame_height,
                y + height
            )

            detections.append([
                x1,
                y1,
                x2,
                y2,
                class_ids[index],
                confidences[index]
            ])

    # --------------------------------------
    # TRACK OBJECTS
    # --------------------------------------

    tracked = update_tracker(
        detections
    )

    # --------------------------------------
    # DRAW RESULTS
    # --------------------------------------

    for detection in tracked:

        x1, y1, x2, y2, class_id, confidence, object_id = detection

        label = classes[class_id]

        text = (
            f"{label} "
            f"ID:{object_id} "
            f"{confidence:.2f}"
        )

        # Bounding box
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        # Label background
        cv2.rectangle(
            frame,
            (int(x1), int(y1) - 30),
            (int(x1) + 180, int(y1)),
            (0, 255, 0),
            -1
        )

        # Label text
        cv2.putText(
            frame,
            text,
            (int(x1) + 5, int(y1) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )

    # --------------------------------------
    # DISPLAY TITLE
    # --------------------------------------

    cv2.putText(
        frame,
        "CodeAlpha - Object Detection and Tracking",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------
    # SHOW WINDOW
    # --------------------------------------

    cv2.imshow(
        "CodeAlpha Task 4",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# ==========================================
# CLOSE EVERYTHING
# ==========================================

cap.release()

cv2.destroyAllWindows()

print()
print("==========================================")
print("OBJECT DETECTION AND TRACKING FINISHED")
print("==========================================")
