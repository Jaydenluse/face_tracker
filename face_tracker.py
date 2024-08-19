import cv2
import mediapipe as mp
import numpy as np
import colorsys
import math

# Customizable variables
FADE_SPEED = .9
LINE_COLOR = (255, 255, 255)  # White color for the body outline
FACE_COLOR = (255, 255, 255)  # White color for face landmarks
HAND_COLOR = (255, 255, 255)  # White color for hand landmarks
GLOW_COLOR = (255, 255, 255)  # Color of the glow effect
LINE_THICKNESS = 1
GLOW_THICKNESS = 1
BLUR_AMOUNT = 1
EFFECT_OPACITY = 1
OVERLAY_SPEED = 0.1
OVERLAY_OPACITY = 0.1
FACE_POINT_SIZE = 1
HAND_POINT_SIZE = 2
HEAD_OFFSET = 1.3  # Offset for duplicate heads (as a fraction of face width)
ROTATION_SPEED = 5
SCALE_SPEED = 0.01
WAVE_AMPLITUDE = 5
WAVE_FREQUENCY = 0.1

# Initialize MediaPipe Solutions
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_faces=1, static_image_mode=True)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame from webcam")
    exit()

height, width = frame.shape[:2]
art_canvas = np.zeros((height, width, 3), dtype=np.uint8)
decay_mask = np.ones((height, width, 3), dtype=np.float32) * FADE_SPEED

# Define the connections for the body outline
POSE_CONNECTIONS = [
    # # Upper Body
    # (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    # (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    # (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    # (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    # (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),

    # # Torso
    # (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    # (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    # (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),

    # Lower Body
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),

    # Feet
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
    (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
    (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
]

def draw_body_outline(canvas, landmarks):
    h, w = canvas.shape[:2]
    for connection in POSE_CONNECTIONS:
        start_idx = connection[0].value
        end_idx = connection[1].value

        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))

        # Draw glowing effect
        cv2.line(canvas, start_point, end_point, GLOW_COLOR, GLOW_THICKNESS)
    
    # Apply blur for glow effect
    blurred = cv2.GaussianBlur(canvas, (BLUR_AMOUNT, BLUR_AMOUNT), 0)
    canvas = cv2.addWeighted(canvas, 0.4, blurred, 0.6, 0)

    # Draw sharp lines over the glow
    for connection in POSE_CONNECTIONS:
        start_idx = connection[0].value
        end_idx = connection[1].value

        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))

        cv2.line(canvas, start_point, end_point, LINE_COLOR, LINE_THICKNESS)

    return canvas

def create_wave_effect(x, y, frame_count):
    wave_x = x + WAVE_AMPLITUDE * math.sin(y * WAVE_FREQUENCY + frame_count * 0.1)
    wave_y = y + WAVE_AMPLITUDE * math.cos(x * WAVE_FREQUENCY + frame_count * 0.1)
    return int(wave_x), int(wave_y)

def draw_face_landmarks(canvas, face_landmarks, offset_x=0, frame_count=0):
    h, w = canvas.shape[:2]
    
    # Create a mask for the face landmarks
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate the center of the face
    face_x = np.mean([lm.x for lm in face_landmarks.landmark])
    face_y = np.mean([lm.y for lm in face_landmarks.landmark])
    center = (int((face_x + offset_x) * w), int(face_y * h))
    
    # Calculate rotation and scale
    rotation = frame_count * ROTATION_SPEED
    scale = 1 + 0.2 * math.sin(frame_count * SCALE_SPEED)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, rotation, scale)
    
    # Draw all 468 face landmarks
    for landmark in face_landmarks.landmark:
        x, y = int((landmark.x + offset_x) * w), int(landmark.y * h)
        
        # Apply wave effect
        x, y = create_wave_effect(x, y, frame_count)
        
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), FACE_POINT_SIZE, 255, -1)
    
    # Draw connections between landmarks
    connections = mp_face_mesh.FACEMESH_TESSELATION
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_point = face_landmarks.landmark[start_idx]
        end_point = face_landmarks.landmark[end_idx]
        
        start_x, start_y = int((start_point.x + offset_x) * w), int(start_point.y * h)
        end_x, end_y = int((end_point.x + offset_x) * w), int(end_point.y * h)
        
        # Apply wave effect
        start_x, start_y = create_wave_effect(start_x, start_y, frame_count)
        end_x, end_y = create_wave_effect(end_x, end_y, frame_count)
        
        if (0 <= start_x < w and 0 <= start_y < h and
            0 <= end_x < w and 0 <= end_y < h):
            cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, 1)
    
    # Apply rotation and scaling
    transformed_mask = cv2.warpAffine(mask, M, (w, h))
    
    # Apply color to the mask
    colored_mask = cv2.cvtColor(transformed_mask, cv2.COLOR_GRAY2BGR)
    hue = (frame_count * OVERLAY_SPEED) % 1.0
    rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
    colored_mask[transformed_mask != 0] = rgb
    
    # Blend the colored mask with the canvas
    canvas = cv2.addWeighted(canvas, 1, colored_mask, 0.7, 0)
    
    return canvas

def draw_hand_landmarks(canvas, hand_landmarks):
    h, w = canvas.shape[:2]
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(canvas, (x, y), HAND_POINT_SIZE, HAND_COLOR, -1)
    
    # Draw connections between hand landmarks
    mp_drawing.draw_landmarks(
        canvas,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=1)
    )
    return canvas

def generate_color_overlay(frame_count):
    hue = (frame_count * OVERLAY_SPEED) % 1.0
    rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
    overlay = np.full((height, width, 3), rgb, dtype=np.uint8)
    return overlay

def get_face_width(face_landmarks):
    left = min(landmark.x for landmark in face_landmarks.landmark)
    right = max(landmark.x for landmark in face_landmarks.landmark)
    return right - left

frame_count = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to grab frame from webcam")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)

    # Apply decay to the entire canvas
    art_canvas = cv2.multiply(art_canvas.astype(np.float32), decay_mask).astype(np.uint8)

    if pose_results.pose_landmarks:
        art_canvas = draw_body_outline(art_canvas, pose_results.pose_landmarks.landmark)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            face_width = get_face_width(face_landmarks)
            offset = face_width * HEAD_OFFSET


            art_canvas = draw_face_landmarks(art_canvas, face_landmarks, -offset, frame_count)
            art_canvas = draw_face_landmarks(art_canvas, face_landmarks, 0, frame_count)
            art_canvas = draw_face_landmarks(art_canvas, face_landmarks, offset, frame_count)


    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            art_canvas = draw_hand_landmarks(art_canvas, hand_landmarks)

    # Generate and apply color overlay
    color_overlay = generate_color_overlay(frame_count)
    art_canvas = cv2.addWeighted(art_canvas, 1 - OVERLAY_OPACITY, color_overlay, OVERLAY_OPACITY, 0)

    # Blend the art canvas with the original image
    blended_image = cv2.addWeighted(image, 1 - EFFECT_OPACITY, art_canvas, EFFECT_OPACITY, 0)

    cv2.imshow('Psychedelic Body and Triple Detailed Face Mesh', blended_image)

    frame_count += 1

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()