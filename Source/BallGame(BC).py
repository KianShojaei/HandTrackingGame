import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageOps
import time

# Initialize MediaPipe Hands.
# The prototype uses the default MediaPipe Hands configuration and keeps the
# tracking pipeline intentionally small: the index fingertip is the main input.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open the default webcam. The prototype assumes camera index 0, which keeps
# setup simple but also makes the original version dependent on local camera settings.
cap = cv2.VideoCapture(0)
drawing = False
prev_x, prev_y = None, None

# The drawing canvas is created after the first frame so its dimensions match
# the actual webcam resolution.
canvas = None

# Create a named window and set it to be resizable
cv2.namedWindow('Hand Drawing', cv2.WINDOW_NORMAL)

# Define the drawing color (BGR format)
drawing_color = (255, 0, 255)  # purple color

# Ball properties
ball_radius = 20
ball_color = (0, 0, 255)  # Red color
ball_position = np.array([100, 100], dtype=np.int32)
ball_velocity = np.array([5, 5], dtype=np.float32)

# Each segment keeps its creation time so the game can remove old strokes
# without having to rebuild the entire drawing history every frame.
points_queue = deque()

# Bonus properties
bonus_radius = 10
#bonus_color = (255, 0, 0)  # Blue color
bonus_position = np.random.randint(10, 470, size=2)
score = 0

# Power-up properties
powerup_radius = 15
powerup_position = np.random.randint(10, 470, size=2)

# Life-increasing power-up properties
life_powerup_radius = 15
life_powerup_position = np.random.randint(10, 470, size=2)
life_powerup_timer = time.time() + 10  # Life power-up appears every 10 seconds
life_powerup_visible = False
life_powerup_duration = 5  # Life power-up stays visible for 5 seconds

# Timer bomb properties
bomb_radius = 15
bomb_position = np.random.randint(10, 470, size=2)
bomb_timer = time.time() + 5  # Bomb appears for 5 seconds
bomb_hit_time = None  # Track the time when the bomb is hit

# Life properties
lives = 100

# Load images
ball_image = Image.open('ball.png').resize((2 * ball_radius, 2 * ball_radius)).convert("RGBA")
bomb_image = Image.open('bomb.png').resize((2 * bomb_radius, 2 * bomb_radius)).convert("RGBA")
powerup_image = Image.open('powerup.png').resize((2 * powerup_radius, 2 * powerup_radius)).convert("RGBA")
life_powerup_image = Image.open('life_powerup.png').resize((2 * life_powerup_radius, 2 * life_powerup_radius)).convert(
    "RGBA")
bonus_image = Image.open('bonus.png').resize((2 * bonus_radius, 2 * bonus_radius)).convert("RGBA")


def calculate_reflection(ball_velocity, line_start, line_end):
    # Reflect the velocity vector around the normal of the drawn line.
    # This gives the drawn stroke the role of a temporary physical barrier.
    line_vector = np.array(line_end) - np.array(line_start)
    line_vector = line_vector / np.linalg.norm(line_vector)
    normal_vector = np.array([-line_vector[1], line_vector[0]])
    velocity_projection = np.dot(ball_velocity, normal_vector)
    reflection_vector = ball_velocity - 2 * velocity_projection * normal_vector
    return reflection_vector


def keep_within_bounds(position, radius, canvas_shape):
    # Clamp an object's center so its full circular hit area stays on-screen.
    position[0] = np.clip(position[0], radius, canvas_shape[1] - radius)
    position[1] = np.clip(position[1], radius, canvas_shape[0] - radius)
    return position


while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Initialize canvas if not already done
    if canvas is None:
        canvas = np.zeros_like(image)

    # Flip the image horizontally to create a mirror effect
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    results = hands.process(image)
    # Convert the image to RGBA mode
    pil_image = pil_image.convert("RGBA")

    # Draw hand landmarks and track index finger
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = image.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if drawing:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), drawing_color, 5)
                    points_queue.append(((prev_x, prev_y), (x, y), time.time()))
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

    # Convert the RGB image back to BGR (we use np.array to remove the hand recognizers)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Advance the ball using its current velocity. Collision handling below
    # may change this velocity before the next frame is rendered.
    # Update ball position
    ball_position += ball_velocity.astype(np.int32)

    # Check for collision with canvas edges
    if ball_position[0] - ball_radius < 0 or ball_position[0] + ball_radius > canvas.shape[1]:
        ball_velocity[0] = -ball_velocity[0]
    if ball_position[1] - ball_radius < 0 or ball_position[1] + ball_radius > canvas.shape[0]:
        ball_velocity[1] = -ball_velocity[1]

    # First use a rasterized mask as a quick indication that the ball overlaps
    # the visible drawing. Detailed segment-level geometry is checked only
    # after this inexpensive test.
    # Check for collision with drawn lines
    mask = np.zeros_like(canvas, dtype=np.uint8)
    cv2.circle(mask, tuple(ball_position), ball_radius, (255, 255, 255), -1)
    collision = cv2.bitwise_and(canvas, mask)
    if np.any(collision):
        for start_point, end_point, _ in points_queue:
            line_length = cv2.norm(np.array(end_point) - np.array(start_point))
            if line_length > 0:
                distance = cv2.norm(np.cross(np.array(end_point) - np.array(start_point),
                                             np.array(start_point) - ball_position)) / line_length
                if distance <= ball_radius:
                    ball_velocity = calculate_reflection(ball_velocity, start_point, end_point)
                    ball_velocity = ball_velocity / np.linalg.norm(ball_velocity) * 7  # Maintain constant speed
                    # Move the ball slightly away from the line to avoid getting stuck
                    line_vector = np.array(end_point) - np.array(start_point)
                    line_vector = line_vector / np.linalg.norm(line_vector)
                    normal_vector = np.array([-line_vector[1], line_vector[0]])
                    ball_position += (normal_vector * (ball_radius - distance)).astype(np.int32)
                    break

    # Collectible objects use simple circle-to-circle distance checks.
    # Check for collision with bonus
    if np.linalg.norm(ball_position - bonus_position) < ball_radius + bonus_radius:
        score += 1
        bonus_position = np.random.randint(0, [canvas.shape[1], canvas.shape[0]])
        bonus_position = keep_within_bounds(bonus_position, bonus_radius, canvas.shape)

    # Check for collision with power-up
    if np.linalg.norm(ball_position - powerup_position) < ball_radius + powerup_radius:
        score += 5  # Example power-up effect: increase score by 5
        powerup_position = np.random.randint(0, [canvas.shape[1], canvas.shape[0]])
        powerup_position = keep_within_bounds(powerup_position, powerup_radius, canvas.shape)

    # Check for collision with life-increasing power-up
    if life_powerup_visible and np.linalg.norm(
            ball_position - life_powerup_position) < ball_radius + life_powerup_radius:
        lives += 10
        life_powerup_visible = False
        #life_powerup_position = np.random.randint(0, [canvas.shape[1], canvas.shape[0]])
        #life_powerup_position = keep_within_bounds(life_powerup_position, life_powerup_radius, canvas.shape)

    # Check for collision with bomb
    if np.linalg.norm(ball_position - bomb_position) < ball_radius + bomb_radius:
        lives -= 1
        bomb_hit_time = time.time()
        # bomb_position = np.random.randint(0, [canvas.shape[1], canvas.shape[0]])
        # bomb_position = keep_within_bounds(bomb_position, bomb_radius, canvas.shape)
        if lives == 0:
            print("Game Over!")
            break

    # Compose the game sprites separately from the camera and drawing canvas.
    # This keeps the visual layers independent before final compositing.
    # Create a separate layer for the ball
    ball_layer = np.zeros_like(canvas)
    cv2.circle(ball_layer, tuple(ball_position), ball_radius, ball_color, -1)

    # Create a separate layer for the ball, bonus, power-up, and bomb
    ball_layer = Image.new("RGBA", (canvas.shape[1], canvas.shape[0]), (0, 0, 0, 0))
    ball_layer.paste(ball_image, (ball_position[0] - ball_radius, ball_position[1] - ball_radius), ball_image)
    ball_layer.paste(bonus_image, (bonus_position[0] - bonus_radius, bonus_position[1] - bonus_radius), bonus_image)
    ball_layer.paste(powerup_image, (powerup_position[0] - powerup_radius, powerup_position[1] - powerup_radius), powerup_image)
    if bomb_hit_time is None or time.time() - bomb_hit_time >= 5:
        ball_layer.paste(bomb_image, (bomb_position[0] - bomb_radius, bomb_position[1] - bomb_radius), bomb_image)
    if life_powerup_visible:
        ball_layer.paste(life_powerup_image, (life_powerup_position[0] - life_powerup_radius, life_powerup_position[1] - life_powerup_radius), life_powerup_image)

    # Remove expired stroke segments from both the queue and the visible canvas.
    # The short lifetime is part of the original interaction design.
    # Erase lines after a delay
    current_time = time.time()
    while points_queue and current_time - points_queue[0][2] > 0.2:
        start_point, end_point, _ = points_queue.popleft()
        cv2.line(canvas, start_point, end_point, (0, 0, 0), 5)

    # Convert ball_layer to BGR format and ensure it matches the size and channels of the combined image
    ball_layer = cv2.cvtColor(np.array(ball_layer), cv2.COLOR_RGBA2BGR)

    # Update bomb position after timer expires
    if current_time > bomb_timer and (bomb_hit_time is None or time.time() - bomb_hit_time >= 5):
        bomb_position = bonus_position + np.random.randint(-100, 100, size=2)
        bomb_position = keep_within_bounds(bomb_position, bomb_radius, canvas.shape)
        bomb_timer = current_time + 5  # Reset timer for next bomb
        bomb_hit_time = None

    # Handle life power-up visibility
    if current_time > life_powerup_timer:
        life_powerup_visible = True
        life_powerup_position = np.random.randint(0, [canvas.shape[1], canvas.shape[0]])
        life_powerup_position = keep_within_bounds(life_powerup_position, life_powerup_radius, canvas.shape)
        life_powerup_timer = current_time + 10  # Reset timer for next life power-up

    # Hide life power-up after its duration
    if life_powerup_visible and current_time > life_powerup_timer - 10 + life_powerup_duration:
        life_powerup_visible = False

    # Final composition: camera view + semi-transparent drawing layer + sprites.
    # Combine the original image, canvas, and ball layer
    combined_image = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)
    combined_image = cv2.addWeighted(combined_image, 1, ball_layer, 1, 0)

    # Display the score
    cv2.putText(combined_image, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    # Display the lives
    cv2.putText(combined_image, f'Lives: {lives}', (canvas.shape[1] - 175, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Display the combined image
    cv2.imshow('Hand Drawing', combined_image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('d'):  # Press 'd' to start drawing
        drawing = True
    elif key == ord('s'):  # Press 's' to stop drawing
        drawing = False
    elif key == ord('r'):  # Press 'r' to change color to red
        drawing_color = (0, 0, 255)

cap.release()
cv2.destroyAllWindows()
