"""Hand-controlled drawing ball game.

This is the structured 2026 revision of the original prototype. The core
interaction remains the same: the index finger creates short-lived barriers
that the ball reacts to. The code is organized into configuration, tracking,
game objects, physics, and rendering so each part has a clear responsibility.
"""

import os
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


# ============================================================
# Game configuration
# ============================================================
class Config:
    WINDOW_NAME = "Hand Drawing Ball Game"

    DEFAULT_DRAWING_COLOR = (255, 0, 255)  # Purple (BGR)
    RED_DRAWING_COLOR = (0, 0, 255)
    LINE_THICKNESS = 5
    LINE_LIFETIME = 0.2  # Seconds before a stroke expires

    BALL_RADIUS = 20
    BALL_BASE_SPEED = 5.0
    BALL_MAX_SPEED = 14.0
    BALL_SPEED_INCREMENT = 0.5   # Additional ball speed per difficulty level
    SCORE_PER_LEVEL = 10         # Score required to increase the difficulty level

    BONUS_RADIUS = 10
    BONUS_SCORE = 1

    POWERUP_RADIUS = 15
    POWERUP_SCORE = 5

    LIFE_POWERUP_RADIUS = 15
    LIFE_POWERUP_INTERVAL = 10   # Interval between life power-up appearances
    LIFE_POWERUP_DURATION = 5    # How long the life power-up remains visible
    LIFE_POWERUP_BONUS = 10      # Lives added when collected

    BOMB_RADIUS = 15
    BOMB_HIDDEN_DURATION = 5     # Hidden duration after a bomb collision
    BOMB_RESPAWN_RANGE = 100     # Maximum bomb respawn distance from the bonus

    STARTING_LIVES = 5

    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5

    IMAGE_PATHS = {
        "ball": "ball.png",
        "bomb": "bomb.png",
        "powerup": "powerup.png",
        "life_powerup": "life_powerup.png",
        "bonus": "bonus.png",
    }


class GameState:
    START = "start"
    PLAYING = "playing"
    GAME_OVER = "game_over"


# ============================================================
# Image loading with clear error messages
# ============================================================
def load_rgba_image(path, size):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Image file not found: '{path}'.\n"
            f"Make sure the file is located in the same directory as this script."
        )
    return Image.open(path).resize(size).convert("RGBA")


# ============================================================
# Shared game-object model for collectibles and hazards
# ============================================================
class GameObject:
    def __init__(self, image, radius):
        self.image = image
        self.radius = radius
        self.position = np.array([radius, radius], dtype=np.int32)

    def randomize_position(self, canvas_shape):
        h, w = canvas_shape[0], canvas_shape[1]
        max_x = max(self.radius + 1, w - self.radius)
        max_y = max(self.radius + 1, h - self.radius)
        x = np.random.randint(self.radius, max_x)
        y = np.random.randint(self.radius, max_y)
        self.position = np.array([x, y], dtype=np.int32)

    def keep_within_bounds(self, canvas_shape):
        self.position[0] = np.clip(self.position[0], self.radius, canvas_shape[1] - self.radius)
        self.position[1] = np.clip(self.position[1], self.radius, canvas_shape[0] - self.radius)

    def paste_on(self, layer):
        x = int(self.position[0] - self.radius)
        y = int(self.position[1] - self.radius)
        layer.paste(self.image, (x, y), self.image)

    def distance_to(self, other):
        return float(np.linalg.norm(self.position.astype(np.float64) - other.position.astype(np.float64)))

    def collides_with(self, other):
        return self.distance_to(other) < (self.radius + other.radius)


# ============================================================
# Ball physics
# ============================================================
class Ball(GameObject):
    def __init__(self, image, radius):
        super().__init__(image, radius)
        self.velocity = np.array([Config.BALL_BASE_SPEED, Config.BALL_BASE_SPEED], dtype=np.float32)

    def reset(self, canvas_shape):
        self.position = np.array([canvas_shape[1] // 4, canvas_shape[0] // 4], dtype=np.int32)
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = (np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                          * Config.BALL_BASE_SPEED)

    def set_speed(self, speed):
        norm = float(np.linalg.norm(self.velocity))
        if norm > 0:
            self.velocity = (self.velocity / norm * speed).astype(np.float32)

    def move_and_bounce(self, canvas_shape):
        self.position = self.position + self.velocity.astype(np.int32)
        if self.position[0] - self.radius < 0 or self.position[0] + self.radius > canvas_shape[1]:
            self.velocity[0] = -self.velocity[0]
        if self.position[1] - self.radius < 0 or self.position[1] + self.radius > canvas_shape[0]:
            self.velocity[1] = -self.velocity[1]
        self.keep_within_bounds(canvas_shape)

    def reflect_off_line(self, line_start, line_end, speed):
        line_vector = np.array(line_end, dtype=np.float64) - np.array(line_start, dtype=np.float64)
        norm = np.linalg.norm(line_vector)
        if norm == 0:
            return
        line_vector /= norm
        normal_vector = np.array([-line_vector[1], line_vector[0]])

        velocity_projection = np.dot(self.velocity, normal_vector)
        new_velocity = self.velocity - 2 * velocity_projection * normal_vector
        v_norm = np.linalg.norm(new_velocity)
        if v_norm > 0:
            new_velocity = new_velocity / v_norm * speed
        self.velocity = new_velocity.astype(np.float32)

        # Move the ball slightly away from the barrier to reduce repeated collisions.
        self.position = (self.position.astype(np.float64) + normal_vector * self.radius * 0.5).astype(np.int32)


# ============================================================
# Drawing canvas: stroke storage, expiry, and collision detection
# ============================================================
class DrawingCanvas:
    def __init__(self, shape):
        self.canvas = np.zeros(shape, dtype=np.uint8)
        self.color = Config.DEFAULT_DRAWING_COLOR
        self.points_queue = deque()

    def rebuild(self, shape):
        self.canvas = np.zeros(shape, dtype=np.uint8)
        self.points_queue.clear()

    def add_line(self, p1, p2):
        cv2.line(self.canvas, p1, p2, self.color, Config.LINE_THICKNESS)
        self.points_queue.append((p1, p2, time.time()))

    def erase_expired(self):
        now = time.time()
        while self.points_queue and now - self.points_queue[0][2] > Config.LINE_LIFETIME:
            p1, p2, _ = self.points_queue.popleft()
            cv2.line(self.canvas, p1, p2, (0, 0, 0), Config.LINE_THICKNESS)

    def clear(self):
        self.canvas[:] = 0
        self.points_queue.clear()

    def has_collision(self, ball):
        mask = np.zeros_like(self.canvas, dtype=np.uint8)
        cv2.circle(mask, tuple(int(v) for v in ball.position), ball.radius, (255, 255, 255), -1)
        return np.any(cv2.bitwise_and(self.canvas, mask))

    def find_reflection_segment(self, ball):
        for start_point, end_point, _ in self.points_queue:
            line_length = cv2.norm(np.array(end_point) - np.array(start_point))
            if line_length == 0:
                continue
            distance = abs(np.cross(np.array(end_point) - np.array(start_point),
                                     np.array(start_point) - ball.position)) / line_length
            if distance <= ball.radius:
                return start_point, end_point
        return None


# ============================================================
# Hand tracking with MediaPipe
# ============================================================
class HandTracker:
    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self.hands = self._mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
        )
        self._mp_drawing = mp.solutions.drawing_utils

    def process(self, rgb_image):
        return self.hands.process(rgb_image)

    def draw_landmarks(self, bgr_image, hand_landmarks):
        self._mp_drawing.draw_landmarks(bgr_image, hand_landmarks, self._mp_hands.HAND_CONNECTIONS)

    @staticmethod
    def index_fingertip_xy(hand_landmarks, width, height):
        tip = hand_landmarks.landmark[8]
        return int(tip.x * width), int(tip.y * height)

    def close(self):
        self.hands.close()


# ============================================================
# Main game controller
# ============================================================
class Game:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam was not found or is unavailable.")

        self.hand_tracker = HandTracker()
        self.images = self._load_images()

        self.canvas_shape = None  # Initialized after the first frame
        self.drawing_canvas = None

        self.ball = Ball(self.images["ball"], Config.BALL_RADIUS)
        self.bonus = GameObject(self.images["bonus"], Config.BONUS_RADIUS)
        self.powerup = GameObject(self.images["powerup"], Config.POWERUP_RADIUS)
        self.life_powerup = GameObject(self.images["life_powerup"], Config.LIFE_POWERUP_RADIUS)
        self.bomb = GameObject(self.images["bomb"], Config.BOMB_RADIUS)

        self.state = GameState.START
        self.drawing = False
        self.prev_finger_pos = None

        self.score = 0
        self.lives = Config.STARTING_LIVES

        self.life_powerup_visible = False
        self.life_powerup_timer = 0.0
        self.bomb_visible = True
        self.bomb_hit_time = None
        self.bomb_timer = 0.0

        cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    @staticmethod
    def _load_images():
        try:
            return {
                "ball": load_rgba_image(Config.IMAGE_PATHS["ball"], (2 * Config.BALL_RADIUS,) * 2),
                "bomb": load_rgba_image(Config.IMAGE_PATHS["bomb"], (2 * Config.BOMB_RADIUS,) * 2),
                "powerup": load_rgba_image(Config.IMAGE_PATHS["powerup"], (2 * Config.POWERUP_RADIUS,) * 2),
                "life_powerup": load_rgba_image(
                    Config.IMAGE_PATHS["life_powerup"], (2 * Config.LIFE_POWERUP_RADIUS,) * 2
                ),
                "bonus": load_rgba_image(Config.IMAGE_PATHS["bonus"], (2 * Config.BONUS_RADIUS,) * 2),
            }
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    # ------------------------------------------------------------
    def _ensure_canvas(self, frame_shape):
        shape = (frame_shape[0], frame_shape[1], 3)
        if self.canvas_shape != shape:
            self.canvas_shape = shape
            self.drawing_canvas = DrawingCanvas(shape)
            self._start_new_round()

    def _start_new_round(self):
        self.score = 0
        self.lives = Config.STARTING_LIVES
        self.drawing = False
        self.prev_finger_pos = None

        self.drawing_canvas.clear()

        self.ball.reset(self.canvas_shape)

        self.bonus.randomize_position(self.canvas_shape)
        self.powerup.randomize_position(self.canvas_shape)

        self.bomb.randomize_position(self.canvas_shape)
        self.bomb_visible = True
        self.bomb_hit_time = None
        self.bomb_timer = time.time() + Config.BOMB_HIDDEN_DURATION

        self.life_powerup_visible = False
        self.life_powerup_timer = time.time() + Config.LIFE_POWERUP_INTERVAL

        self._apply_difficulty()

    def _apply_difficulty(self):
        level = self.score // Config.SCORE_PER_LEVEL
        speed = min(Config.BALL_BASE_SPEED + level * Config.BALL_SPEED_INCREMENT, Config.BALL_MAX_SPEED)
        self.ball.set_speed(speed)
        return speed

    # ------------------------------------------------------------
    def _handle_hand_tracking(self, rgb_image, bgr_image_for_drawing):
        results = self.hand_tracker.process(rgb_image)
        if not results.multi_hand_landmarks:
            self.prev_finger_pos = None
            return

        hand_landmarks = results.multi_hand_landmarks[0]
        self.hand_tracker.draw_landmarks(bgr_image_for_drawing, hand_landmarks)

        h, w = rgb_image.shape[:2]
        x, y = self.hand_tracker.index_fingertip_xy(hand_landmarks, w, h)

        if self.drawing:
            if self.prev_finger_pos is not None:
                self.drawing_canvas.add_line(self.prev_finger_pos, (x, y))
            self.prev_finger_pos = (x, y)
        else:
            self.prev_finger_pos = None

    # ------------------------------------------------------------
    def _update_ball_physics(self):
        self.ball.move_and_bounce(self.canvas_shape)

        if self.drawing_canvas.has_collision(self.ball):
            segment = self.drawing_canvas.find_reflection_segment(self.ball)
            if segment is not None:
                speed = self._apply_difficulty()
                self.ball.reflect_off_line(segment[0], segment[1], speed)

    def _update_collectibles(self):
        if self.ball.collides_with(self.bonus):
            self.score += Config.BONUS_SCORE
            self.bonus.randomize_position(self.canvas_shape)
            self._apply_difficulty()

        if self.ball.collides_with(self.powerup):
            self.score += Config.POWERUP_SCORE
            self.powerup.randomize_position(self.canvas_shape)
            self._apply_difficulty()

        if self.life_powerup_visible and self.ball.collides_with(self.life_powerup):
            self.lives += Config.LIFE_POWERUP_BONUS
            self.life_powerup_visible = False

        # Previous version checked bomb collisions even while the bomb was hidden.
        # Collision is checked only while the bomb is actually visible.
        if self.bomb_visible and self.ball.collides_with(self.bomb):
            self.lives -= 1
            self.bomb_visible = False
            self.bomb_hit_time = time.time()
            if self.lives <= 0:
                self.state = GameState.GAME_OVER

    def _update_timers(self):
        now = time.time()

        if not self.bomb_visible and self.bomb_hit_time is not None:
            if now - self.bomb_hit_time >= Config.BOMB_HIDDEN_DURATION:
                self.bomb.position = self.bonus.position + np.random.randint(
                    -Config.BOMB_RESPAWN_RANGE, Config.BOMB_RESPAWN_RANGE, size=2
                )
                self.bomb.keep_within_bounds(self.canvas_shape)
                self.bomb_visible = True
                self.bomb_hit_time = None

        if not self.life_powerup_visible and now > self.life_powerup_timer:
            self.life_powerup.randomize_position(self.canvas_shape)
            self.life_powerup_visible = True
            self.life_powerup_timer = now + Config.LIFE_POWERUP_DURATION

        elif self.life_powerup_visible and now > self.life_powerup_timer:
            self.life_powerup_visible = False
            self.life_powerup_timer = now + Config.LIFE_POWERUP_INTERVAL

    # ------------------------------------------------------------
    def _compose_frame(self, camera_bgr):
        self.drawing_canvas.erase_expired()

        overlay = Image.new("RGBA", (self.canvas_shape[1], self.canvas_shape[0]), (0, 0, 0, 0))
        self.ball.paste_on(overlay)
        self.bonus.paste_on(overlay)
        self.powerup.paste_on(overlay)
        if self.bomb_visible:
            self.bomb.paste_on(overlay)
        if self.life_powerup_visible:
            self.life_powerup.paste_on(overlay)

        overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGBA2BGR)

        combined = cv2.addWeighted(camera_bgr, 0.5, self.drawing_canvas.canvas, 0.5, 0)
        combined = cv2.addWeighted(combined, 1.0, overlay_bgr, 1.0, 0)

        cv2.putText(combined, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, f"Lives: {self.lives}", (self.canvas_shape[1] - 175, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        level = self.score // Config.SCORE_PER_LEVEL + 1
        cv2.putText(combined, f"Level: {level}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        return combined

    def _draw_message_screen(self, frame, lines):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = frame.shape[0] // 2 - (len(lines) * 20)
        for i, (text, scale, color) in enumerate(lines):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
            x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, text, (x, y + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------
    def run(self):
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("[WARNING] Failed to read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                self._ensure_canvas(frame.shape)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.state == GameState.PLAYING:
                    self._handle_hand_tracking(rgb, frame)
                    self._update_ball_physics()
                    self._update_collectibles()
                    self._update_timers()
                    display_frame = self._compose_frame(frame)
                else:
                    display_frame = frame

                if self.state == GameState.START:
                    display_frame = self._draw_message_screen(display_frame, [
                        ("Hand Drawing Ball Game", 1.1, (255, 255, 255)),
                        ("Press SPACE to start", 0.8, (0, 255, 0)),
                        ("D: draw   S: stop   C: clear   ESC: quit", 0.55, (200, 200, 200)),
                    ])
                elif self.state == GameState.GAME_OVER:
                    display_frame = self._draw_message_screen(display_frame, [
                        ("Game Over", 1.2, (0, 0, 255)),
                        (f"Final Score: {self.score}", 0.9, (255, 255, 255)),
                        ("Press SPACE to play again, ESC to quit", 0.6, (200, 200, 200)),
                    ])

                cv2.imshow(Config.WINDOW_NAME, display_frame)

                if not self._handle_keys():
                    break
        finally:
            self.cap.release()
            self.hand_tracker.close()
            cv2.destroyAllWindows()

    def _handle_keys(self):
        key = cv2.waitKey(5) & 0xFF

        if key == 27:  # ESC
            return False

        if key == ord(' '):
            if self.state in (GameState.START, GameState.GAME_OVER):
                self.state = GameState.PLAYING
                self._start_new_round()

        elif self.state == GameState.PLAYING:
            if key == ord('d'):
                self.drawing = True
            elif key == ord('s'):
                self.drawing = False
            elif key == ord('c'):
                self.drawing_canvas.clear()
            elif key == ord('r'):
                self.drawing_canvas.color = Config.RED_DRAWING_COLOR
            elif key == ord('p'):
                self.drawing_canvas.color = Config.DEFAULT_DRAWING_COLOR

        return True


def main():
    try:
        game = Game()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    game.run()


if __name__ == "__main__":
    main()