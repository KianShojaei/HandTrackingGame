"""
Hand-Controlled Drawing Ball Game -- Creative Edition
------------------------------------------------------
نسخهٔ خلاقانهٔ بازی: علاوه بر کشیدن خط با انگشت، حالا کل دستت یک پدال زندهٔ
فیزیکی است که توپ از رویش برخورد می‌کند، مشت‌کردن دست یک سپر موقت فعال
می‌کند، و جمع‌کردن سریع آیتم‌ها امتیاز کمبو می‌دهد.

مسیر فایل‌های تصویر دقیقاً مثل نسخهٔ اصلی حفظ شده است:
ball.png, bomb.png, powerup.png, life_powerup.png, bonus.png

مکانیزم‌های جدید نسبت به نسخهٔ قبلی:
  - دست به‌عنوان پدال: کل دست (نه فقط یک خط) توپ را پس می‌زند
  - مشت‌کردن دست = فعال‌سازی سپر موقت (بمب آسیب نمی‌زند)
  - کمبو امتیاز: جمع‌کردن سریع و پیاپیِ آیتم‌ها ضریب امتیاز را بالا می‌برد
  - توپ دوم در سطوح بالاتر برای چالش بیشتر
  - جلوه‌های بصری: ذرات، دنبالهٔ توپ، فلاش برخورد با بمب، حلقهٔ سپر

کلیدها:
  SPACE : شروع بازی / شروع دوباره بعد از باخت
  D     : شروع نقاشی با انگشت اشاره
  S     : توقف نقاشی
  C     : پاک کردن بوم (خط‌های کشیده‌شده)
  R     : تغییر رنگ نقاشی به قرمز
  P     : تغییر رنگ نقاشی به بنفش (رنگ پیش‌فرض)
  ESC   : خروج از بازی

حرکت دست:
  مشت کن  -> فعال‌سازی سپر موقت (چند ثانیه بعد از کول‌داون دوباره در دسترس است)
  کل دستت -> پدال زنده است؛ توپ از رویش پس می‌خورد
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
# تنظیمات کلی بازی
# ============================================================
class Config:
    WINDOW_NAME = "Hand Drawing Ball Game - Creative Edition"

    DEFAULT_DRAWING_COLOR = (255, 0, 255)  # بنفش (BGR)
    RED_DRAWING_COLOR = (0, 0, 255)
    LINE_THICKNESS = 5
    LINE_LIFETIME = 0.2

    BALL_RADIUS = 20
    BALL_BASE_SPEED = 5.0
    BALL_MAX_SPEED = 14.0
    BALL_SPEED_INCREMENT = 0.5
    SCORE_PER_LEVEL = 10

    BONUS_RADIUS = 10
    BONUS_SCORE = 1

    POWERUP_RADIUS = 15
    POWERUP_SCORE = 5

    LIFE_POWERUP_RADIUS = 15
    LIFE_POWERUP_INTERVAL = 10
    LIFE_POWERUP_DURATION = 5
    LIFE_POWERUP_BONUS = 10

    BOMB_RADIUS = 15
    BOMB_HIDDEN_DURATION = 5
    BOMB_RESPAWN_RANGE = 100

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

    # --- مکانیزم‌های خلاقانهٔ جدید ---
    HAND_HULL_COLOR = (255, 255, 0)      # آبی‌فیروزه‌ای (BGR)
    HAND_HULL_THICKNESS = 2
    HAND_PADDLE_MARGIN = 4
    HAND_PADDLE_COOLDOWN = 0.15
    HAND_PADDLE_SPEED_BOOST = 1.15

    COMBO_WINDOW = 2.0
    MAX_COMBO = 5

    SHIELD_DURATION = 3.0
    SHIELD_COOLDOWN = 8.0
    SHIELD_COLOR = (255, 255, 100)

    SECOND_BALL_LEVEL = 3
    MAX_BALLS = 2

    TRAIL_LENGTH = 8

    PARTICLE_COUNT_COLLECT = 12
    PARTICLE_COUNT_PADDLE = 10
    PARTICLE_LIFE = 12  # فریم

    BOMB_FLASH_DURATION = 0.15
    BOMB_FLASH_COLOR = (0, 0, 255)


class GameState:
    START = "start"
    PLAYING = "playing"
    GAME_OVER = "game_over"


# ============================================================
# توابع کمکی هندسی
# ============================================================
def point_segment_distance(p, a, b):
    """فاصلهٔ نقطهٔ p تا پاره‌خط ab (نه خط بی‌نهایت)."""
    p = np.array(p, dtype=np.float64)
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def load_rgba_image(path, size):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"فایل تصویر پیدا نشد: '{path}'.\n"
            f"لطفاً مطمئن شو این فایل در همان پوشه‌ای قرار دارد که اسکریپت را اجرا می‌کنی."
        )
    return Image.open(path).resize(size).convert("RGBA")


# ============================================================
# اشیای بازی (بونوس، پاورآپ، بمب و ...)
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


class Ball(GameObject):
    def __init__(self, image, radius):
        super().__init__(image, radius)
        self.velocity = np.array([Config.BALL_BASE_SPEED, Config.BALL_BASE_SPEED], dtype=np.float32)
        self.trail = deque(maxlen=Config.TRAIL_LENGTH)
        self.last_paddle_bounce_time = 0.0

    def reset(self, canvas_shape, offset=(0, 0)):
        base_x = int(np.clip(canvas_shape[1] // 4 + offset[0], self.radius, canvas_shape[1] - self.radius))
        base_y = int(np.clip(canvas_shape[0] // 4 + offset[1], self.radius, canvas_shape[0] - self.radius))
        self.position = np.array([base_x, base_y], dtype=np.int32)
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = (np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                          * Config.BALL_BASE_SPEED)
        self.trail.clear()

    def record_trail(self):
        self.trail.append(tuple(int(v) for v in self.position))

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

        self.position = (self.position.astype(np.float64) + normal_vector * self.radius * 0.5).astype(np.int32)


# ============================================================
# سیستم ذرات (برای جلوه‌های بصری)
# ============================================================
class Particle:
    def __init__(self, position, velocity, color, life):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.color = color
        self.life = life
        self.max_life = life

    def update(self):
        self.position += self.velocity
        self.velocity *= 0.9
        self.life -= 1

    @property
    def alive(self):
        return self.life > 0

    def draw(self, frame):
        t = self.life / self.max_life
        radius = max(1, int(5 * t))
        cv2.circle(frame, (int(self.position[0]), int(self.position[1])), radius, self.color, -1)


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def spawn_burst(self, center, color, count):
        for _ in range(count):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2, 6)
            velocity = (np.cos(angle) * speed, np.sin(angle) * speed)
            self.particles.append(Particle(center, velocity, color, Config.PARTICLE_LIFE))

    def update(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.alive]

    def draw(self, frame):
        for p in self.particles:
            p.draw(frame)

    def clear(self):
        self.particles.clear()


# ============================================================
# بوم نقاشی
# ============================================================
class DrawingCanvas:
    def __init__(self, shape):
        self.canvas = np.zeros(shape, dtype=np.uint8)
        self.color = Config.DEFAULT_DRAWING_COLOR
        self.points_queue = deque()

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
        best_edge = None
        best_dist = None
        for start_point, end_point, _ in self.points_queue:
            d = point_segment_distance(ball.position, start_point, end_point)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_edge = (start_point, end_point)
        if best_dist is not None and best_dist <= ball.radius:
            return best_edge
        return None


# ============================================================
# ردیابی دست با MediaPipe + تشخیص حرکت (gesture)
# ============================================================
class HandTracker:
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]

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
    def get_pixel_landmarks(hand_landmarks, width, height):
        return [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark]

    @staticmethod
    def convex_hull(landmarks_px):
        pts = np.array(landmarks_px, dtype=np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        return hull.reshape(-1, 2)

    @staticmethod
    def is_fist(landmarks_px):
        wrist = np.array(landmarks_px[0], dtype=np.float64)
        curled = 0
        for tip_idx, pip_idx in zip(HandTracker.FINGER_TIPS, HandTracker.FINGER_PIPS):
            tip = np.array(landmarks_px[tip_idx], dtype=np.float64)
            pip = np.array(landmarks_px[pip_idx], dtype=np.float64)
            if np.linalg.norm(tip - wrist) < np.linalg.norm(pip - wrist):
                curled += 1
        return curled >= 3

    def close(self):
        self.hands.close()


# ============================================================
# کلاس اصلی بازی
# ============================================================
class Game:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("وبکم پیدا نشد یا در دسترس نیست.")

        self.hand_tracker = HandTracker()
        self.images = self._load_images()

        self.canvas_shape = None
        self.drawing_canvas = None

        self.balls = []
        self.bonus = GameObject(self.images["bonus"], Config.BONUS_RADIUS)
        self.powerup = GameObject(self.images["powerup"], Config.POWERUP_RADIUS)
        self.life_powerup = GameObject(self.images["life_powerup"], Config.LIFE_POWERUP_RADIUS)
        self.bomb = GameObject(self.images["bomb"], Config.BOMB_RADIUS)

        self.state = GameState.START
        self.drawing = False
        self.prev_finger_pos = None
        self.current_hand_hull = None

        self.score = 0
        self.lives = Config.STARTING_LIVES

        self.life_powerup_visible = False
        self.life_powerup_timer = 0.0
        self.bomb_visible = True
        self.bomb_hit_time = None

        self.combo_multiplier = 1
        self.last_collect_time = 0.0

        self.shield_active_until = 0.0
        self.shield_ready_at = 0.0
        self.prev_fist_state = False

        self.particles = ParticleSystem()
        self.bomb_flash_until = 0.0

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
            print(f"[خطا] {e}")
            sys.exit(1)

    @property
    def shield_active(self):
        return time.time() < self.shield_active_until

    # -------------------- راه‌اندازی / ری‌استارت --------------------
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
        self.current_hand_hull = None

        self.drawing_canvas.clear()

        first_ball = self.balls[0] if self.balls else Ball(self.images["ball"], Config.BALL_RADIUS)
        first_ball.reset(self.canvas_shape)
        self.balls = [first_ball]

        self.bonus.randomize_position(self.canvas_shape)
        self.powerup.randomize_position(self.canvas_shape)

        self.bomb.randomize_position(self.canvas_shape)
        self.bomb_visible = True
        self.bomb_hit_time = None

        self.life_powerup_visible = False
        self.life_powerup_timer = time.time() + Config.LIFE_POWERUP_INTERVAL

        self.combo_multiplier = 1
        self.last_collect_time = 0.0

        self.shield_active_until = 0.0
        self.shield_ready_at = 0.0
        self.prev_fist_state = False

        self.particles.clear()
        self.bomb_flash_until = 0.0

        self._apply_difficulty()

    def _apply_difficulty(self):
        level = self.score // Config.SCORE_PER_LEVEL + 1
        speed = min(Config.BALL_BASE_SPEED + (level - 1) * Config.BALL_SPEED_INCREMENT, Config.BALL_MAX_SPEED)
        for ball in self.balls:
            ball.set_speed(speed)

        if level >= Config.SECOND_BALL_LEVEL and len(self.balls) < Config.MAX_BALLS:
            new_ball = Ball(self.images["ball"], Config.BALL_RADIUS)
            new_ball.reset(self.canvas_shape, offset=(self.canvas_shape[1] // 4, -self.canvas_shape[0] // 4))
            new_ball.set_speed(speed)
            self.balls.append(new_ball)
            self.particles.spawn_burst(new_ball.position, (0, 200, 255), Config.PARTICLE_COUNT_COLLECT)

        return speed

    # -------------------- ثبت امتیاز و کمبو --------------------
    def _register_collect(self, base_score):
        now = time.time()
        if now - self.last_collect_time <= Config.COMBO_WINDOW:
            self.combo_multiplier = min(self.combo_multiplier + 1, Config.MAX_COMBO)
        else:
            self.combo_multiplier = 1
        self.last_collect_time = now
        gained = base_score * self.combo_multiplier
        self.score += gained
        return gained

    # -------------------- پردازش دست، نقاشی و حرکت --------------------
    def _handle_gesture(self, landmarks_px):
        fist_now = HandTracker.is_fist(landmarks_px)
        now = time.time()
        if fist_now and not self.prev_fist_state and now >= self.shield_ready_at:
            self.shield_active_until = now + Config.SHIELD_DURATION
            self.shield_ready_at = now + Config.SHIELD_COOLDOWN
            if self.balls:
                self.particles.spawn_burst(self.balls[0].position, Config.SHIELD_COLOR, 16)
        self.prev_fist_state = fist_now

    def _handle_hand_tracking(self, rgb_image, bgr_image_for_drawing):
        results = self.hand_tracker.process(rgb_image)
        if not results.multi_hand_landmarks:
            self.prev_finger_pos = None
            self.current_hand_hull = None
            return

        hand_landmarks = results.multi_hand_landmarks[0]
        self.hand_tracker.draw_landmarks(bgr_image_for_drawing, hand_landmarks)

        h, w = rgb_image.shape[:2]
        landmarks_px = HandTracker.get_pixel_landmarks(hand_landmarks, w, h)
        self.current_hand_hull = HandTracker.convex_hull(landmarks_px)

        self._handle_gesture(landmarks_px)

        x, y = landmarks_px[8]
        if self.drawing:
            if self.prev_finger_pos is not None:
                self.drawing_canvas.add_line(self.prev_finger_pos, (x, y))
            self.prev_finger_pos = (x, y)
        else:
            self.prev_finger_pos = None

    # -------------------- فیزیک و برخوردها --------------------
    def _find_hand_paddle_edge(self, ball):
        hull = self.current_hand_hull
        if hull is None or len(hull) < 3:
            return None

        n = len(hull)
        best_edge = None
        best_dist = None
        for i in range(n):
            a = tuple(int(v) for v in hull[i])
            b = tuple(int(v) for v in hull[(i + 1) % n])
            d = point_segment_distance(ball.position, a, b)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_edge = (a, b)

        if best_edge is None:
            return None

        inside = cv2.pointPolygonTest(
            hull.reshape(-1, 1, 2).astype(np.float32),
            (float(ball.position[0]), float(ball.position[1])),
            False,
        ) >= 0

        if inside or best_dist <= ball.radius + Config.HAND_PADDLE_MARGIN:
            return best_edge
        return None

    def _update_ball_physics(self):
        now = time.time()
        for ball in self.balls:
            ball.record_trail()
            ball.move_and_bounce(self.canvas_shape)

            if self.drawing_canvas.has_collision(ball):
                segment = self.drawing_canvas.find_reflection_segment(ball)
                if segment is not None:
                    speed = self._apply_difficulty()
                    ball.reflect_off_line(segment[0], segment[1], speed)

            edge = self._find_hand_paddle_edge(ball)
            if edge is not None and now - ball.last_paddle_bounce_time > Config.HAND_PADDLE_COOLDOWN:
                speed = self._apply_difficulty() * Config.HAND_PADDLE_SPEED_BOOST
                ball.reflect_off_line(edge[0], edge[1], speed)
                ball.last_paddle_bounce_time = now
                self.particles.spawn_burst(ball.position, Config.HAND_HULL_COLOR, Config.PARTICLE_COUNT_PADDLE)

    def _update_collectibles(self):
        for ball in self.balls:
            if ball.collides_with(self.bonus):
                self._register_collect(Config.BONUS_SCORE)
                self.particles.spawn_burst(self.bonus.position, (0, 255, 255), Config.PARTICLE_COUNT_COLLECT)
                self.bonus.randomize_position(self.canvas_shape)
                self._apply_difficulty()

            if ball.collides_with(self.powerup):
                self._register_collect(Config.POWERUP_SCORE)
                self.particles.spawn_burst(self.powerup.position, (255, 0, 255), Config.PARTICLE_COUNT_COLLECT)
                self.powerup.randomize_position(self.canvas_shape)
                self._apply_difficulty()

            if self.life_powerup_visible and ball.collides_with(self.life_powerup):
                self.lives += Config.LIFE_POWERUP_BONUS
                self.life_powerup_visible = False
                self.particles.spawn_burst(self.life_powerup.position, (0, 255, 0), Config.PARTICLE_COUNT_COLLECT)

            if self.bomb_visible and ball.collides_with(self.bomb):
                self.bomb_visible = False
                self.bomb_hit_time = time.time()
                if self.shield_active:
                    self.particles.spawn_burst(self.bomb.position, Config.SHIELD_COLOR, Config.PARTICLE_COUNT_COLLECT)
                else:
                    self.lives -= 1
                    self.bomb_flash_until = time.time() + Config.BOMB_FLASH_DURATION
                    self.particles.spawn_burst(self.bomb.position, Config.BOMB_FLASH_COLOR, Config.PARTICLE_COUNT_COLLECT)
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

        if self.combo_multiplier > 1 and now - self.last_collect_time > Config.COMBO_WINDOW:
            self.combo_multiplier = 1

    # -------------------- رسم صحنه --------------------
    def _draw_ball_trails(self, frame):
        for ball in self.balls:
            trail_list = list(ball.trail)
            n = len(trail_list)
            for i, pos in enumerate(trail_list):
                ratio = (i + 1) / max(n, 1)
                alpha = ratio * 0.35
                radius = max(2, int(ball.radius * ratio * 0.6))
                overlay = frame.copy()
                cv2.circle(overlay, (int(pos[0]), int(pos[1])), radius, (0, 0, 255), -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_hand_hull(self, frame):
        hull = self.current_hand_hull
        if hull is None or len(hull) < 3:
            return
        pts = hull.reshape(-1, 1, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], Config.HAND_HULL_COLOR)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [pts], True, Config.HAND_HULL_COLOR, Config.HAND_HULL_THICKNESS)

    def _draw_shield(self, frame):
        if not self.shield_active:
            return
        pulse = int(3 * np.sin(time.time() * 10))
        for ball in self.balls:
            radius = ball.radius + 8 + pulse
            cv2.circle(frame, tuple(int(v) for v in ball.position), radius, Config.SHIELD_COLOR, 2)

    def _apply_bomb_flash(self, frame):
        if time.time() < self.bomb_flash_until:
            overlay = frame.copy()
            overlay[:] = Config.BOMB_FLASH_COLOR
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    def _draw_hud(self, frame):
        cv2.putText(frame, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Lives: {self.lives}", (self.canvas_shape[1] - 175, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        level = self.score // Config.SCORE_PER_LEVEL + 1
        cv2.putText(frame, f"Level: {level}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        if self.combo_multiplier > 1:
            text = f"Combo x{self.combo_multiplier}!"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            x = (self.canvas_shape[1] - size[0]) // 2
            cv2.putText(frame, text, (x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        now = time.time()
        if self.shield_active:
            remaining = self.shield_active_until - now
            status = f"Shield: ACTIVE {remaining:.1f}s"
            color = Config.SHIELD_COLOR
        elif now < self.shield_ready_at:
            remaining = self.shield_ready_at - now
            status = f"Shield cooldown: {remaining:.1f}s"
            color = (150, 150, 150)
        else:
            status = "Shield: make a fist!"
            color = (0, 255, 0)
        cv2.putText(frame, status, (10, self.canvas_shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def _compose_frame(self, camera_bgr):
        self.drawing_canvas.erase_expired()
        self.particles.update()

        overlay = Image.new("RGBA", (self.canvas_shape[1], self.canvas_shape[0]), (0, 0, 0, 0))
        for ball in self.balls:
            ball.paste_on(overlay)
        self.bonus.paste_on(overlay)
        self.powerup.paste_on(overlay)
        if self.bomb_visible:
            self.bomb.paste_on(overlay)
        if self.life_powerup_visible:
            self.life_powerup.paste_on(overlay)

        overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGBA2BGR)

        combined = cv2.addWeighted(camera_bgr, 0.5, self.drawing_canvas.canvas, 0.5, 0)
        combined = cv2.addWeighted(combined, 1.0, overlay_bgr, 1.0, 0)

        self._draw_ball_trails(combined)
        self._draw_hand_hull(combined)
        self.particles.draw(combined)
        self._draw_shield(combined)
        self._apply_bomb_flash(combined)
        self._draw_hud(combined)

        return combined

    def _draw_message_screen(self, frame, lines):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = frame.shape[0] // 2 - (len(lines) * 20)
        for i, (text, scale, color) in enumerate(lines):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
            x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, text, (x, y + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return frame

    # -------------------- حلقهٔ اصلی --------------------
    def run(self):
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("[هشدار] دریافت فریم از وبکم شکست خورد.")
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
                        ("Your hand is a paddle - the ball bounces off it!", 0.55, (0, 255, 255)),
                        ("Make a FIST for a temporary Shield", 0.55, (255, 255, 100)),
                        ("D: draw   S: stop   C: clear   ESC: quit", 0.5, (200, 200, 200)),
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
        print(f"[خطا] {e}")
        sys.exit(1)
    game.run()


if __name__ == "__main__":
    main()