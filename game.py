import cv2
import time
import random
import mediapipe as mp
import math
import numpy as np

# MediaPipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Game settings
SLASH_LENGTH = 15
LIVES = 3
SPAWN_INTERVAL = 1.2

# Globals
fruits = []
splashes = []
explosions = []
score = 0
lives = LIVES
last_spawn_time = 0
last_slice_time = 0
slash_points = []
slash_color = (255, 255, 255)
game_over = False
combo_count = 0
combo_multiplier = 1
combo_time_limit = 2
combo_message = ""

w = 0
h = 0

# Load emoji images
apple_img = cv2.imread("apple.png", cv2.IMREAD_UNCHANGED)
banana_img = cv2.imread("banana.png", cv2.IMREAD_UNCHANGED)
watermelon_img = cv2.imread("watermelon.png", cv2.IMREAD_UNCHANGED)
bomb_img = cv2.imread("bomb.png", cv2.IMREAD_UNCHANGED)

fruit_map = {
    "apple": {"img": apple_img, "points": 100, "size": 40},
    "banana": {"img": banana_img, "points": 150, "size": 50},
    "watermelon": {"img": watermelon_img, "points": 200, "size": 70}
}

def overlay_image_alpha(img, img_overlay, pos, overlay_size):
    x, y = pos
    overlay = cv2.resize(img_overlay, overlay_size)
    b, g, r, a = cv2.split(overlay)
    mask = cv2.merge((a, a, a)) / 255.0
    overlay_rgb = cv2.merge((b, g, r)).astype(float)

    # Safeguard to ensure x, y, and sizes are within bounds
    h, w = img.shape[:2]
    overlay_h, overlay_w = overlay.shape[:2]

    if y < 0 or x < 0 or y + overlay_h > h or x + overlay_w > w:
        y_start = max(0, y)
        x_start = max(0, x)
        y_end = min(h, y + overlay_h)
        x_end = min(w, x + overlay_w)

        overlay_y_start = max(0, -y)
        overlay_x_start = max(0, -x)
        overlay_y_end = overlay_h - max(0, (y + overlay_h) - h)
        overlay_x_end = overlay_w - max(0, (x + overlay_w) - w)

        # Safeguard to skip processing if region is invalid
        if overlay_y_start >= overlay_y_end or overlay_x_start >= overlay_x_end:
            return

        # Partial overlay in case of out-of-boundary
        background = img[y_start:y_end, x_start:x_end].astype(float)
        overlay_part = overlay[
                       overlay_y_start:overlay_y_end,
                       overlay_x_start:overlay_x_end
                       ]

        # Handle the case where the overlay part might still be invalid
        if overlay_part.size == 0:
            return

        b, g, r, a = cv2.split(overlay_part)
        mask = cv2.merge((a, a, a)) / 255.0
        overlay_rgb = cv2.merge((b, g, r)).astype(float)

        # Blend the overlay only on the valid region
        blended = background * (1 - mask) + overlay_rgb * mask
        img[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)
    else:
        # No need for boundary handling if within bounds
        background = img[y:y + overlay_h, x:x + overlay_w].astype(float)
        blended = background * (1 - mask) + overlay_rgb * mask
        img[y:y + overlay_h, x:x + overlay_w] = blended.astype(np.uint8)

def spawn_fruit():
    x = random.randint(50, 600)
    is_bomb = random.random() < 0.2
    if is_bomb:
        fruits.append({
            "pos": [x, 600],
            "vel": [random.randint(-4, 4), -random.randint(13, 18)],
            "size": 40,
            "bomb": True,
            "img": bomb_img,
            "points": 0
        })
    else:
        fruit_type = random.choice(list(fruit_map.keys()))
        fruit = fruit_map[fruit_type]
        fruits.append({
            "pos": [x, 600],
            "vel": [random.randint(-3, 3), -random.randint(12, 15)],
            "size": fruit["size"],
            "bomb": False,
            "img": fruit["img"],
            "points": fruit["points"]
        })

def move_fruits():
    global lives
    for fruit in fruits[:]:
        fruit["vel"][1] += 0.5
        fruit["pos"][0] += fruit["vel"][0]
        fruit["pos"][1] += fruit["vel"][1]
        if fruit["pos"][1] > 700:
            fruits.remove(fruit)
            if not fruit["bomb"]:
                lives -= 1

def create_explosion_effect(pos):
    for _ in range(80):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(15, 30)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        explosions.append({
            "pos": list(pos),
            "vel": [vx, vy],
            "timer": random.randint(20, 40),
            "radius": random.randint(8, 15),
            "color": (random.randint(200, 255), random.randint(50, 150), 0)
        })

def draw_explosions(img):
    for explosion in explosions[:]:
        x, y = int(explosion["pos"][0]), int(explosion["pos"][1])
        cv2.circle(img, (x, y), explosion["radius"], explosion["color"], -1)
        explosion["pos"][0] += explosion["vel"][0]
        explosion["pos"][1] += explosion["vel"][1]
        explosion["vel"][0] *= 0.9
        explosion["vel"][1] *= 0.9
        explosion["timer"] -= 1
        if explosion["timer"] <= 0:
            explosions.remove(explosion)

def create_fruit_splash(pos, color):
    for _ in range(25):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 6)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        splashes.append({
            "pos": list(pos),
            "vel": [vx, vy],
            "timer": 20,
            "color": color,
            "radius": random.randint(3, 6)
        })

def draw_splashes(img):
    for splash in splashes[:]:
        x, y = int(splash["pos"][0]), int(splash["pos"][1])
        cv2.circle(img, (x, y), splash["radius"], splash["color"], -1)
        splash["pos"][0] += splash["vel"][0]
        splash["pos"][1] += splash["vel"][1]
        splash["vel"][1] += 0.4
        splash["timer"] -= 1
        if splash["timer"] <= 0:
            splashes.remove(splash)

def draw_fruits(img):
    for fruit in fruits:
        x, y = int(fruit["pos"][0]), int(fruit["pos"][1])
        size = fruit["size"] * 2
        overlay_image_alpha(img, fruit["img"], (x - size // 2, y - size // 2), (size, size))

def check_slices(index_pos):
    global score, lives, slash_color, combo_count, combo_multiplier, combo_message, last_slice_time, last_spawn_time
    combo_scored = False
    for fruit in fruits[:]:
        d = distance(index_pos, fruit["pos"])
        if d < fruit["size"]:
            slash_color = (255, 255, 255)
            if fruit["bomb"]:
                lives -= 1
                create_explosion_effect(tuple(map(int, fruit["pos"])))
                fruits.clear()
                splashes.clear()
                last_spawn_time = time.time() + 2  # delay next spawn
                return
            else:
                create_fruit_splash(fruit["pos"], (0, 0, 255))
                score += fruit["points"] * combo_multiplier
                fruits.remove(fruit)
                combo_scored = True
    if combo_scored:
        if time.time() - last_slice_time <= combo_time_limit:
            combo_count += 1
            combo_multiplier = 1 + (combo_count // 2)
            combo_message = f"{combo_multiplier}x Combo!"
        else:
            combo_count = 1
            combo_multiplier = 1
        last_slice_time = time.time()

def distance(a, b):
    return int(math.hypot(a[0] - b[0], a[1] - b[1]))

def draw_ui(img, fps):
    cv2.putText(img, f"Score: {score}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(img, f"Lives: {lives}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f"FPS: {fps}", (w - 120, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if combo_message:
        cv2.putText(img, combo_message, (w // 3, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

def draw_slash(img, index_pos):
    global slash_points
    slash_points.append(index_pos)
    if len(slash_points) > SLASH_LENGTH:
        slash_points.pop(0)
    if len(slash_points) > 1:
        pts = np.array(slash_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, slash_color, 15)

# Main game loop
cap = cv2.VideoCapture(0)
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    img = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    index_pos = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark[8]
        index_pos = (int(lm.x * w), int(lm.y * h))
        cv2.circle(img, index_pos, 20, slash_color, -1)
        draw_slash(img, index_pos)
        check_slices(index_pos)

    if not game_over:
        move_fruits()
        draw_fruits(img)
        draw_splashes(img)
        draw_explosions(img)

        if time.time() > last_spawn_time:
            spawn_fruit()
            last_spawn_time = time.time() + SPAWN_INTERVAL

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if curr_time != prev_time else 0
    prev_time = curr_time

    if lives <= 0:
        game_over = True
        cv2.putText(img, "GAME OVER", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    draw_ui(img, fps)
    cv2.imshow("Fruit Slash Game", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r") and game_over:
        lives = LIVES
        score = 0
        combo_count = 0
        combo_multiplier = 1
        game_over = False
        slash_points.clear()

cap.release()
cv2.destroyAllWindows()