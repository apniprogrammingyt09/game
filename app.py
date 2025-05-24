import cv2
import streamlit as st
import numpy as np
import time
import random
import mediapipe as mp
import math

# Declare all global variables upfront
global last_spawn_time
last_spawn_time = 0  # Initialize the global variable before any other usage


@st.cache_data
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found or invalid at path: {path}")
    return img  # Keep in BGRA format


# Load images
apple_img = load_image("apple.png")
banana_img = load_image("banana.png")
watermelon_img = load_image("watermelon.png")
bomb_img = load_image("bomb.png")

# Define fruit properties
fruit_map = {
    "apple": {"img": apple_img, "points": 100, "size": 40, "color": (0, 0, 255)},
    "banana": {"img": banana_img, "points": 150, "size": 50, "color": (0, 255, 255)},
    "watermelon": {"img": watermelon_img, "points": 200, "size": 70, "color": (0, 255, 0)},
}

# Constants and Variables
SLASH_LENGTH = 15
LIVES = 3
SPAWN_INTERVAL = 1.2

fruits, splashes, explosions = [], [], []
score, lives = 0, LIVES
last_slice_time = 0
slash_points = []
slash_color = (255, 255, 255)
combo_count, combo_multiplier = 0, 1
combo_time_limit = 2
combo_message = ""

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
)


def overlay_image_alpha(background, overlay, pos, overlay_size):
    x, y = pos
    overlay = cv2.resize(overlay, overlay_size)

    # Ensure the overlay image has an alpha channel
    if overlay.shape[2] != 4:
        raise ValueError(f"Overlay image must have 4 channels but has {overlay.shape[2]}!")

    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))
    mask = a / 255.0  # Normalize alpha channel

    h, w = overlay.shape[:2]

    # Validate dimensions to avoid out-of-bounds errors
    if x < 0 or y < 0 or y + h > background.shape[0] or x + w > background.shape[1]:
        # Do not overlay if it goes out of bounds
        return

    roi = background[y:y + h, x:x + w]
    for c in range(3):
        roi[:, :, c] = (1.0 - mask) * roi[:, :, c] + mask * overlay_rgb[:, :, c]

    background[y:y + h, x:x + w] = roi


def distance(a, b):
    return int(math.hypot(a[0] - b[0], a[1] - b[1]))


def spawn_fruit():
    x = random.randint(50, 600)
    is_bomb = random.random() < 0.2
    fruit = {
        "pos": [x, 600],
        "vel": [random.randint(-4, 4), -random.randint(13, 18)],
        "size": 40,
        "bomb": is_bomb,
        "img": bomb_img if is_bomb else None,
        "points": 0,
        "color": (255, 255, 255),
    }
    if not is_bomb:
        kind = random.choice(list(fruit_map.keys()))
        f = fruit_map[kind]
        fruit.update({"img": f["img"], "points": f["points"], "size": f["size"], "color": f["color"], "kind": kind})
    else:
        fruit["kind"] = "bomb"
    fruits.append(fruit)


def move_fruits():
    global lives
    for fruit in fruits[:]:
        fruit["vel"][1] += 0.5
        fruit["pos"][0] += fruit["vel"][0]
        fruit["pos"][1] += fruit["vel"][1]
        if fruit["pos"][1] > 700:
            if not fruit["bomb"]:
                lives -= 1
            fruits.remove(fruit)


def add_splash(pos, color):
    for _ in range(10):
        splashes.append({
            "pos": [pos[0] + random.randint(-10, 10), pos[1] + random.randint(-10, 10)],
            "vel": [random.uniform(-3, 3), random.uniform(-3, 3)],
            "radius": random.randint(3, 6),
            "color": color,
            "life": 15,
        })


def add_explosion(pos):
    for _ in range(30):
        explosions.append({
            "pos": [pos[0], pos[1]],
            "vel": [random.uniform(-5, 5), random.uniform(-5, 5)],
            "radius": random.randint(5, 10),
            "color": (255, 255, 255),
            "life": 20,
        })


def update_effects(img):
    for s in splashes[:]:
        s["pos"][0] += s["vel"][0]
        s["pos"][1] += s["vel"][1]
        s["life"] -= 1
        alpha = s["life"] / 15
        cv2.circle(img, (int(s["pos"][0]), int(s["pos"][1])), s["radius"], tuple(int(c * alpha) for c in s["color"]),
                   -1)
        if s["life"] <= 0:
            splashes.remove(s)

    for e in explosions[:]:
        e["pos"][0] += e["vel"][0]
        e["pos"][1] += e["vel"][1]
        e["life"] -= 1
        alpha = e["life"] / 20
        cv2.circle(img, (int(e["pos"][0]), int(e["pos"][1])), e["radius"], tuple(int(c * alpha) for c in e["color"]),
                   -1)
        if e["life"] <= 0:
            explosions.remove(e)


def check_slices(index_pos):
    global score, lives, slash_color, combo_count, combo_multiplier, combo_message, last_slice_time
    combo_scored = False
    for fruit in fruits[:]:
        d = distance(index_pos, fruit["pos"])
        if d < fruit["size"]:
            if fruit["bomb"]:
                lives -= 1
                add_explosion(fruit["pos"])
                fruits.clear()
                splashes.clear()
                return
            else:
                score += fruit["points"] * combo_multiplier
                add_splash(fruit["pos"], fruit["color"])
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
            combo_message = ""
        last_slice_time = time.time()


def draw_fruits(img):
    for fruit in fruits:
        if fruit.get("img") is None or fruit.get("size") is None:
            continue  # Skip if image or size is invalid

        x, y = int(fruit["pos"][0]), int(fruit["pos"][1])
        size = fruit["size"] * 2
        overlay_image_alpha(img, fruit["img"], (x - size // 2, y - size // 2), (size, size))


def draw_slash(img, index_pos):
    global slash_points
    slash_points.append(index_pos)
    if len(slash_points) > SLASH_LENGTH:
        slash_points.pop(0)
    if len(slash_points) > 1:
        pts = np.array(slash_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, slash_color, 15)


def draw_ui(img, fps):
    h, w = img.shape[:2]
    cv2.putText(img, f"Score: {score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(img, f"Lives: {lives}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f"FPS: {int(fps)}", (w - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if combo_message:
        cv2.putText(img, combo_message, (w // 3, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)


# Streamlit UI
st.title("Fruit Ninja in Streamlit 🍉")
run_game = st.button("Start Game")
frame_placeholder = st.empty()

if run_game:
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    last_spawn_time = time.time()  # Global variable used here

    while cap.isOpened() and lives > 0:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)  # Improve brightness and contrast

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        index_pos = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark[8]
            index_pos = (int(lm.x * w), int(lm.y * h))
            check_slices(index_pos)
            draw_slash(frame, index_pos)

        # Spawn fruit
        if time.time() - last_spawn_time > SPAWN_INTERVAL:
            spawn_fruit()
            last_spawn_time = time.time()

        move_fruits()
        draw_fruits(frame)
        update_effects(frame)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        draw_ui(frame, fps)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    st.write("Game Over! Your score:", score)
