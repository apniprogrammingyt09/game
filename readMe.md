# Fruit Slash Game

This project implements a "Fruit Slash Game" in Python using the OpenCV and MediaPipe libraries for hand tracking and rendering. The user interacts with the game by using their hand's index finger to slice fruits (or avoid bombs). The goal is to score as many points as possible while avoiding bomb collisions and preventing missed fruits.

---

## Features

- **Hand Tracking:** Uses MediaPipe to detect and track the player's hand.
- **Fruit Spawn and Motion:** Fruits and bombs are spawned and move dynamically across the screen.
- **Combo Multiplier:** Score higher points with consecutive slices within a short period.
- **Special Effects:** Splashes and explosions for gameplay visuals.
- **Game Mechanics:** Includes scoring, lives control, and a game-over scenario.
- **Graphics Overlay:** Fruits, splashes, and UI dynamically rendered on webcam feed.

---

## Requirements

To run this project, ensure the following prerequisites are installed:

- **Python version:** Python 3.9.13 (as used in this project)
- **Libraries:**
  - OpenCV (`opencv-python`)
  - MediaPipe (`mediapipe`)
  - NumPy (`numpy`)

Use the following command to install missing libraries:

```bash
pip install mediapipe opencv-python numpy
```

---

## How to Run

1. Clone or download the repository containing this script.
2. Add the following image assets in the same directory as the script:
   - `apple.png` (for apple fruit)
   - `banana.png` (for banana fruit)
   - `watermelon.png` (for watermelon fruit)
   - `bomb.png` (for bombs)

3. Run the script using the command:

   ```bash
   python game.py
   ```

4. Turn on your webcam and use your index finger to slice fruits in the video window.

5. To exit the game, press the **`q`** key. To restart during a Game Over, press the **`r`** key.

---

## Code Overview

### 1. **Game Initialization**
   - **Libraries Used:**
     - `cv2`, `mediapipe`, and `numpy` for vision processing and rendering.
   - **Global Variables:**
     - Define game settings like fruit speed, lives, intervals, screen dimensions, etc.
   - **Images:**
     - Load emoji images for fruits and bombs.

### 2. **Functions**
   - **`overlay_image_alpha`:** Handles transparent overlays (used for rendering fruits and bombs).
   - **`spawn_fruit`:** Randomly spawns fruits or bombs at the bottom of the screen with upward velocity.
   - **`move_fruits`:** Moves fruits along their trajectory and handles missed fruits (reduces lives).
   - **`create_explosion_effect/create_fruit_splash`:** Generates effects for bomb explosions and fruit slices.
   - **`draw_splashes/draw_explosions/draw_fruits`:** Renders effects and fruits dynamically.
   - **`check_slices`:** Detects slicing events and updates the score, multiplier, and effects accordingly.
   - **`distance`:** Calculates Euclidean distance used to detect slicing proximity.
   - **`draw_ui`:** Renders game status (score, lives, FPS, combo message).
   - **`draw_slash`:** Draws slash effect when the index finger is moving.

### 3. **Main Game Loop**
   - **Webcam Capture:** Captures and flips frames for input.
   - **Hand Tracking:** Uses MediaPipe to detect the index finger's position.
   - **Game Rendering:**
     - Updates the game state (fruit motion, slices, effects).
     - Refreshes the UI with scores, lives, and combo messages.
   - **Game Over Logic:**
     - Displays "Game Over" when lives reach zero.
     - Press `r` to reset the game.
   - **Keyboard Input:**
     - Press `q` to quit the game.

---

## Controls and Gameplay

- **Use Your Index Finger:** Move your index finger into the webcam's view to interact with the game.
- **Slice Fruits:** Move your finger over fruits before they fall to slice them.
- **Avoid Bombs:** Be careful not to slice bombs, as they will reduce your lives.
- **Combo Bonuses:** Slice multiple fruits in quick succession to earn combo points.

   | Control         | Key         | Description            |
   |-----------------|-------------|------------------------|
   | Quit Game       | `q`         | Exit the game window.  |
   | Restart Game    | `r`         | Restart after game over.|

---

## Example Visuals

The game captures live webcam footage and dynamically overlays game graphics. Example scenarios:

1. **Spawn Timer:** Fruits and bombs are spawned at regular intervals from the bottom of the screen, launching upward with random speed and angles.
2. **Fruit Splashes:** Create colored splashes when fruits are sliced.
3. **Explosion Effect:** Triggered upon hitting bombs, rendering an impactful burst animation.
4. **UI Elements:** Display live score, lives left, and FPS at the top of the screen.

---

## Improvements & Customizations

- **Add More Fruits/Bombs:** Add additional fruit/bomb images and configure their behavior in the `fruit_map`.
- **Adjust Difficulty:** Modify the `SPAWN_INTERVAL`, fruit speeds, and lives to control difficulty.
- **Special Fruit Effects:** Add functionality for power-ups or time extensions.
- **Visual Enhancements:** Improve graphical effects using more advanced rendering techniques.

---

## Troubleshooting

1. **Webcam Not Detected:** Ensure your webcam is connected and properly configured.
2. **Required Images Missing:** Place the images (`apple.png`, `banana.png`, `watermelon.png`, `bomb.png`) in the same directory as the script.
3. **Imports Not Found:** Install necessary libraries using `pip install mediapipe opencv-python numpy`.

---

Feel free to extend and customize the game as per your creativity. Enjoy slicing fruits!