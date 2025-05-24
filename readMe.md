# Fruit Slash Game

This project is for  **build-your-own game program**  to create an interactive "Fruit Slash Game" in Python. Leveraging modern vision libraries like OpenCV and MediaPipe, players can use their hand movements (tracked via the webcam) to slice fruits while avoiding bombs and achieving high scores. The project serves as a template for educational or personal game development projects.

---

## Overview

The Fruit Slash Game features:
- Gesture-based gameplay, using hand tracking with MediaPipe.
- Vivid fruit-slicing effects and bomb explosions.
- Game mechanics include scoring, combos, and lives.
- Customizable difficulty settings and game behavior.

---

### Table of Contents

1. [Requirements](#requirements)
2. [Setup Instructions](#setup-instructions)
3. [Running the Game](#running-the-game)
4. [Code Explanation](#code-explanation)
5. [Game Play Instructions](#game-play-instructions)
6. [Customization Options](#customization-options)
7. [Troubleshooting](#troubleshooting)

---

## Requirements

Before you begin, ensure you have the following:

1. **Python 3.9.13** or higher installed.
2. **Required Libraries:**
   - OpenCV (`opencv-python`)
   - MediaPipe (`mediapipe`)
   - NumPy (`numpy`)

   Install missing libraries using pip:

   ```shell
   pip install mediapipe opencv-python numpy
   ```

3. **Assets:**
   - Place these required image files in the same directory as the script:
     - `apple.png`, `banana.png`, `watermelon.png`, `bomb.png`

4. **Webcam:** A working webcam is required for hand tracking.

---

## Setup Instructions

1. **Clone the Repository:** 
   Download or clone the repository containing this script to your local machine.

   ```bash
   git clone https://github.com/apniprogrammingyt09/game.git
   ```

2. **Install Dependencies:** Refer to the [Requirements](#requirements) section and ensure all packages and assets are ready.

3. **Prepare Assets:** Add the images (`apple.png`, `banana.png`, `watermelon.png`, `bomb.png`) to the script folder.

---

## Running the Game

Run the script using Python. Open a terminal in the project folder and execute:

```bash
python game.py
```

1. Once the game starts:
   - The webcam feed will appear in a game window.
   - Use your index finger to interact with the game interface.

2. **Quit the Game:** Press `q` to close the game window.
3. **Restart After Game Over:** Press `r` to restart the game if you run out of lives.

---

## Code Explanation

The code has been divided into the following functional sections:

1. **Game Initialization:**
   - Imports required libraries (`cv2`, `numpy`, and `mediapipe`).
   - Defines game settings (`Lives`, `Spawn Interval`, etc.).
   - Loads fruit and bomb images for visuals.

2. **Core Functions:**
   - `overlay_image_alpha`: Handles rendering of transparent fruit/bomb images.
   - `spawn_fruit`: Spawns random fruits or bombs dynamically.
   - `move_fruits`: Updates the positions of fruits/bombs on-screen and handles "missed" events for lives reduction.
   - `create_explosion_effect` / `create_fruit_splash`: Adds visual effects for interactions (slicing/explosions).
   - `check_slices`: Calculates slicing events and scoring logic.
   - `draw_ui`: Updates game visuals such as Score, FPS, Lives, and Combo messages.

3. **Game Loop:**
   - Continuously captures webcam feed using OpenCV.
   - Tracks gestures using MediaPipe, particularly monitoring the index finger.
   - Displays fruits, bombs, and UI elements, refreshing the screen with real-time updates.

4. **Gameplay Mechanics:**
   - **Scoring System:** Increases the score for slicing fruits. Combos result in multipliers.
   - **Lives:** Lose a life if fruits are missed or bombs are hit.
   - **Restart/Exit Control:** Restart with `r` and exit with `q`.

---

## Game Play Instructions

- **Control:** Use your **index finger** to slice fruits by moving it through the path of fruit objects.
- **Goal:** 
   1. Slice as many fruits as possible before losing all lives.
   2. Avoid cutting bombs, as they reduce lives or end the game.
   3. Chain multiple slices together quickly to earn combo points.

- **Keyboard Controls:**

   | Key     | Action                    |
   |---------|---------------------------|
   | `q`     | Quit the game             |
   | `r`     | Restart after game over   |

---

## Customization Options

You can modify the game's behavior as follows:

1. **Add More Fruits:**
   - Add more fruit image files in the project folder.
   - Update the `fruit_map` dictionary in the code with custom properties.

   ```python
   "pineapple": {"img": pineapple_img, "points": 250, "size": 60}
   ```

2. **Configure Difficulty:**
   - Adjust the `SPAWN_INTERVAL` variable to control how fast fruits/bombs appear.
   - Increase/decrease fruit speed in the `spawn_fruit` function.

3. **Special Effects:**
   - Edit explosion or splash effects for enhanced animations in:
     - `create_explosion_effect`
     - `create_fruit_splash`

4. **Visual Improvements:**
   - Use better-resolution emoji or sprite images for fruits/bombs.

---

## Troubleshooting

1. **Webcam Issues:**
   - Ensure your webcam is connected and not in use by another application.
   - Troubleshoot using OpenCV's basic camera capture script to verify compatibility.

2. **Missing Assets:**
   - Double-check that all required assets (`apple.png`, `banana.png`, etc.) are in the correct directory.

3. **Imports Not Found:**
   - Run the following to install modules:
     ```bash
     pip install mediapipe opencv-python numpy
     ```

4. **Low Frame Rate:**
   - Performance may vary on slower systems. Close background applications for better FPS during gameplay.

---

## Future Enhancements

- Add power-ups and time-limited bonuses.
- Implement high-score tracking using a local database or file storage.
- Introduce multi-player mode for competitive slicing.

---

Enjoy the game and happy slicing! For suggestions or support, feel free to reach out.

---