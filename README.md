# HandTrackingGame

A real-time hand-controlled drawing and physics game built with Python, OpenCV, and MediaPipe.

The project started as a small experimental prototype and was later revisited to improve its structure, interaction model, and gameplay. The repository keeps the different stages of the project together so the evolution of the implementation can be followed directly from the source code.

## Project Evolution

### 1. Original Prototype — 2024

**File:** `Source/original_prototype.py`

The original implementation established the core interaction:

- Track the user's hand through a webcam.
- Use the index fingertip as the drawing input.
- Create short-lived line segments on the screen.
- Let the ball interact with those drawn segments through vector reflection.
- Collect bonuses and power-ups.
- Handle lives and bomb interactions.
- Render the game over the live camera feed.

This version intentionally remains close to the original implementation. It represents the starting point of the project rather than the final architecture.

### 2. Structured Revision — 2026

**File:** `Source/structured_revision_2026.py`

The second version revisits the original idea from an engineering perspective.

The main changes include:

- Object-oriented game structure.
- Centralized configuration through `Config`.
- Explicit game states for start, gameplay, and game over.
- Separate components for hand tracking, drawing, game objects, and ball physics.
- More explicit resource and webcam handling.
- Clearer image-loading errors.
- Difficulty progression through ball-speed scaling.
- Finite line-segment collision geometry.
- Restart and canvas-clear controls.

The interaction remains intentionally close to the original prototype while the internal organization is substantially cleaner.

### 3. Creative Revision — 2026

**File:** `Source/creative_revision_2026.py`

The third version extends the structured game into a richer interactive experiment.

Additional mechanics include:

- The tracked hand can act as a temporary paddle.
- Convex-hull geometry is used to represent the hand for paddle interaction.
- A fist gesture activates a temporary shield.
- Combo scoring rewards quick successive collections.
- A second ball is introduced at higher difficulty.
- Ball trails provide motion feedback.
- Particle effects are used for collisions and item collection.
- Bomb interactions include temporary visual feedback.
- The HUD and interaction feedback are expanded.

This version focuses on exploring how hand tracking can become part of the game mechanics itself rather than serving only as an input method.

---

## Core Interaction

The central idea is simple:

**Draw a line with your hand → the ball reacts to it → collect items → survive the hazards.**

The later revision adds another interaction layer:

**Move your hand as a paddle → use gestures → interact with the game more directly.**

## Technical Pipeline

```text
Webcam Frame
     ↓
MediaPipe Hands
     ↓
Hand Landmarks
     ↓
Pixel Coordinates
     ↓
Drawing / Hand Geometry
     ↓
Collision Detection
     ↓
Vector Reflection
     ↓
Game Physics
     ↓
Game State
     ↓
Rendering and HUD
```

The project therefore combines several areas in one real-time loop:

- Computer Vision
- Hand Tracking
- Human-Computer Interaction
- 2D Geometry
- Basic Physics
- Real-Time Rendering
- Interactive Game Design

## Interaction Model

| Interaction | Prototype | Structured Revision | Creative Revision |
|---|:---:|:---:|:---:|
| Index-finger drawing | ✓ | ✓ | ✓ |
| Temporary drawing barriers | ✓ | ✓ | ✓ |
| Ball reflection | ✓ | ✓ | ✓ |
| Bonus collection | ✓ | ✓ | ✓ |
| Power-up system | ✓ | ✓ | ✓ |
| Life power-up | ✓ | ✓ | ✓ |
| Bomb hazard | ✓ | ✓ | ✓ |
| Explicit game states | — | ✓ | ✓ |
| Hand as paddle | — | — | ✓ |
| Fist-based shield | — | — | ✓ |
| Combo scoring | — | — | ✓ |
| Multiple balls | — | — | ✓ |
| Particle effects | — | — | ✓ |
| Ball trail | — | — | ✓ |

## Physics and Collision

The ball is represented by a position, velocity, and radius.

When the ball interacts with a drawn barrier, the implementation uses the normal of the segment to reflect the velocity vector. The revised versions also evaluate the distance to the **finite segment**, rather than treating the stroke as an infinitely long line.

A small separation step is applied after a collision to reduce repeated overlap with the same barrier.

The creative revision applies the same reflection idea to the hand-paddle interaction.

## Hand Tracking

The project uses MediaPipe Hands to obtain hand landmarks from the webcam stream.

The original prototype uses the index fingertip as its primary interaction point.

The structured revision isolates hand tracking into a dedicated `HandTracker` component.

The creative revision builds further on the landmark data to:

- Convert landmarks to pixel coordinates.
- Construct a convex hull around the hand.
- Detect a fist using a geometric heuristic.
- Use the hand geometry as an additional collision surface.

## Game State

The structured and creative revisions explicitly separate the main game states:

```text
START
  ↓
PLAYING
  ↓
GAME_OVER
  ↓
PLAYING
```

This makes starting, restarting, and game-over behavior easier to manage than in the original prototype.

## Controls

The revised versions use the following keyboard controls:

| Key | Action |
|---|---|
| `SPACE` | Start / restart the game |
| `D` | Start drawing |
| `S` | Stop drawing |
| `C` | Clear the drawing canvas |
| `R` | Switch drawing color to red |
| `P` | Switch drawing color to purple |
| `ESC` | Quit |

The original prototype has a smaller keyboard-control set and is intentionally preserved as the earlier implementation.

## Project Structure

```text
HandTrackingGame/
│
├── Source/
│   ├── original_prototype.py
│   ├── structured_revision_2026.py
│   └── creative_revision_2026.py
│
├── ball.png
├── bomb.png
├── powerup.png
├── life_powerup.png
├── bonus.png
│
└── README.md
```

The three source files are kept separately rather than replacing one version with another. This makes the development history and technical progression easier to inspect.

## Why Keep Three Versions?

The three implementations answer different questions.

**Original Prototype**

> Can hand tracking be used to draw temporary barriers that influence a moving ball?

**Structured Revision**

> How can the original experiment be reorganized into clearer, reusable components?

**Creative Revision**

> How far can the hand itself become part of the game mechanics?

Keeping all three versions makes these changes visible in the code rather than describing them only in documentation.

## Development Approach

The project was originally created as an experimental computer-vision interaction project and later revisited as a technical refinement and feature-development exercise.

The later revisions focus on:

- Refactoring the original implementation.
- Making technical responsibilities clearer.
- Improving collision handling.
- Adding explicit game states.
- Exploring additional interaction patterns.
- Experimenting with richer feedback and gameplay.

The source code is intentionally kept readable so that the implementation decisions can be followed directly.

## Technical Stack

- **Python**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- **Pillow**

## Limitations

The project is an interactive prototype rather than a production-ready game engine.

Current limitations include:

- Performance depends on webcam resolution and hardware.
- Hand tracking can be affected by lighting, occlusion, and camera quality.
- Gesture detection uses geometric heuristics rather than a trained gesture-classification model.
- The physics model is intentionally lightweight.
- The original prototype assumes a default webcam setup and has less defensive error handling than the later revisions.

## Future Directions

Potential extensions include:

- More robust gesture classification.
- Better calibration for different camera setups.
- More sophisticated collision handling.
- Additional game mechanics driven by hand gestures.
- Performance profiling and optimization.
- A cleaner separation between the rendering layer and game logic.

## Project Context

HandTrackingGame is part of a broader set of experiments exploring real-time computer vision and human-computer interaction.

The project is particularly useful as a compact example of how a computer-vision input stream can be connected to geometry, physics, state management, and interactive feedback in a single application.

---

**Author:** Kian Shojaei  
**Repository:** [KianShojaei/HandTrackingGame](https://github.com/KianShojaei/HandTrackingGame)
