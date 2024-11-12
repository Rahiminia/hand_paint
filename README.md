<div align='center'> 
  <img alt="HandPaint" src="https://github.com/user-attachments/assets/221b6cf4-2b31-46f6-831c-91e94ae8ca9b">
</div>

<div align='center'>
  <h1>HandPaint</h1>
</div>

<p align='center'>
  A fun project to draw with your fingertips 🖌️
</p>
<div align='center'>
  <a href="#features-">Features</a> .
  <a href="#requirements-">Requirements</a> .
  <a href="#installation-">Installation</a> .
  <a href="#usage-%EF%B8%8F">Usage</a> .
  <a href="#keymap-%EF%B8%8F">Keymap</a>
</div>
</br>
</br>
Welcome to HandPaint! This application allows users to create digital drawings using their webcam. Using gesture & motion detection, the program translates hand movements into drawing, making art fun, accessible and interactive.


## Overview 🔎

webcam live video is captured by [OpenCV](https://github.com/opencv/opencv) and fingertip locations are determined by [MediaPipe](https://github.com/google-ai-edge/mediapipe).


## Features ✨
- Point your finger to draw 
- Draw in 4 Different Colors
- Change brush size (3 Sizes)
- Save canvas in .jpg format
- Erase canvas by pressing one key


## Requirements 🧰

- python 3
- Opencv 4
- Mediapipe
*exact project requirements can be found in requirements.txt file*


## Installation 🔌

1. Clone the project

`git clone https://github.com/Rahiminia/hand_paint.git`

2. Install the requirements

`pip install -a requirements.txt`

> [!Note]
> It's always better to use an insolated dev environment in python (venv, virtualenv, etc.).
>


## Usage 🖌️
`python main.py`


## Keymap ⌨️
- [R G B L]: Change brush color
- [1 2 3]: Change brush size
- E: Erase canvas
- S: Save canvas *in the same dir as main.py*
- Q: Quit

## Contributing 🤝

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.


## License 📃

This project is licensed under the MIT License. See the LICENSE file for more details.


<h4>Please give this project a Star⭐ if you like it :)</h4>
