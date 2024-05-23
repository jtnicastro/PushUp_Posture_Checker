# Push Up Posture Checker
This project is a Python script that uses OpenCV and MediaPipe to process a video of a person performing push ups. Users can upload a video of themselves performing push ups and the program detects arm angles, provides real-time feedback on muscular symmetry, and counts repetitions.

## Features
+ **Real-Time Feedback**: Displays arm angles and shoulder alignment status.
+ **Repetition Counting**: Counts push-up repetitions based on arm movement.
+ **Output Video**: Annotates the input video with arm angles, shoulder alignment, and repetition count.

## Installation
1. **Clone the Repository:**
    ```python
    git clone https://github.com/jtnicastro/PushUp_Posture_Checker.git
    ```

2. **Navigate to the Directory:**
    ```python
    cd PushUp_Posture_Checker
    ```

3. **Install the Required Packages:**
    ```python
    pip install -r requirements.txt
    ```

## Usage
1. **Place Your Video:**

    Ensure your input video file is named **'pushup.mp4'** and is placed in the project directory.

   
2. **Run the Script:**
    ```python
    python script.py
    ```

3. **Output:** 

    The script processes the video, provides real-time feedback, and saves an annotated output video as **'output.mp4'**.

## Functions
+ findDistance(x1, y1, x2, y2): Calculates the Euclidean distance between two points.
+ findVector(x1, x2): Calculates the vector from point x1 to point x2.
+ findAngle(x1, y1, x2, y2): Calculates the angle between two vectors.
+ inRange(a, b, rel_tol, abs_tol): Checks if two values are within a specified range.


## Example Outputs

https://github.com/jtnicastro/PushUp_Posture_Checker/assets/78828629/163f2bb6-cc91-4914-b0b5-a5bca9626161


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.tldrlegal.com/license/mit-license) file for details.

## Acknowledgments
+ [Mediapipe](https://pypi.org/project/mediapipe/)
+ [OpenCV](https://opencv.org/)




This project aims to assist individuals in maintaining proper form during push-ups by providing real-time visual feedback and monitoring repetitions. Proper form helps prevent injuries and ensures effective workouts.
