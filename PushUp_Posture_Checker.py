# Import necessary libraries
import cv2
import math as m
import mediapipe as mp


# Function to calculate Euclidean distance between two points (x1, y1) and (x2, y2)
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Function to calculate vector difference between two points x1 and x2
def findVector(x1, x2):
    vect = x2 - x1
    return vect

# Function to calculate angle between two vectors (x1, y1) and (x2, y2)
def findAngle(x1, y1, x2, y2):
    theta = m.acos((x1 * x2 + y1 * y2) / (m.sqrt((x1**2)+(y1**2))
                *m.sqrt(((x2)**2)+((y2)**2))))
    degree = int(180 / m.pi) * theta
    return degree

# Function to check if two values a and b are within a relative tolerance
def inRange(a, b, rel_tol=.02, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# Initialize repetition counters
reps = 0

# Select font for text display
font = cv2.FONT_HERSHEY_DUPLEX

# Define colors for drawing on the video frame
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
yellow = (0, 255, 255)
white = (248, 248, 255)


# Initialize MediaPipe pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Main processing block
if __name__ == "__main__":
    # Set the video file name
    file_name = 'pushup.mp4'
    cap = cv2.VideoCapture(file_name)

    # Retrieve video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize video writer for output
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    while cap.isOpened():
        # Capture each frame from the video
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break

        # Capture fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Capture height and width.
        h, w = image.shape[:2]

        # Convert frame from BGR to RGB for MediaPipe processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe to get pose landmarks
        keypoints = pose.process(image)

        # Convert the processed frame back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if pose landmarks are detected
        if keypoints.pose_landmarks:
            # Reference the pose landmarks and landmark positions
            lm = keypoints.pose_landmarks.landmark
            lmPose = mp_pose.PoseLandmark

            # Extract coordinates for key joints: shoulders, elbows, hips
            r_shldr_x, r_shldr_y = int(lm[lmPose.RIGHT_SHOULDER].x * width), int(lm[lmPose.RIGHT_SHOULDER].y * height)
            l_shldr_x, l_shldr_y = int(lm[lmPose.LEFT_SHOULDER].x * width), int(lm[lmPose.LEFT_SHOULDER].y * height)
            r_elbow_x, r_elbow_y = int(lm[lmPose.RIGHT_ELBOW].x * width), int(lm[lmPose.RIGHT_ELBOW].y * height)
            l_elbow_x, l_elbow_y = int(lm[lmPose.LEFT_ELBOW].x * width), int(lm[lmPose.LEFT_ELBOW].y * height)
            r_hip_x, r_hip_y = int(lm[lmPose.RIGHT_HIP].x * width), int(lm[lmPose.RIGHT_HIP].y * height)
            l_hip_x, l_hip_y = int(lm[lmPose.LEFT_HIP].x * width), int(lm[lmPose.LEFT_HIP].y * height)

            # Calculate distances between shoulders and hips, shoulders and elbows
            r_offset = findDistance(r_shldr_x, r_shldr_y, r_hip_x, r_hip_y)
            l_offset = findDistance(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)
            r_arm_distance = findDistance(r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y)
            l_arm_distance = findDistance(l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y)

            # Count repetitions based on right arm distance threshold
            if int(r_arm_distance) == 108:
                reps += 0.5
            # Display the repetition count on the frame
            cv2.putText(image, 'Repetitions: ' + str(int(reps)), (50, 100), font, 1.5, white, 2)
            # Display hip/shoulder alignment status
            cv2.putText(image, 'Hip/Shoulder: ', (5, 850), font, 0.9, white, 2)

            # Determine alignment status and set the corresponding color
            if inRange(r_offset, l_offset):
                alignment_color = green
                alignment_status = 'Aligned'
            else:
                alignment_color = red
                alignment_status = 'Not Aligned'

            # Display alignment status and draw lines between joints
            cv2.putText(image, alignment_status, (250, 850), font, 0.8, alignment_color, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), alignment_color, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (r_hip_x, r_hip_y), alignment_color, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), alignment_color, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y), alignment_color, 4)

            # Calculate angles for right and left arm
            r_arm_angle = findAngle(r_shldr_x - r_elbow_x, r_shldr_y - r_elbow_y, 0, 150)
            l_arm_angle = findAngle(l_shldr_x - l_elbow_x, l_shldr_y - l_elbow_y, 0, 150)

            # Draw circles at the key joint positions
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, yellow, -1)
            cv2.circle(image, (l_elbow_x, l_elbow_y), 7, yellow, -1)
            cv2.circle(image, (r_elbow_x, r_elbow_y), 7, yellow, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
            cv2.circle(image, (r_hip_x, r_hip_y), 7, yellow, -1)
            cv2.circle(image, (l_shldr_x, l_shldr_y - 75), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y - 75), 7, yellow, -1)
            cv2.circle(image, (l_shldr_x, l_shldr_y + 150), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y + 150), 7, yellow, -1)

            # Display arm angle correctness
            cv2.putText(image, 'Arm Angle: ', (5, 900), font, 0.9, white, 2)
            angle_color = green if 35 < r_arm_angle < 55 and 35 < l_arm_angle < 55 else red
            angle_status = 'Correct' if angle_color == green else 'Incorrect'

            # Display angle correctness and draw angle values near the elbows
            cv2.putText(image, angle_status, (250, 900), font, 0.9, angle_color, 2)
            cv2.putText(image, f'{int(r_arm_angle)}{chr(176)}', (r_elbow_x - 20, r_elbow_y + 30), font, 0.8, angle_color, 1)
            cv2.putText(image, f'{int(l_arm_angle)}{chr(176)}', (l_elbow_x - 10, l_elbow_y + 30), font, 0.8, angle_color, 1)

            # Draw lines to represent arm angles
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y + 150), blue, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 75), blue, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_elbow_x, l_elbow_y), angle_color, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y + 150), blue, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y - 75), blue, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_elbow_x, r_elbow_y), angle_color, 4)

        # Write annotated frame to output video
        video_output.write(image)

        # Display the current frame with annotations
        dsize = (w+200,h)
        output = cv2.resize(image, dsize )
        cv2.imshow('Push Up Posture Checker', output)
        
        # Exit loop if 'q' key is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
# Release video capture and writer resources
cap.release()
cv2.destroyAllWindows()