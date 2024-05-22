#                   IMPORTING                   #
# Import necessary libraries
import cv2
import math as m
import mediapipe as mp


# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate vector
def findVector(x1, x2):
    vect = x2 - x1
    return vect

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((x1 * x2 + y1 * y2) / (m.sqrt((x1**2)+(y1**2))
                *m.sqrt(((x2)**2)+((y2)**2))))
    degree = int(180 / m.pi) * theta
    return degree

# Calculate closeness
def inRange(a, b, rel_tol=.02, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)



#                   INITIALIZATION                  #

# Initialize repetition counters
reps = 0

# Select font
font = cv2.FONT_HERSHEY_DUPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
yellow = (0, 255, 255)
white = (248, 248, 255)


# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize footage
if __name__ == "__main__":
    file_name = 'pushup.mp4'
    cap = cv2.VideoCapture(file_name)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    while cap.isOpened():
        # Capture each frames
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # Capture fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Capture height and width.
        h, w = image.shape[:2]

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        keypoints = pose.process(image)

        # Convert back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # Obtain positions of desired joints
        # Right shoulder
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        # Left shoulder.
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        # Right elbow
        r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
        r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
        # Left elbow
        l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
        l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
        # Right hip
        r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
        # Left hip
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)


        # Calculate distance
        # between shoulders and hips
        r_offset = findDistance(r_shldr_x, r_shldr_y, r_hip_x, r_hip_y)
        l_offset = findDistance(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)
        # distance of shoulder to elbow to signify repetition
        r_arm_distance = findDistance(r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y)
        l_arm_distance = findDistance(l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y)

        # Count number of repetitions by calculating when shoulder to elbow distance is lowest
        "Ideally 108 was selected as a low threshold so everytime the elbow to shoulder distance went to this value" \
        "it would count half a repetition (one for one the way down the other for coming back up). Unfortunately the camera " \
        "angle caused some inconsistencies in the arm distance and the repetition did not work as well as anticipated"
        if int(r_arm_distance) == 108:
            reps += .5
        cv2.putText(image, 'Repetitions: ' + str(int(reps)), (50, 100), font, 1.5, white, 2)

        # Display text
        cv2.putText(image, 'Hip/Shoulder: ', (5, 850), font, .9, white, 2)

        # Verifies proper hip and shoulder alignment
        if inRange(r_offset, l_offset):
            cv2.putText(image, 'Aligned', (250, 850), font, .8, green, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (r_hip_x, r_hip_y), green, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), green, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y), green, 4)
        else:
            cv2.putText(image, 'Not Aligned', (250, 850), font, .8, red, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (r_hip_x, r_hip_y), red, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), red, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y), red, 4)

        # Calculate vectors
        r_shldr_vect_x = findVector(r_shldr_x, r_shldr_x)
        r_shldr_vect_y = findVector(r_shldr_y, r_shldr_y + 150)
        l_shldr_vect_x = findVector(l_shldr_x, l_shldr_x)
        l_shldr_vect_y = findVector(l_shldr_y, l_shldr_y + 150)
        r_elbow_vect_x = findVector(r_shldr_x, r_elbow_x)
        r_elbow_vect_y = findVector(r_shldr_y, r_elbow_y)
        l_elbow_vect_x = findVector(l_shldr_x, l_elbow_x)
        l_elbow_vect_y = findVector(l_shldr_y, l_elbow_y)
        # Calculate angles.
        r_arm_angle = findAngle(r_shldr_vect_x, r_shldr_vect_y, r_elbow_vect_x, r_elbow_vect_y)
        l_arm_angle = findAngle(l_shldr_vect_x, l_shldr_vect_y, l_elbow_vect_x, l_elbow_vect_y)

        # Draw landmarks
        # shoulders
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, yellow, -1)
        # elbows
        cv2.circle(image, (l_elbow_x, l_elbow_y), 7, yellow, -1)
        cv2.circle(image, (r_elbow_x, r_elbow_y), 7, yellow, -1)
        # hips
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        cv2.circle(image, (r_hip_x, r_hip_y), 7, yellow, -1)
        # parallel line through shoulder to display where angle is measured from
        cv2.circle(image, (l_shldr_x, l_shldr_y - 75), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y - 75), 7, yellow, -1)
        cv2.circle(image, (l_shldr_x, l_shldr_y + 150), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y + 150), 7, yellow, -1)


        # Display text
        cv2.putText(image, 'Arm Angle: ', (5, 900), font, .9, white, 2)
        degree_sign = u"\N{DEGREE SIGN}"

        # Determine if arms within proper position
        # These values were picked to ensure the elbow is approximately 45 degrees from the shoulder
        if 35 < r_arm_angle < 55  and 35 < l_arm_angle < 55:
            cv2.putText(image, 'Correct', (250, 900), font, .9, green, 2)
            cv2.putText(image, str(int(r_arm_angle)) + '*', (r_elbow_x - 20, r_elbow_y + 30), font, 0.8, green, 1)
            cv2.putText(image, str(int(l_arm_angle)) + '*', (l_elbow_x - 10, l_elbow_y + 30), font, 0.8, green, 1)


            # Connect landmarks
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y + 150), blue, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 75), blue, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_elbow_x, l_elbow_y), green, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y + 150), blue, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y - 75), blue, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_elbow_x, r_elbow_y), green, 4)

        else:
            cv2.putText(image, 'Incorrect', (250, 900), font, 0.9, red, 2)
            cv2.putText(image, str(int(r_arm_angle)) + '*', (r_elbow_x - 20, r_elbow_y + 30), font, 0.8, red, 1)
            cv2.putText(image, str(int(l_arm_angle)) + '*', (l_elbow_x - 10, l_elbow_y + 30), font, 0.8, red, 1)

            # Connect landmarks
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y + 150), blue, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 75), blue, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_elbow_x, l_elbow_y), red, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y + 150), blue, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y - 75), blue, 4)
            cv2.line(image, (r_shldr_x, r_shldr_y), (r_elbow_x, r_elbow_y), red, 4)


        # Write frames.
        video_output.write(image)

        # Display.
        dsize = (w+200,h)
        output = cv2.resize(image, dsize )
        cv2.imshow('Push Up Posture Checker', output)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()