import cv2
import numpy as np
import json
import os
import glob


def click_event(event, x, y, flags, params):
    frame = params['frame']
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the BGR color of the pixel where the user clicked
        bgr_color = frame[y, x, :]

        color_range = 20

        lower_range = np.clip(bgr_color - color_range, 0, 255)
        upper_range = np.clip(bgr_color + color_range, 0, 255)

        for i in range(3):
            if bgr_color[i] - color_range < 0:
                lower_range[i] = 0
            if bgr_color[i] + color_range > 255:
                upper_range[i] = 255


        params['range'] = (lower_range, upper_range)
        # Signal that the color has been picked
        params['picked'] = True
        # Provide immediate feedback and close window
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Select Color on the Frame', frame)
        cv2.waitKey(500)  # Wait for 500 ms
        cv2.destroyAllWindows()

def find_color_clusters(image, lower_range, upper_range):
    # Create a mask with the specified BGR color range
    # Note: OpenCV does not support inRange for BGR directly, we use bitwise operations instead
    lower_bound = np.array(lower_range, dtype="uint8")
    upper_bound = np.array(upper_range, dtype="uint8")
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def cluster_centers(centers, threshold):
    clustered = []

    for center in centers:
        # Check if the center is close to an existing cluster
        found_cluster = False
        for cluster in clustered:
            if np.linalg.norm(np.array(center) - np.array(cluster)) <= threshold:
                found_cluster = True
                cluster.append(center)
                break

        if not found_cluster:
            clustered.append([center])

    # Calculate average of each cluster
    return [tuple(np.mean(cluster, axis=0).astype(int)) for cluster in clustered]

def calculate_center_of_mass(contour):
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        return cx, cy
    return None


def process_and_visualize_video(video_path, output_path, centers):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error reading first frame")
        return

    color_params = {'frame': frame, 'picked': False, 'range': None}

    # Show the first frame and wait for a click
    cv2.imshow('Select Color on the Frame', frame)
    cv2.setMouseCallback('Select Color on the Frame', click_event, color_params)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q') or color_params['picked']:
            break

    cv2.destroyAllWindows()

    if not color_params['picked']:
        print("No color picked")
        return

    lower_range, upper_range = color_params['range']

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Calculate the distance threshold based on the video frame width
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    distance_threshold = frame_width / 15

    frame_centers = {}  # Dictionary to store centers frame by frame
    frame_id = 0

    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        centers_this_frame = []
        contours = find_color_clusters(frame, color_params['range'][0], color_params['range'][1])
        for contour in contours:
            center = calculate_center_of_mass(contour)
            if center:
                centers_this_frame.append(center)

        clustered_centers = cluster_centers(centers_this_frame, distance_threshold)
        clustered_centers.sort(key=lambda x: x[0])  # Sort centers based on x coordinate

        frame_centers[frame_id] = clustered_centers

        for center in clustered_centers:
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    return frame_centers


def save_to_json(frame_centers, json_file_path):
    data_to_save = {}

    for frame_id, centers in frame_centers.items():
        data_to_save[frame_id] = [
            {
                "robot_no": int(i),  # Ensure i is a native int
                "positions": {
                    "x": int(center[0]),  # Convert to native int
                    "y": int(center[1])   # Convert to native int
                }
            } 
            for i, center in enumerate(centers)
        ]

    with open(json_file_path, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)


def ensure_directories_exist(folder_path):
    processed_folder = os.path.join(folder_path, 'processed')
    json_folder = os.path.join(folder_path, 'json')

    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    return processed_folder, json_folder

def process_folder(folder_path):
    processed_folder, json_folder = ensure_directories_exist(folder_path)

    # Find all video files in the folder
    for video_file in glob.glob(os.path.join(folder_path, '*.mp4')):
        print(f"Processing {video_file}...")

        # Set up output paths
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        output_video_path = os.path.join(processed_folder, f'processed_{base_name}.mp4')
        output_json_path = os.path.join(json_folder, f'json_{base_name}.json')

        centers_list = []
        centers_per_frame = process_and_visualize_video(video_file, output_video_path, centers_list)
        if centers_per_frame:
            save_to_json(centers_per_frame, output_json_path)

centers_list = []

#put your folder name here
process_folder('vids')
