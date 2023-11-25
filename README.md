# Trajectory-Extraction

Colab:https://colab.research.google.com/drive/1Z8yCl7Q6esqlDZZfhtI4xfWNPnJKPA9E?usp=sharing

# Video Processing Script

This script processes video files in a specified directory, detecting and tracking a user-selected color. For each video, it generates a processed copy with visual highlights on the tracked color and a JSON file containing the positional data of the tracked objects.

## Requirements

- Python 3.x
- OpenCV library (cv2)
- NumPy
- glob

## Installation

You can install the required Python libraries using pip:

```bash
pip install numpy opencv-python
```



Usage
Prepare Your Videos: Place all the .mp4 video files you want to process in a single directory. DONT FORGET TO RENAME THIS LINE WITH YOU PATH:

```
#put your folder name here
process_folder('vids')
```

Run the Script: Execute the script and follow the on-screen instructions to select the color to track in each video. The script will process each video file in the directory.

Processed Files: After running the script, check the processed and json subdirectories in your video folder. Processed videos will be in the processed folder, and the positional data JSON files will be in the json folder.

How It Works
The script scans the specified directory for .mp4 video files.
For each video file, it displays the first frame and waits for the user to click on a color to track.
After the color is selected, it processes the video, tracking the selected color and marking it with red dots.
The processed video is saved in the processed subdirectory.
Positional data of the tracked objects is saved as JSON in the json subdirectory.
Each JSON file contains frame-by-frame positional data of the tracked color, structured as follows:


```JSON
{
  "frame_id": [
    {
      "robot_no": 0,
      "positions": {
        "x": X-coordinate,
        "y": Y-coordinate
      }
    },
    ...
  ],
  ...
}
```




