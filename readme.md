# SuperDrive
A live-processing capable, clean(-ish) implementation of lane & path detection based on comma.ai's SuperCombo neural network model

Tested with Python 3.8.2, OpenCV 4.2.0, and TensorFlow 2.2.0 on Ubuntu 20.04

[![SuperDrive Demo Video](https://img.youtube.com/vi/kE7CyqWYzKQ/0.jpg)](https://www.youtube.com/watch?v=kE7CyqWYzKQ)

### Running
To get a list of all the options, run `python3 drive.py -h`
To run the code with visualizations, run `python3 drive.py --input <input> --show-opencv-window`
To run this code with an actual camera, use something in `/dev/v4l/by-id/<camera>` for your `<input>` :)

### Note
I don't really know what I'm doing here, so this code *will probably* contain bugs. This is not production-quality by any means. While the model itself has proven to be quite successful at driving millions of miles elsewhere, that doesn't mean it'll translate to this quick implementation! Use with caution.
