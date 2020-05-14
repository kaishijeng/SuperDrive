# SuperDrive
A live-processing capable, clean(-ish) implementation of lane & path detection based on comma.ai's SuperCombo neural network model

### Running
A simple call of `python3 drive.py <input>` will do. `<input>` can be anything... from an input file to a capture device. Whatever is supported in OpenCV's `VideoCapture()` function *should* also work here!

### Note
I don't really know what I'm doing here, so this code *will probably* contain bugs (as of this first push, it still doesn't really work and is awfully slow), please don't refer to it until everything has been squared out!
