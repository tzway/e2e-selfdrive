# an Implementation of NVIDIA PilotNet

Reference:
https://github.com/rslim087a/Self-Driving-Car-Course-Codes
https://github.com/rslim087a/track

### Model Input and Output
Input: H×W×C 66 * 200 * 3 Raw pixel values [0, 255] linearly scaled to [0, 1]
Output: Steering angle as real-valued scalar

### development environment setup
use devcontainer

### To Test Self-Driving in Simulator
run `python3 drive_torch.py` or `python3 drive_tf.py`to deploy the self driving ws server

run the simulator executable in auto mode and it will connect to server automatically

### Model Training Process
two ipynb files are two standalone implementations of training using tensorflow or pytorch

compare.ipynb shows the performance difference of using and not using 2 side cams for data augmentation

### issues
if the training crashes, try increasing the shm size of devcontainer