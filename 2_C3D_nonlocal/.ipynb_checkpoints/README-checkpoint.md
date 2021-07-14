# This is modified code from "C3D-Tensorflow-slim" <https://github.com/2012013382/C3D-Tensorflow-slim>
A simple Tensorflow code for C3D
## Dataset
UCF111. You need to place it in the root directory of the workspace.
## Usage
```Bash
sudo ./convert_video_to_images.sh UCF101/ 5
```
to convert videos into images(5FPS per-second).
```Bash
./convert_images_to_list.sh UCF101/ 4
```
to obtain train and test sets.(3/4 train; 1/4 test)
```Bash
python train.py
```
for training.
```
for testing.

## Attention
Radom choose 16 frames as a clip to represent a video for training and testing.
## Reference
The files convert_video_to_images.sh, convert_images_to_list.sh and crop_mean.npy are copied from https://github.com/hx173149/C3D-tensorflow
