## Nvidia Driver Change
- The Nvidia Display Driver(which includes the CUDA Driver needed from CUDA runtime computation) is upgraded to 470, which support CUDA Toolkit version upto 11.4. I believe it is backward compatible (my Ubuntu runs Display Driver 535 that compatible to 12.4, but I successfully compiled DSP-SLAM with CUDA Toolkit 10.2 and 11.3).
- Is better to [disable the neovue driver](https://askubuntu.com/questions/841876/how-to-disable-nouveau-kernel-driver) to avoid black screen when changing Nvidia proprietary driver

## Changes to`~/.profile` and  `~/.bashrc`
1. Added this to `~/.profile` 
```bash
export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
If other project need other version of `nvcc`, then download the [specific version](https://developer.nvidia.com/cuda-toolkit-archive) of CUDA Toolkit. Follow the TUI steps. Install only the Toolkit (Driver not needed as long as the Display driver is a higher version). Restart the system. Any Pytorch used need to compatible with the version set here.

2. Commented out the `PATH` and `LD_LIBRARY_PATH` lines, as it's duplicate from the `.profile`'s setting.
3. It's necessary to set the env. vars. in `.profile`. Just setting them in `.bashrc` will not work.

## DSP-SLAM Installation Steps
1. Add `-ltiff` to `CMakeList.txt`
```
target_link_libraries(${PROJECT_NAME}
        -ltiff
        pybind11::embed
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)
```
1. Ran `build_cuda113.sh` with all the options
```
./build_cuda113.sh --install-cuda --build-dependencies --create-conda-env
```
2. Download the `data` and `weight` folder from the sharepoint.

## PointFlow Integration and how to run example
1. Added the PointFlow python codes into the project (minor difference from the original ones on GitHub)
2. The car model weight is copied to `weight/pointflow/checkpoint-612.pt`. (612 epochs)
3. Run the `reconstruct_frame_point_flow.py`
```
python reconstruct_frame_point_flow.py --config configs/config_kitti.json --sequence_dir data/kitti/07 --frame_id 100 --cates car --resume_checkpoint weights/pointflow/checkpoint-612.pt --dims 512-512-512 --use_deterministic_encoder --evaluate_recon --batch_size 1
```
4. Not sure why. Might because I was remote accessing the computer, the `DISPLAY` environment variable was not set. Manually setting it worked. Set before running any test.
```
export DISPLAY=:1
```
