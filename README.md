# Yolo3D
perform 3D object detection base on Yolov5 and 3DDeepbox. To transform to SNPE, change partial layers of yolov5,
such as change nn.Upsample to nn.ConvTranspose2d, and adjust Focus module in yolov5.


## Quick Start
### datasets
Applying kitt dataset. Place [kitti_dev](https://github.com/hitfee01/kitti_dev) sub contents to datasets/data/kitti/ of this project.

Please place it as following:

    root
    |
    ---datasets
       |
       ---data
          |
          ---kitti
              ---cache
                 |
                 ---k_*.npy // list K of camera. * -> (train or test)
                 |
                 ---label_*.npy // list label. * -> (train or test) 
                 |
                 ---shape_*.npy // list size of images. * -> (train or test)
              |
              ---ImageSets
                 |
                 ---train.txt // list of training image.
                 |
                 ---test.txt // list of testing image.
              ---testing
              |
              ---training
                 |
                 ---calib
                    |
                    ---calib_cam_to_cam.txt // camera calibration file for kitti
                 |
                 ---image_2
                 |
                 ---label_2
          

### training
    
    python train.py --data ./datasets/configs/kitti.yaml --cfg models/configs/yolo3d_5m.yaml --weights ./weights/yolov5m.pt --batch-size 64 --epochs 2000 --is-rect --is-mosaic --multi-scale --resume


### detect

    python detect.py --weights ./weights/model3d_5m_best_transconv_11_25.pt --device cpu --is-rect
    
### export onnx
    
    python export_onnx.py --weights ./weights/model3d_5m_best_transconv_11_25.pt --img-size 224 640 --batch-size 1
    
## Pretrained Model

We provide a set of trained models available for download in the  [Pretrained Model](https://pan.baidu.com/s/1-RQ0Dd_Kb-bwA2LhWitn8Q).
提取码: tpqg

## License
MIT

## Refenrence
yolov5.
3D-BoundingBox.