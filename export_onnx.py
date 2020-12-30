
"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse

import torch
import torch.nn as nn
from models.yolo import Model
import yaml
from utils import model_utils
from onnxsim import simplify


def _transform_weights_(m: nn.Module):
    if isinstance(m, torch.Tensor):
        print('type:%s, device:%s' % (m.dtype, m.device))
        m = m.cuda()

def _print_weights_(model):
    for k, v in model.named_parameters():
        print('name:%s, type:%s, device:%s' % (k, v.dtype, v.device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/model3d_5m_best_11_18.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[224, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    device = torch.device('cuda')
    # Input
    # img = torch.zeros((opt.batch_size, 3, *opt.img_size)).to(device)  # image size(1,3,320,192) iDetection
    img = torch.rand(1, 3, *opt.img_size).to(device)
    half = device.type != 'cpu'
    half = False
    # Load PyTorch model
    # ckpt = torch.load(opt.weights, map_location=lambda storage, loc: storage.cuda())['model'].float()
    with open('./models/configs/yolo3d_5m.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg.update(opt.__dict__)
        print(cfg)
    model = Model(cfg)
    model.eval()
    checkpointer = model_utils.CheckPointer(model, device=device)
    checkpointer.load(opt.weights, load_solver=False)
    # model.to(device)
    if half:
        model.half()  # to FP16
        img = img.half()
    else:
        model.to(torch.float32)
    model.to(device)
    # _print_weights_(model)
    _ = model(img)  # dry run
    model.model[-1].export = True  # set Detect() layer export=True
    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.fuse()  # only for ONNX
        model.to(device)
        model.to(torch.float32)
        # _print_weights_(model)
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx_model, check = simplify(onnx_model)
        onnx.checker.check_model(onnx_model)  # check onnx model
        onnx.save(onnx_model, f)
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # # CoreML export
    # try:
    #     import coremltools as ct
    #
    #     print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
    #     # convert model from torchscript and apply pixel scaling as per detect.py
    #     model = ct.convert(ts, inputs=[ct.ImageType(name='images', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
    #     f = opt.weights.replace('.pt', '.mlmodel')  # filename
    #     model.save(f)
    #     print('CoreML export success, saved as %s' % f)
    # except Exception as e:
    #     print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')

