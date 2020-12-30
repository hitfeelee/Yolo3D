import argparse
from models.experimental import *
import yaml
from utils import torch_utils
from utils import utils
from models import losses
from models.enconder_decoder import Coder
from torch.nn import functional as F
from postprocess import postprocess


def parse_model(md, ch):  # model_dict, input_channels(3)
    print('\n%3s%15s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    detect_strides, anchors, num_classes, gd, gw = md['detect_strides'], md['anchors'], md['num_classes'], \
                                                   md['depth_multiple'], md['width_multiple']
    na = (len(anchors[0]) // 2)  # number of anchors
    no = na * (num_classes + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(md['backbone'] + md['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        mn = utils.get_class_name(m)
        if mn in ['Conv2d', 'Conv', 'Bottleneck', 'SPP', 'DWConv', 'MixConv2d', 'Focus', 'ConvPlus', 'BottleneckCSP', 'UpSample']:
            c1, c2 = ch[f], args[0]
            c2 = utils.make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]
            if mn == 'BottleneckCSP':
                args.insert(2, n)
                n = 1
        elif mn in ['BatchNorm2d']:
            args = [ch[f]]
        elif mn == 'Concat':
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif mn == 'Detect':
            f = f or list(reversed([(-1 if j == i else j - 1) for j, x in enumerate(ch) if x == no]))
            c1 = [ch[-1 if j == -1 else j + 1] for j in f]
            args = [md, c1] + args
            # args.insert(1, c1)
            c2 = no
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%15s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class Model(nn.Module):
    def __init__(self, cfg, ch=3):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.md = cfg
        # Define model
        self.model, self.save = parse_model(self.md, ch=[ch])  # model, savelist, ch_out
        # Init weights, biases
        torch_utils.initialize_weights(self)
        # torch_utils.model_info(self)
        print('')

    def forward(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for f in sorted([x % m.i for x in m.f]):  #  from
            b = self.model[f].bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%g Conv2d.bias:' + '%10.3g' * 6) % (f, *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers...')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        # torch_utils.model_info(self)


class Detect(nn.Module):
    def __init__(self, cfg, in_ches=(), export=False):  # detection layer
        super(Detect, self).__init__()
        self.num_classes = cfg['num_classes']  # number of classes
        self.num_bbox_header_outputs = self.num_classes + 5  # number of outputs per anchor

        self.num_layers = len(cfg['anchors'])  # number of detection layers
        self.num_anchors = len(cfg['anchors'][0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.num_layers  # init grid
        # a = torch.tensor(cfg['anchors']).float().view(self.num_layers, -1, 2)
        self.anchors = torch.tensor(cfg['anchors']).float().view(self.num_layers, -1, 2)  # shape(num_layers,num_anchors,2)
        self.in_ches = in_ches
        self.in_strides = torch.tensor(cfg['detect_strides'])
        self.export = export  # onnx export
        self.bbox_headers = nn.ModuleList(nn.Sequential(nn.Conv2d(x, x, 3, padding=1),
                                                        nn.Conv2d(x, self.num_anchors * self.num_bbox_header_outputs, 1)
                                                        ) for x in in_ches)  # output conv
        self.encoder_decoder = Coder(cfg['dim_ref'])
        self.num_bbox3d_header_outputs = self.encoder_decoder.multibin.bin_num * 3 + 3  #(bin_num + bin_num*2 + dim_num)
        self.bbox3d_headers = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, x, 3, padding=1),
                          nn.Conv2d(x, self.num_anchors * self.num_bbox3d_header_outputs, 1)
                          ) for x in in_ches)  # output conv
        self._initialize_biases()

    def forward(self, x):
        x1, x2 = [None]*self.num_layers, [None]*self.num_layers
        for i in range(self.num_layers):
            x1[i] = self.bbox_headers[i](x[i])  # conv
            bs, _, ny, nx = x1[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x1[i] = x1[i].view(bs, self.num_anchors, -1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x2[i] = self.bbox3d_headers[i](x[i])  # conv
            x2[i] = x2[i].view(bs, self.num_anchors, -1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return [torch.cat([k, m], dim=-1) for k, m in zip(x1, x2)]
        elif self.export:
            return [torch.cat([k, m], dim=-1) for k, m in zip(x1, x2)]#, self.inference(x1, x2)[0]
        else:
            return self.inference(x1, x2)

    def inference(self, pred2d_logits, pred3d_logits):
        device = pred2d_logits[0].device
        preds = []
        anchors = self.anchors.clone().to(device)
        anchors = anchors.view(self.num_layers, 1, -1, 1, 1, 2)
        in_strides = self.in_strides.to(device)
        for i in range(self.num_layers):
            bs, _, ny, nx, _ = pred2d_logits[i].shape
            pred2d = pred2d_logits[i].sigmoid().view(bs, -1, ny, nx, self.num_bbox_header_outputs)
            pred3d = pred3d_logits[i].view(bs, -1, self.num_bbox3d_header_outputs)
            bin_num = self.encoder_decoder.multibin.bin_num

            if not self.export:
                if self.grid[i].shape[2:4] != pred2d_logits[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(device)
                pred2d[..., 0:2] = (pred2d[..., 0:2] * 2. - 0.5 + self.grid[i]) * in_strides[i]  # xy
                pred2d[..., 2:4] = ((pred2d[..., 2:4] * 2) ** 2) * anchors[i]  # wh
                bin_orient = pred3d[..., bin_num:bin_num + bin_num * 2].view(bs, -1, bin_num, 2)
                pred3d[..., bin_num:bin_num + bin_num * 2] = F.normalize(bin_orient, dim=-1).view(bs, -1, bin_num * 2)
                pred3d[..., -3:] = pred3d[..., -3:].sigmoid() * 2 - 1.
            else:
                pred2d_xy = (pred2d[..., 0:2] * 2. - 0.5 + self.grid[i]) * in_strides[i]
                pred2d_wh = ((pred2d[..., 2:4] * 2) ** 2) * anchors[i]  # wh
                pred2d_giou_class = pred2d[..., 4:]
                pred2d = torch.cat([pred2d_xy, pred2d_wh, pred2d_giou_class], dim=-1)

                pred3d_bins = pred3d[..., 0:bin_num]
                pred3d_bin_orient = pred3d[..., bin_num:bin_num + bin_num * 2].view(bs, -1, bin_num, 2)
                pred3d_bin_orient = F.normalize(pred3d_bin_orient, dim=-1).view(bs, -1, bin_num * 2)
                pred3d_dim = pred3d[..., -3:].sigmoid() * 2 - 1.
                pred3d = torch.cat([pred3d_bins, pred3d_bin_orient, pred3d_dim], dim=-1)
            p = torch.cat([pred2d.view(bs, -1, self.num_bbox_header_outputs), pred3d], dim=-1)
            preds.append(p)
        preds = torch.cat(preds, 1)
        preds = postprocess.apply_nms_onnx(preds, self.num_classes, 0.4, 0.5) if self.export else preds
        return preds, [torch.cat([k, m], dim=-1) for k, m in zip(pred2d_logits, pred3d_logits)]

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        self.bbox_headers.apply(_header_init_)
        for mi, si in zip(self.bbox_headers, self.in_strides.numpy()):
            b = mi[-1].bias.view(self.num_anchors, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / si) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (self.num_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        self.bbox3d_headers.apply(_header_init_)
        for mi, si in zip(self.bbox3d_headers, self.in_strides.numpy()):
            b = mi[-1].bias.view(self.num_anchors, -1)  # conv.bias(255) to (3,85)
            b[:, :self.encoder_decoder.multibin.bin_num] += math.log(0.6 / (self.encoder_decoder.multibin.bin_num - 0.99))  # cls
            mi[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def _header_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, 0.)


def fuse(model):
    for m in model.modules():
        if type(m) is Conv:
            m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
            m.bn = None  # remove batchnorm
            m.forward = m.fuseforward  # update forward


import onnx
from onnxsim import simplify

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./models/configs/yolo3d_5m.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = utils.check_file(opt.cfg)  # check file
    device = 'cpu'
    with open(opt.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg.update(opt.__dict__)
        print(cfg)
    # Create model
    model = Model(cfg)
    ckpt = torch.load('./weights/yolov5m.pt', map_location=torch.device(device))['model'].float().state_dict()
    state = model.state_dict()
    # model = UpSample(3, None, 2)
    model.eval()
    model.to(device)
    for k, v in model.named_parameters():
        print('name:%s, type:%s, device:%s' % (k, v.dtype, v.device))

    img = torch.rand(1, 3, 224, 640).to(device)
    y = model(img)

    # ONNX export
    model.model[-1].export = True
    try:
        fuse(model)
        model.to(device)
        f = opt.cfg.replace('.yaml', '.onnx')
        torch.onnx.export(model, img, f, verbose=False, opset_version=7, input_names=['images'],
                          output_names=['output'])
        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        # onnx_model, check = simplify(onnx_model)
        # onnx.checker.check_model(onnx_model)  # check onnx model
        # assert check, "fail to simplify onnx model"
        # onnx.save(onnx_model, f)
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('transform successfully')
    except Exception as e:
        print('ONNX export failure: %s' % e)
    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
