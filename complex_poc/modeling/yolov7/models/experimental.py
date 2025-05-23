import torch
import torch.nn as nn

from modeling.yolov7.models.common import Conv


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models.
    # weights can be a list of model paths or a single path.
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # Since the weight file is assumed to be local,
        # we directly load it without calling attempt_download.
        ckpt = torch.load(
            w, map_location=map_location, weights_only=False
        )  # load checkpoint from local path

        # Depending on if 'ema' exists in the checkpoint, select the corresponding model.
        model_to_append = ckpt["ema" if ckpt.get("ema") else "model"]
        model.append(
            model_to_append.float().fuse().eval()
        )  # convert to FP32, fuse layers, and set eval mode

    # Compatibility updates for various layers.
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # Enable in-place operations for better memory efficiency (pytorch 1.7.0 compatibility)
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # Update for torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # For pytorch 1.6.0 compatibility

    # If only one model was loaded, return it directly.
    if len(model) == 1:
        return model[-1]
    else:
        print("Ensemble created with %s\n" % weights)
        # In case of an ensemble, add selected properties from the last model.
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model  # return the ensemble
