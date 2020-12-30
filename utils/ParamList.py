import torch
import numpy as np

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class ParamList(object):
    """
    This class represents labels of specific object.
    """

    def __init__(self, image_size, is_training=True):
        self.size = image_size
        self.is_training = is_training
        self.extra_fields = {}

    def add_field(self, field, field_data, to_tensor=False):
        field_data = field_data if isinstance(field_data, torch.Tensor) else \
            torch.as_tensor(field_data) if to_tensor else field_data
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def update_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def merge_field(self, field, field_data, axis=0):
        if not self.has_field(field):
            self.add_field(field, field_data)
        else:
            v = self.get_field(field)
            if isinstance(v, torch.Tensor):
                assert v.ndim == field_data.ndim
                mv = v.clone()
                mv = torch.cat([mv, field_data], dim=axis)
                self.update_field(field, mv)
            elif isinstance(v, np.ndarray):
                v = np.concatenate([v, field_data], axis=axis)
                self.update_field(field, v)

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def merge(self, params, axis=0):
        if isinstance(params, ParamList):
            for k, v in params.extra_fields.items():
                self.merge_field(k, v, axis=axis)

    def _copy_extra_fields(self, target):
        for k, v in target.extra_fields.items():
            self.add_field(k, v.clone() if isinstance(v, torch.Tensor) else np.copy(v))

    def to(self, device):
        target = ParamList(self.size, self.is_training)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            target.add_field(k, v)
        return target

    def to_tensor(self):
        for k, v in self.extra_fields.items():
            self.update_field(k, torch.as_tensor(v))

    def to_float32(self):
        for k, v in self.extra_fields.items():
            if isinstance(v, torch.Tensor):
                v = v.to(torch.float32)

            elif isinstance(v, np.ndarray):
                v = v.astype(np.float32)
            self.update_field(k, v)

    def to_float16(self):
        for k, v in self.extra_fields.items():
            if isinstance(v, torch.Tensor):
                v = v.to(torch.float16)
            elif isinstance(v, np.ndarray):
                v = v.astype(np.float16)
            self.update_field(k, v)

    def __len__(self):
        if self.is_training:
            reg_num = len(torch.nonzero(self.extra_fields["mask"], as_tuple=True))
        else:
            reg_num = 0
        return reg_num

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "regress_number={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s

    def delete_by_mask(self):
        if self.has_field('mask'):
            mask = self.get_field('mask')
            mask = mask.clone() if isinstance(mask, torch.Tensor) else np.copy(mask)
            mask = torch.nonzero(mask, as_tuple=True)[0] if isinstance(mask, torch.Tensor) else np.nonzero(mask)[0]
            for k, v in self.extra_fields.items():
                v = v[mask]
                self.update_field(k, v)

    def copy_from(self, target):
        self.size = target.size
        self.is_training = target.is_training
        self._copy_extra_fields(target)

    def numpy(self):
        c = ParamList(None)
        c.copy_from(self)
        for k, v in c.extra_fields.items():
            if isinstance(v, torch.Tensor):
                c.update_field(k, v.cpu().numpy())
        return c
