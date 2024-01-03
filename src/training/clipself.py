import random
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

class CLIPSelf:
    def __call__(self, batch, model, dist_model, loss, device, cast_dtype, distributed, args):
        if distributed:
            model = model.module
            dist_model = dist_model.module
        images, normed_boxes, image_crops = batch       # note texts are not paired with images

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        normed_boxes = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
        image_crops = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)

        if args.multiscale:
            cur_h, cur_w = images.shape[2:]
            assert cur_h == cur_w
            if cur_h == 1024:
                tar_sizes = [320, 640, 896, 1024]
            elif cur_h == 896:
                tar_sizes = [336, 448, 672, 896]
            else:
                raise NotImplementedError
            tar_size = random.choice(tar_sizes)
            images = F.interpolate(images, size=(tar_size, tar_size), mode='bilinear')

        rois_list = []
        crops_list = []
        for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])
            crops_list.append(crops_per_image[valid])

        image_crops = torch.cat(crops_list)
        with torch.no_grad():
            teacher_crop_features = dist_model.encode_image(image_crops, normalize=False)
        student_roi_features = model.encode_pseudo_boxes(images, rois_list, normalize=False,
                                                         extract_type=args.extract_type,
                                                         window_size=args.window_size,
                                                         window_block_indexes=args.window_block_indexes)

        normed_student_features = F.normalize(student_roi_features, dim=-1)
        normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)

        loss_cosine = 1.0 - (normed_student_features *
                             normed_teacher_features).sum(-1).mean()
        losses = dict(loss_cosine=loss_cosine*args.cosine_weight)

        return losses, len(images), model.logit_scale.exp()


class CLIPSelfLlava:
    def __call__(self, batch, model, dist_model, loss, device, cast_dtype, distributed, args):
        if distributed:
            model = model.module
            dist_model = dist_model.module
        images, bboxes, image_crops = batch       # note texts are not paired with images

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        bboxes = bboxes.to(device=device, dtype=cast_dtype, non_blocking=True)
        image_crops = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)

        rois_list = []
        crops_list = []
        for bboxes_per_image, crops_per_image in zip(bboxes, image_crops):
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])
            crops_list.append(crops_per_image[valid])

        image_crops = torch.cat(crops_list)
        with torch.no_grad():
            teacher_crop_features = dist_model(image_crops).mean(dim=1)

        student_image_features = model(images)
        bs, sq_len, _ = student_image_features.shape
        h = w = int(sq_len ** 0.5)
        student_image_features = student_image_features.permute(0, 2, 1).contiguous().view(bs, -1, h, w)
        student_roi_features = roi_align(student_image_features,
                                         self._denormalize_boxes(rois_list, h, w),
                                         (1, 1),
                                         1.0, -1, True)[..., 0, 0]

        loss_l1 = F.l1_loss(student_roi_features, teacher_crop_features)
        losses = dict(loss_l1=loss_l1*args.l1_weight)

        return losses, len(images), torch.tensor(100.0).to(teacher_crop_features)


    @staticmethod
    def _denormalize_boxes(normed_boxes, h, w):
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes
