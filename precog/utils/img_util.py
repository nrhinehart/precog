
def batch_center_crop(img, target_h, target_w):
    *_, h, w, _ = img.shape
    center = (h // 2, w // 2)
    return img[..., center[0] - target_h // 2:center[0] + target_h // 2, center[1] - target_w // 2:center[1] + target_w // 2, :]
