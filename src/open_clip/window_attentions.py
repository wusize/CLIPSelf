import torch
# import torch.nn.functional as F


def window_partition(x, window_size=16):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    # 1024 -> 64 -> 16x4
    # x: bs, 1 + h*w, c
    x_cls = x[:, :1]   # bs, 1, c
    x = x[:, 1:]
    B, H_W, C = x.shape
    H = W = int(H_W ** 0.5)
    x = x.view(B, H, W, C)

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    assert pad_h == 0 and pad_w == 0, f"For now we do not allow additional padding"
    # if pad_h > 0 or pad_w > 0:
    #     x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    x_cls = x_cls.repeat(1, Hp*Wp // int(window_size**2), 1).view(-1, 1, C)
    windows = torch.cat([x_cls, windows], dim=1)

    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, 1+window_size*window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, 1+H*W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw

    windows_cls = windows[:, :1]    # cls tokens
    windows = windows[:, 1:]

    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp*Wp, -1)

    windows_cls = windows_cls.view(B, Hp*Wp // int(window_size**2), 1, -1)

    x_cls = windows_cls.mean(1)   # average all cls tokens
    x = torch.cat([x_cls, x], dim=1)

    assert Hp == H and Wp == W, "For now we do not allow additional padding"
    # if Hp > H or Wp > W:
    #     x = x[:, :H, :W, :].contiguous()
    return x
