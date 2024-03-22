import cv2
import numpy as np


def random_flip_image(
    image: np.ndarray, flip_prob_threshold: float = 0.5
) -> np.ndarray:
    """
    flip an image horizontally if random probability bigger than threshold.

    Args:
        image: an image to be flipped or not
        flip_prob_threshold: the probability threshold to flip image [0 ~ 1)
    Returns:
        randomly flipped image
    """
    if np.random.random() > flip_prob_threshold:
        image = image[:, ::-1]

    return image


def random_rotate_scale_image(
    image: np.ndarray, rotation_range: float = 0.05, scale_range: float = 0.01
) -> np.ndarray:
    """
    rotate and scaling an image randomly under given range.

    Args:
        image: an image to be rotated
        rotation_range: possible rotate angles in degrees[-rotation_range, rotation_range)
        scale_range: possible zoom scale range [1 - scale_range, 1 + scale_range]
                     0.0 means using original scale
    Returns:
        randomly warpped image
    """
    h, w = image.shape[:2]
    angle = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)

    M = cv2.getRotationMatrix2D((h // 2, w // 2), angle, scale)

    # ratation 후 가장자리 처리 - bordermode replicate 설정(주변 픽셀 값 복사)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
