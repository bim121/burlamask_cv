import cv2
import dlib
import numpy as np

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(im):
    rects = DETECTOR(im, 1)
    
    if len(rects) > 1:
        raise Exception('Too many faces')
    if len(rects) == 0:
        raise Exception('Not enough faces')
    
    return np.array([[p.x, p.y] for p in PREDICTOR(im, rects[0]).parts()])

def transformation_from_points(points1, points2):
    """Return a transformation that maps points1 onto points2."""
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T @ points2)
    R = (U @ Vt).T

    return np.hstack([(s2 / s1) * R, (c2.T - (s2 / s1) * R @ c1.T)[:, None]])

def create_mask(points, shape, face_scale):
    """Create a mask of the given shape, using the given landmark points"""
    groups = [
        # indices of brow and eye landmarks
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43,
         44, 45, 46, 47],

        # indices of mouth and nose landmarks
        [27, 28, 29, 30, 31, 32, 33, 34, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
         58, 59, 60]
    ]

    mask_im = np.zeros(shape, dtype=np.float64)
    for group in groups:
        cv2.fillConvexPoly(mask_im,
                           cv2.convexHull(points[group]),
                           color=(1, 1, 1))

    feather_amount = int(0.2 * face_scale * 0.5) * 2 + 1
    kernel_size = (feather_amount, feather_amount)
    mask_im = (cv2.GaussianBlur(mask_im, kernel_size, 0) > 0) * 1.0  # dilate
    mask_im = cv2.GaussianBlur(mask_im, kernel_size, 0) # blur

    return mask_im

def correct_colours(warped_face_im, body_im, face_scale): 
    """Take the details from warped face im and the rest from body_im."""
    blur_amount = int(3 * 0.5 * face_scale) * 2 + 1
    kernel_size = (blur_amount, blur_amount)
    
    face_im_blur = cv2.GaussianBlur(warped_face_im, kernel_size, 0) 
    body_im_blur = cv2.GaussianBlur(body_im, kernel_size, 0)
    
    return np.clip(0. + body_im_blur + warped_face_im - face_im_blur, 0, 255)

face_im = cv2.imread('face1.jpg')
body_im = cv2.imread('face2.jpg')
face_points = get_landmarks(face_im)
body_points = get_landmarks(body_im)


# Зображення, в якому перше фото використовується як тіло, а друге - як обличчя
M = transformation_from_points(face_points, body_points)
warped_face_im = cv2.warpAffine(face_im, M, (body_im.shape[1], body_im.shape[0]))

face_scale = np.std(face_points)
corrected_face_im = correct_colours(warped_face_im, body_im, face_scale)

mask_im = create_mask(body_points, body_im.shape, face_scale)
combined_im = (warped_face_im * mask_im + body_im * (1 - mask_im))

cv2.imwrite('combined3.jpg', combined_im)

# Зображення, в якому перше фото використовується як обличчя, а друге - як тіло
M = transformation_from_points(body_points, face_points)
warped_face_im = cv2.warpAffine(body_im, M, (face_im.shape[1], face_im.shape[0]))

face_scale = np.std(face_points)
corrected_face_im = correct_colours(warped_face_im, face_im, face_scale)

mask_im = create_mask(face_points, face_im.shape, face_scale)
combined_im = (warped_face_im * mask_im + face_im * (1 - mask_im))

cv2.imwrite('combined4.jpg', combined_im)