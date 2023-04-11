import cv2
import os
from tqdm import tqdm
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

src_path = "pics/"
crop_path = "cropped/"
prepared_path = "prepared/"
ignored_no_face_path = "ignored_no_face/"
ignored_face_too_small_path = "ignored_face_too_small/"
ignored_image_too_small_path = "ignored_image_too_small/"


if not os.path.exists(crop_path):
    os.makedirs(crop_path)
if not os.path.exists(prepared_path):
    os.makedirs(prepared_path)
if not os.path.exists(ignored_no_face_path):
    os.makedirs(ignored_no_face_path)
if not os.path.exists(ignored_face_too_small_path):
    os.makedirs(ignored_face_too_small_path)
if not os.path.exists(ignored_image_too_small_path):
    os.makedirs(ignored_image_too_small_path)

files = os.listdir(src_path)
for filename in tqdm(files, desc="Processing images"):
    
    img = Image.open(src_path + filename)
    if min(img.size) >= 512:
        img.save(prepared_path + filename)
    else:
        img.save(ignored_image_too_small_path + filename)

        
        
files = os.listdir(prepared_path)
for filename in tqdm(files, desc="Detecting faces"):
    
    img = cv2.imread(prepared_path + filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=2, minSize=(200, 200))

    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            center_x = x + w // 2
            center_y = y + h // 2
            crop_size = int(max(w, h) / 0.7)
            crop_x1 = max(center_x - crop_size // 2, 0)
            crop_y1 = max(center_y - crop_size // 2, 0)
            crop_x2 = min(crop_x1 + crop_size, img.shape[1])
            crop_y2 = min(crop_y1 + crop_size, img.shape[0])
            if crop_x2 - crop_x1 >= 512 and crop_y2 - crop_y1 >= 512:
                crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_img = cv2.resize(crop_img, (512, 512))
                cv2.imwrite(crop_path + os.path.splitext(filename)[0] + "_" + str(i) + os.path.splitext(filename)[1], crop_img)
            else:
                cv2.imwrite(ignored_face_too_small_path + filename, img)
    else:
        cv2.imwrite(ignored_no_face_path + filename, img)
