import cv2
import numpy as np
import extcolors
from PIL import Image
from colormap import rgb2hex
from typing import Dict
from ultralytics import YOLO


class ColorExtractor:
    def __init__(self, w=900, tolerance=12, color_limit=4):
        self.output_width = w
        self.tolerance = tolerance
        self.limit = color_limit
        self.model = YOLO('best_n_20ep.pt')
        self.model_input_size = 640

    def predict(self, img: np.array):
        return self.model.predict(img, imgsz=self.model_input_size, conf=0.5, verbose=False)

    def getMask(self, img: np.array, pred):
        h, w, _ = img.shape

        if pred[0].masks is None:
            return None

        mask = pred[0].masks.data.cpu().numpy()[0]

        return mask

    def extractFromImageMaskPair(self, img: np.array, mask: np.array) -> Dict:
        # img resize
        img_h, img_w, _ = img.shape
        img = cv2.resize(img, dsize=(self.output_width, int(img_h * self.output_width / img_w)))

        # mask resize
        mask = cv2.resize(mask, dsize=(self.output_width, int(img_h * self.output_width / img_w)))
        _, mask = cv2.threshold(mask, 0.5, 1, type=cv2.THRESH_BINARY)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype('uint8')

        masked_img = img * mask
        masked_img = masked_img[:, :, ::-1]
        masked_img = Image.fromarray(masked_img)

        color_list, occ_sum = extcolors.extract_from_image(masked_img, tolerance=self.tolerance, limit=self.limit)

        hexs = {}
        for rgb_tuple, occur in color_list:
            r, g, b = rgb_tuple
            hex_value = rgb2hex(r, g, b)

            if hex_value == '#000000':
                occ_sum -= occur

            else:
                hexs[hex_value] = occur

        for k, v in hexs.items():
            hexs[k] = round(v / occ_sum * 100, 2)

        return hexs

    def getThumbnail(self, img, mask, pred, mode=0, bbox=True):
        """
        mode
        - 0: mask object fluorescent (opaque green)
        - 1: mask background (transparent black)
        - 2: mask background (transparent white)
        """
        img_h, img_w, _ = img.shape

        # mask resize
        mask = cv2.resize(mask, dsize=(img_w, img_h))
        _, mask = cv2.threshold(mask, 0.5, 1, type=cv2.THRESH_BINARY)
        mask = mask.astype('uint8')

        if mode == 0:  # fluorescent mask (opaque green)
            fluorescent_color = (0, 255, 0)
            transparency = 0.25

            hli = np.copy(img)
            hli[mask > 0] = ((1 - transparency) * hli[mask > 0] + transparency * np.array(fluorescent_color)).astype(np.uint8)

        elif mode == 1:
            mask = np.expand_dims(mask, axis=-1)
            hli = img * mask

        elif mode == 2:
            mask = np.expand_dims(mask, axis=-1)
            mask -= 1

            img = img.astype('uint16')
            mask = mask.astype('uint16')

            hli = img + mask
            hli = np.clip(hli, 0, 255)
            hli = hli.astype('uint8')

        if bbox:
            nx, ny, nxx, nyy = pred[0].boxes.xyxyn.cpu().numpy()[0]

            pt1 = (int(nx*img_w), int(ny*img_h))
            pt2 = (int(nxx*img_w), int(nyy*img_h))
            clr = (0, 255, 0)

            cv2.rectangle(hli, pt1, pt2, clr, 2)

        return hli


# Debug
if __name__ == "__main__":
    cer = ColorExtractor(color_limit=4)

    # img = cv2.imread('/Users/jeonghyojun/api/pet-color-api/uploads/dog-waiting-in-the-veterinarian-office.png')
    img = cv2.imread('/Users/jeonghyojun/api/pet-color-api/uploads/IMG_4340.png')

    pred = cer.predict(img)
    mask = cer.getMask(img, pred)
    thumbnail = cer.getThumbnail(img, mask, pred, mode=2, bbox=False)
    clr_info = cer.extractFromImageMaskPair(img, mask)

    cv2.imshow('temp', thumbnail)
    cv2.waitKey()
    cv2.destroyAllWindows()
