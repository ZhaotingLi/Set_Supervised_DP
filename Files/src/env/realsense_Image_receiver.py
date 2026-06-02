import logging

logger = logging.getLogger(__name__)

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
# from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class ImageReceiver:
    def __init__(self, topic, enable_crop = False):
        # self.bridge = CvBridge()
        self.topic = topic
        self.subscriber = rospy.Subscriber(self.topic, CompressedImage, self.callback)
        logger.debug(f"Subscribed to {self.topic}")
        self.target_width  = 320
        self.target_height = 240

        self.image = np.zeros((self.target_height, self.target_width, 3))

        h_max, w_max = 480, 640
        x = 120
        y = 80
        hw = w_max - 2 * x
        hh = h_max - 2 * y
        self.crop_roi = (x, y, hw, hh)
        self.enable_crop = enable_crop

    def callback(self, data):
        # try:
        # Convert compressed ROS image message to OpenCV image
        np_arr = np.frombuffer(data.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Crop if enabled
        if self.enable_crop:
            x, y, w, h = self.crop_roi
            # Ensure ROI within image bounds
            h_max, w_max = cv_image.shape[:2]
            # print("cv_image shape: ", cv_image.shape[:2])
            x = max(0, min(x, w_max-1))
            y = max(0, min(y, h_max-1))
            w = min(w, w_max - x)
            h = min(h, h_max - y)
            cv_image = cv_image[y:y+h, x:x+w]

        cv_image = cv2.resize(
            cv_image,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )
        self.image = cv_image
        # print("shape of np_arr: ", np_arr.shape) # (47245,)

        # print("shape of cv_image: ", cv_image.shape) # (240, 320, 3)
        
        # except CvBridgeError as e:
        #     rospy.logerr(f"CvBridge Error: {e}")

if __name__ == '__main__':
    # receiver = ImageReceiver('/camera/color/image_raw/compressed')
    rospy.init_node('compressed_image_listener1', anonymous=True)
    receiver = ImageReceiver('/camera1/color/image_raw/compressed',  enable_crop=False)
    receiver_img2 = ImageReceiver('/camera2/color/image_raw/compressed', enable_crop=False)


    rate = rospy.Rate(30)  # 30 Hz is more than enough for display
    try:
        while not rospy.is_shutdown():
            image1 = receiver.image
            image2 = receiver_img2.image

            cv2.imshow('Camera 1', image1)
            cv2.imshow('Camera 2', image2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rate.sleep()
    except KeyboardInterrupt:
        logger.debug("Shutting down")
    finally:
        cv2.destroyAllWindows()


# import rospy
# from sensor_msgs.msg import CompressedImage
# import cv2
# import numpy as np

# import re

# PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# def _find_png_start(buf: bytes) -> int:
#     """Return index of PNG magic or -1 if not found."""
#     return buf.find(PNG_MAGIC)

# def _parse_quant_params(fmt: str):
#     """
#     Extract quantization params for compressedDepth 32FC1.
#     Returns (A, B, inv_depth_flag) or (None, None, False) if not found.
#     """
#     if not fmt:
#         return None, None, False
#     fmt_l = fmt.lower()
#     mA = re.search(r"depthquanta\s*[:=]\s*([0-9eE\.\+\-]+)", fmt_l)
#     mB = re.search(r"depthquantb\s*[:=]\s*([0-9eE\.\+\-]+)", fmt_l)
#     inv = False
#     mInv = re.search(r"inv[_\s]*depth\s*[:=]\s*([01truefals]+)", fmt_l)
#     if mInv:
#         v = mInv.group(1)
#         inv = v in ("1", "true", "yes")
#     A = float(mA.group(1)) if mA else None
#     B = float(mB.group(1)) if mB else None
#     return A, B, inv

# class ImageReceiver:
#     def __init__(self, color_topic, depth_topic=None, enable_crop=False, name="cam"):
#         """
#         color_topic: str (compressed RGB, e.g. '/camera/color/image_raw/compressed')
#         depth_topic: Optional[str] (compressedDepth, e.g. '/camera/aligned_depth_to_color/image_raw/compressedDepth')
#                       If None, depth subscription is disabled.
#         """
#         self.name = name
#         self.color_topic = color_topic
#         self.depth_topic = depth_topic
#         self.enable_crop = enable_crop

#         # target display size (also used to resize depth for visualization)
#         self.target_width  = 320
#         self.target_height = 240

#         # Buffers
#         self.color_bgr = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

#         # Depth-related buffers are optional
#         self.depth_enabled = self.depth_topic is not None
#         self.depth_m = None           # float32 meters
#         self.depth_vis = None         # uint8 BGR for imshow

#         # Default crop ROI assuming 640x480 input
#         h_max, w_max = 480, 640
#         x, y = 120, 80
#         hw = w_max - 2 * x
#         hh = h_max - 2 * y
#         self.crop_roi = (x, y, hw, hh)

#         # Subscribers
#         self.sub_color = rospy.Subscriber(
#             self.color_topic, CompressedImage, self._cb_color, queue_size=1, buff_size=2**20
#         )
#         rospy.loginfo(f"[{self.name}] Subscribed to color: {self.color_topic}")

#         if self.depth_enabled:
#             self.sub_depth = rospy.Subscriber(
#                 self.depth_topic, CompressedImage, self._cb_depth, queue_size=1, buff_size=2**20
#             )
#             rospy.loginfo(f"[{self.name}] Subscribed to depth: {self.depth_topic}")
#         else:
#             self.sub_depth = None
#             rospy.loginfo(f"[{self.name}] Depth subscription disabled (depth_topic=None)")

#     def _apply_crop(self, img):
#         if not self.enable_crop:
#             return img
#         x, y, w, h = self.crop_roi
#         h_max, w_max = img.shape[:2]
#         x = max(0, min(x, w_max - 1))
#         y = max(0, min(y, h_max - 1))
#         w = max(1, min(w, w_max - x))
#         h = max(1, min(h, h_max - y))
#         return img[y:y+h, x:x+w]

#     def _cb_color(self, msg: CompressedImage):
#         try:
#             np_arr = np.frombuffer(msg.data, np.uint8)
#             img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR
#             if img is None:
#                 return
#             img = self._apply_crop(img)
#             img = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
#             self.color_bgr = img
#         except Exception as e:
#             rospy.logwarn(f"[{self.name}] Color decode error: {e}")

#     def _cb_depth(self, msg: CompressedImage):
#         """
#         Decode RealSense 'compressedDepth' and visualize like the official
#         OpenCV viewer example: use cv2.convertScaleAbs(..., alpha=0.03) followed
#         by cv2.applyColorMap(..., JET). Also fills self.depth_m (meters).
#         """
#         try:
#             buf = msg.data  # bytes
#             idx = _find_png_start(buf)
#             if idx == -1:
#                 rospy.logwarn(f"[{self.name}] No PNG header found in compressedDepth buffer.")
#                 return

#             depth_png = np.frombuffer(buf[idx:], np.uint8)
#             depth_raw = cv2.imdecode(depth_png, cv2.IMREAD_UNCHANGED)  # expect 16UC1 (Z16)
#             if depth_raw is None:
#                 rospy.logwarn(f"[{self.name}] cv2.imdecode returned None for depth.")
#                 return

#             # Preserve raw values when cropping/resizing
#             depth_raw = self._apply_crop(depth_raw)
#             depth_raw = cv2.resize(
#                 depth_raw, (self.target_width, self.target_height),
#                 interpolation=cv2.INTER_NEAREST
#             )

#             # Keep a meters copy for downstream use
#             if depth_raw.dtype == np.uint16:  # typical RealSense Z16 in millimeters
#                 self.depth_m = depth_raw.astype(np.float32) * 1e-3  # mm -> m
#                 # RealSense example visualization:
#                 depth_8u = cv2.convertScaleAbs(depth_raw, alpha=0.03)
#             else:
#                 # If a float32 depth arrives (meters), scale similarly (~0.03 on mm == 30 on meters)
#                 self.depth_m = depth_raw.astype(np.float32)
#                 depth_8u = cv2.convertScaleAbs(self.depth_m, alpha=30.0)

#             self.depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

#         except Exception as e:
#             rospy.logwarn(f"[{self.name}] Depth decode error: {e}")


#     @staticmethod
#     def _make_depth_vis(depth_m: np.ndarray, vmin=0.2, vmax=3.0):
#         d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
#         d = np.clip(d, vmin, vmax)
#         norm = (d - vmin) / max(1e-6, (vmax - vmin))
#         norm_u8 = (norm * 255.0).astype(np.uint8)
#         vis = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
#         vis[d <= 0.0] = (0, 0, 0)
#         return vis


# if __name__ == '__main__':
#     rospy.init_node('rs_dual_image_listener', anonymous=True)

#     # Example: cam1 with depth, cam2 without depth
#     cam1 = ImageReceiver(
#         color_topic='/camera1/color/image_raw/compressed',
#         # depth_topic='/camera/depth/image_rect_raw/compressedDepth',
#         enable_crop=False,
#         name='camera1'
#     )
#     cam2 = ImageReceiver(
#         color_topic='/camera2/color/image_raw/compressed',
#         depth_topic=None,  # <- no depth for camera 2
#         enable_crop=True,
#         name='camera2'
#     )

#     rate = rospy.Rate(30)
#     try:
#         while not rospy.is_shutdown():
#             # RGB
#             cv2.imshow('Camera 1 (RGB)', cam1.color_bgr)
#             cv2.imshow('Camera 2 (RGB)', cam2.color_bgr)

#             # Depth (only if enabled and available)
#             if cam1.depth_enabled and cam1.depth_vis is not None:
#                 cv2.imshow('Camera 1 (Depth)', cam1.depth_vis)
#             # if cam2.depth_enabled and cam2.depth_vis is not None:
#             #     cv2.imshow('Camera 2 (Depth)', cam2.depth_vis)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             rate.sleep()
#     except KeyboardInterrupt:
#         print("Shutting down")
#     finally:
#         cv2.destroyAllWindows()
