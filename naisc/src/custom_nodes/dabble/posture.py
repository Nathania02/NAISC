"""
Node template for creating custom nodes.
"""
from typing import Any, Dict, List, Tuple
import cv2
from peekingduck.pipeline.nodes.abstract_node import AbstractNode

# setup global constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)       # opencv loads file in BGR format
YELLOW = (0, 255, 255)
NAVY = (128, 0, 0) 
THRESHOLD = 0.6               # ignore keypoints below this threshold
KP_RIGHT_SHOULDER = 6         # PoseNet's skeletal keypoints
KP_LEFT_SHOULDER = 5
KP_RIGHT_ELBOW = 8
KP_LEFT_ELBOW = 7
KP_RIGHT_KNEE = 14
KP_LEFT_KNEE = 13
KP_LEFT_WRIST =9 
KP_RIGHT_WRIST = 10 

def map_keypoint_to_image_coords(
   keypoint: List[float], image_size: Tuple[int, int]
) -> List[int]:
   """Second helper function to convert relative keypoint coordinates to
   absolute image coordinates.
   Keypoint coords ranges from 0 to 1
   where (0, 0) = image top-left, (1, 1) = image bottom-right.

   Args:
      bbox (List[float]): List of 2 floats x, y (relative)
      image_size (Tuple[int, int]): Width, Height of image

   Returns:
      List[int]: x, y in integer image coords
   """
   width, height = image_size[0], image_size[1]
   x, y = keypoint
   x *= width
   y *= height
   return int(x), int(y)

def draw_text_coordinates(img, x, y, text_str: str, color_code):
   """Helper function to call opencv's drawing function,
   to improve code readability in node's run() method.
   """
   cv2.putText(
      img=img,
      text=text_str,
      org=(x, y),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=0.4,
      color=color_code,
      thickness=2,
   )

def draw_text(img, x, y, text_str: str, color_code):
   """Helper function to call opencv's drawing function,
   to improve code readability in node's run() method.
   """
   cv2.putText(
      img=img,
      text=text_str,
      org=(x, y),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=1,
      color=color_code,
      thickness=2,
   )


class Node(AbstractNode):
   """Custom node to display keypoints and count number of hand waves

   Args:
      config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
   """

   def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
      super().__init__(config, node_path=__name__, **kwargs)

class Node(AbstractNode):
   """Custom node to display keypoints and count number of hand waves

   Args:
      config (:obj:Dict[str, Any] | :obj:None): Node configuration.
   """

   def init(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
      super().init(config, node_path=__name__, **kwargs)
      # setup object working variables
      self.right_wrist = None
      self.direction = None
      self.num_direction_changes = 0
      self.num_waves = 0

   def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
      """This node draws keypoints and count hand waves.

      Args:
            inputs (dict): Dictionary with keys
               "img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores".

      Returns:
            outputs (dict): Empty dictionary.
      """

      # get required inputs from pipeline
      img = inputs["img"]
      keypoints = inputs["keypoints"]
      keypoint_scores = inputs["keypoint_scores"]

      img_size = (img.shape[1], img.shape[0])  # image width, height

      # hand wave detection using a simple heuristic of tracking the
      # right wrist movement
      the_keypoints = keypoints[0]              # image only has one person
      the_keypoint_scores = keypoint_scores[0]  # only one set of scores

      coordinates = {}

      for i, keypoints in enumerate(the_keypoints):
         keypoint_score = the_keypoint_scores[i]

         if keypoint_score >= THRESHOLD:
            x, y = map_keypoint_to_image_coords(keypoints.tolist(), img_size)
            x_y_str = f"({x}, {y})"

            if i == KP_LEFT_ELBOW:
               left_elbow = keypoints
               coordinates["Left_Elbow"] = left_elbow
               the_color = YELLOW

            elif i == KP_RIGHT_ELBOW:
               right_elbow = keypoints
               coordinates["Right_Elbow"] = right_elbow
               the_color = YELLOW

            elif i == KP_RIGHT_SHOULDER:
               right_shoulder = keypoints
               coordinates["Right_Shoulder"] = right_shoulder

            elif i == KP_LEFT_SHOULDER:
               left_shoulder = keypoints
               coordinates["Left_Shoulder"] = left_shoulder

            elif i == KP_RIGHT_WRIST:
               right_wrist = keypoints
               coordinates["Right_Wrist"] = right_wrist

            elif i == KP_LEFT_WRIST:
               left_wrist = keypoints
               coordinates["Left_Wrist"] = left_wrist

            else:                   # generic keypoint
               the_color = WHITE

            import numpy as np
            if len(coordinates)>=6:
               a = coordinates["Left_Wrist"]
               b = coordinates["Left_Elbow"]
               c = coordinates["Left_Shoulder"]
               d = coordinates["Right_Wrist"]
               e = coordinates["Right_Elbow"]
               f = coordinates["Right_Shoulder"]

               ba = a - b
               bc = c - b
               ed = d-e
               ef = f-e

               cosine_angle_left = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
               angle_left = np.arccos(cosine_angle_left)
               actual_angle_left = np.degrees(angle_left)

               cosine_angle_right = np.dot(ed, ef) / (np.linalg.norm(ed) * np.linalg.norm(ef))
               angle_right = np.arccos(cosine_angle_right)
               actual_angle_right = np.degrees(angle_right)

               if actual_angle_left >= float(55) and actual_angle_left <= float(130) and actual_angle_right >= float(55) and actual_angle_right<=float(130):
                  draw_text(img, 100, 140, "Correct Posture", NAVY)
               else:
                  draw_text(img, 100, 140, "Wrong Posture", NAVY)


            draw_text_coordinates(img, x, y, x_y_str, the_color)
      
      return {}
