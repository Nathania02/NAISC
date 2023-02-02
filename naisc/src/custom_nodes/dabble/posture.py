"""
Node template for creating custom nodes.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

THRESHOLD = 0.5
KP_RIGHT_SHOULDER = 6
KP_LEFT_SHOULDER = 5
KP_RIGHT_ELBOW = 8
KP_LEFT_ELBOW = 7
KP_RIGHT_KNEE = 14
KP_LEFT_KNEE = 13

# class Node(AbstractNode):
#     """This is a template class of how to write a node for PeekingDuck.

#     Args:
#         config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
#     """

#     def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
#         super().__init__(config, node_path=__name__, **kwargs)

#         # initialize/load any configs and models here
#         # configs can be called by self.<config_name> e.g. self.filepath
#         # self.logger.info(f"model loaded with configs: config")

#     def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
#         """This node does ___.

#         Args:
#             inputs (dict): Dictionary with keys "__", "__".

#         Returns:
#             outputs (dict): Dictionary with keys "__".
#         """
#         print("hello")
#         print(inputs)
#         # ankle = inputs["keypoints"][15]
#         # knee = inputs["keypoints"][13]
#         # hip = inputs["keypoints"][11]
#         keypoints = inputs["keypoints"]
#         print("Keypoints", keypoints)

        # the_keypoints = keypoints[0]
        # the_keypoint_scores = keypoint_score[0]

        # for i, keypoints in enumerate(the_keypoints):
        #     keypoint_score = the_keypoint_scores[i]

        #     if keypoint_score >= THRESHOLD:
        #         if i == KP_RIGHT_SHOULDER:
        #             right_shoulder = keypoints
        #             print(right_shoulder)

    # def _get_config_types(self) -> Dict[str, Any]:
    #     return {"tag_msg":str}

        # result = do_something(inputs["in1"], inputs["in2"])
        # outputs = {"out1": result}
        # return outputs


from typing import Any, Dict, List, Tuple
import cv2
from peekingduck.pipeline.nodes.abstract_node import AbstractNode

# setup global constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)       # opencv loads file in BGR format
YELLOW = (0, 255, 255)
THRESHOLD = 0.6               # ignore keypoints below this threshold
KP_RIGHT_SHOULDER = 6         # PoseNet's skeletal keypoints
KP_RIGHT_WRIST = 10


def map_bbox_to_image_coords(
   bbox: List[float], image_size: Tuple[int, int]
) -> List[int]:
   """First helper function to convert relative bounding box coordinates to
   absolute image coordinates.
   Bounding box coords ranges from 0 to 1
   where (0, 0) = image top-left, (1, 1) = image bottom-right.

   Args:
      bbox (List[float]): List of 4 floats x1, y1, x2, y2
      image_size (Tuple[int, int]): Width, Height of image

   Returns:
      List[int]: x1, y1, x2, y2 in integer image coords
   """
   width, height = image_size[0], image_size[1]
   x1, y1, x2, y2 = bbox
   x1 *= width
   x2 *= width
   y1 *= height
   y2 *= height
   return int(x1), int(y1), int(x2), int(y2)


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


def draw_text(img, x, y, text_str: str, color_code):
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


class Node(AbstractNode):
   """Custom node to display keypoints and count number of hand waves

   Args:
      config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
   """

   def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
      super().__init__(config, node_path=__name__, **kwargs)
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
      bboxes = inputs["bboxes"]
      bbox_scores = inputs["bbox_scores"]
      keypoints = inputs["keypoints"]
      keypoint_scores = inputs["keypoint_scores"]
      print(keypoints)

      img_size = (img.shape[1], img.shape[0])  # image width, height

      # get bounding box confidence score and draw it at the
      # left-bottom (x1, y2) corner of the bounding box (offset by 30 pixels)
      the_bbox = bboxes[0]             # image only has one person
      the_bbox_score = bbox_scores[0]  # only one set of scores

      x1, y1, x2, y2 = map_bbox_to_image_coords(the_bbox, img_size)
      score_str = f"BBox {the_bbox_score:0.2f}"
      cv2.putText(
         img=img,
         text=score_str,
         org=(x1, y2 - 30),            # offset by 30 pixels
         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
         fontScale=1.0,
         color=WHITE,
         thickness=3,
      )

      # hand wave detection using a simple heuristic of tracking the
      # right wrist movement
      the_keypoints = keypoints[0]              # image only has one person
      the_keypoint_scores = keypoint_scores[0]  # only one set of scores
      right_wrist = None
      right_shoulder = None

      for i, keypoints in enumerate(the_keypoints):
         keypoint_score = the_keypoint_scores[i]

         if keypoint_score >= THRESHOLD:
            x, y = map_keypoint_to_image_coords(keypoints.tolist(), img_size)
            x_y_str = f"({x}, {y})"

            if i == KP_RIGHT_SHOULDER:
               right_shoulder = keypoints
               the_color = YELLOW
            elif i == KP_RIGHT_WRIST:
               right_wrist = keypoints
               the_color = YELLOW
            else:                   # generic keypoint
               the_color = WHITE

            draw_text(img, x, y, x_y_str, the_color)

      if right_wrist is not None and right_shoulder is not None:
         # only count number of hand waves after we have gotten the
         # skeletal poses for the right wrist and right shoulder
         if self.right_wrist is None:
            self.right_wrist = right_wrist            # first wrist data point
         else:
            # wait for wrist to be above shoulder to count hand wave
            if right_wrist[1] > right_shoulder[1]:
               pass
            else:
               if right_wrist[0] < self.right_wrist[0]:
                  direction = "left"
               else:
                  direction = "right"

               if self.direction is None:
                  self.direction = direction          # first direction data point
               else:
                  # check if hand changes direction
                  if direction != self.direction:
                     self.num_direction_changes += 1
                  # every two hand direction changes == one wave
                  if self.num_direction_changes >= 2:
                     self.num_waves += 1
                     self.num_direction_changes = 0   # reset direction count

               self.right_wrist = right_wrist         # save last position
               self.direction = direction

         wave_str = f"#waves = {self.num_waves}"
         draw_text(img, 20, 30, wave_str, YELLOW)

      return {}
