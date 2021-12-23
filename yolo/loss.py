import tensorflow as tf
from tensorflow.keras.losses import Loss


def Rect2Box(xyxy: tf.Variable) -> tf.Variable:
    return tf.stack([
        (xyxy[..., 2:4] + xyxy[..., 0:2]) / 2,
        xyxy[..., 2:4] - xyxy[..., 0:2],
    ], axis=-1)


def Box2Rect(xywh: tf.Variable) -> tf.Variable:
    return tf.concat([
        xywh[..., 0:2] - (xywh[..., 2:4] / 2),
        xywh[..., 0:2] + (xywh[..., 2:4] / 2),
    ], axis=-1)


def intersection_area(xyxy_a: tf.Tensor, xyxy_b: tf.Tensor) -> tf.Tensor:
    wh = tf.minimum(xyxy_a[..., 2:4], xyxy_b[..., 2:4]) - \
        tf.maximum(xyxy_a[..., 0:2], xyxy_b[..., 0:2])
    return tf.reduce_prod(tf.where(wh >= 0, wh, 0), axis=-1, keepdims=True)


def union_area(xyxy_a: tf.Tensor, xyxy_b: tf.Tensor, intersection: tf.Tensor = None) -> tf.Tensor:
    if intersection is None:
        intersection = intersection_area(xyxy_a, xyxy_b)

    return (
        tf.reduce_prod(xyxy_a[..., 2:4] - xyxy_a[..., 0:2], axis=-1, keepdims=True) +
        tf.reduce_prod(xyxy_b[..., 2:4] - xyxy_b[..., 0:2], axis=-1, keepdims=True) -
        intersection
    )


def IoU(xyxy_a: tf.Tensor, xyxy_b: tf.Tensor) -> tf.Tensor:
    intersection = intersection_area(xyxy_a, xyxy_b)
    union = union_area(xyxy_a, xyxy_b, intersection=intersection)
    return intersection / union


class YOLOLoss(Loss):
    lambda_coord: float
    lambda_no_obj: float
    num_boxes: int

    def __init__(
            self,
            lambda_coord: float = 5,
            lambda_no_obj: float = 0.5,
            num_boxes: int = 1,
            reduction='sum',
            name="YOLO_Loss"
    ):
        super().__init__(reduction=reduction, name=name)
        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj
        self.num_boxes = num_boxes

    def call(self, y_true, y_pred) -> tf.Variable:
        true_prediction = y_true[..., 0:1]
        true_box = y_true[..., tf.newaxis, 1:5]
        true_classes = y_true[..., 5:]

        predict_trust = y_pred[..., :2]
        predict_boxes = tf.reshape(y_pred[..., 2:10], [-1, 7, 7, 2, 4])
        predict_classes = y_pred[..., 10:]

        true_box_xyxy = Box2Rect(true_box)
        predict_boxes_xyxy = Box2Rect(predict_boxes)

        iou_scores = IoU(true_box_xyxy, predict_boxes_xyxy)[..., 0]
        best_box = tf.reduce_max(iou_scores, axis=3, keepdims=True)

        mask = tf.cast(iou_scores >= best_box, iou_scores.dtype)
        l_obj = true_prediction * mask
        l_no_obj = 1 - true_prediction

        object_loss = l_obj * tf.square(1 - predict_trust)
        no_object_loss = l_no_obj * tf.square(0 - predict_trust)

        confidence_loss = tf.reduce_sum(
            object_loss, axis=[1, 2, 3]
        ) + self.lambda_no_obj * tf.reduce_sum(
            no_object_loss, axis=[1, 2, 3]
        )
        classes_loss = tf.reduce_sum(
            true_prediction * tf.square(true_classes - predict_classes), axis=[1, 2, 3])

        mask = tf.expand_dims(mask, axis=-1)
        true_prediction = tf.expand_dims(true_prediction, axis=-1)
        l_obj = mask * true_prediction

        position_loss = self.lambda_coord * tf.reduce_sum(
            l_obj * tf.square(true_box[..., 0:2] - predict_boxes[..., 0:2]), axis=[1, 2, 3, 4]
        )

        size_loss = self.lambda_coord * tf.reduce_sum(
            l_obj * tf.square(tf.sqrt(true_box[..., 2:4]) - tf.sqrt(predict_boxes[..., 2:4])), axis=[1, 2, 3, 4]
        )

        return position_loss + size_loss + confidence_loss + classes_loss
