import os.path

import matplotlib.pyplot as plt
import config
import numpy as np
import matplotlib.patches as patches
import torch
from utils.bboxes_utils import non_max_suppression as nms

# ALADDIN'S
def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = box_predictions[..., 0:2].sigmoid() * 2 - 0.5
        box_predictions[..., 2:4] = (box_predictions[..., 2:4].sigmoid() * 2) ** 2 * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


# ALADDIN'S ADAPTED

    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Display the image
    ax1.imshow(im)
    ax2.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height
    axes = [ax1, ax2]
    # Create a Rectangle patch
    boxes = [gtbboxes, pred_boxes]
    for i in range(2):
        for box in boxes[i]:
            assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
            class_pred = int(box[0])
            box = box[2:]
            # CHECK CHECK CHECK CHECK CHECK
            upper_left_x = max(box[0] - box[2] / 2, 0)
            upper_left_x = min(upper_left_x, 1)

            lower_left_y = max(box[1] - box[3] / 2, 0)
            lower_left_y = min(lower_left_y, 1)
            # print(upper_left_x)
            # print(lower_left_y)
            rect = patches.Rectangle(
                (upper_left_x * width, lower_left_y * height),
                box[2] * width,
                box[3] * height,
                linewidth=2,
                edgecolor=colors[class_pred],
                facecolor="none",
            )
            # Add the patch to the Axes
            if i == 0:
                axes[i].set_title("Ground Truth bboxes")
            else:
                axes[i].set_title("Predicted bboxes")
            axes[i].add_patch(rect)
            axes[i].text(
                upper_left_x * width,
                lower_left_y * height,
                s=class_labels[class_pred],
                color="white",
                verticalalignment="top",
                bbox={"color": colors[class_pred], "pad": 0},
            )

    plt.show()

def save_predictions(model, loader, folder, epoch, device, num_images=10):

    if not os.path.exists(path=os.path.join(os.getcwd(), folder, f'EPOCH_{str(epoch+1)}')):
        os.makedirs(os.path.join(os.getcwd(), folder, f'EPOCH_{str(epoch+1)}'))

    path = os.path.join(os.getcwd(), folder, f'EPOCH_{str(epoch+1)}')
    
    # using directly outputs shapes
    # S = getattr(model.head, "stride")
    anchors = getattr(model.head, "anchors")

    """scaled_anchors = torch.tensor(anchors).clone().detach() / (
            1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )"""
    # shape = 640  #resize h,w for validation set
    # S = [640 / i for i in S]

    for idx, (images, targets) in enumerate(loader):
        images = images.to(config.DEVICE)
        boxes = []
        gt_boxes = []
        if idx < num_images:
            with torch.no_grad():
                model.eval()
                out = model(images)
                for i in range(getattr(model.head, "naxs")):
                    anchor = anchors[i]

                    boxes += cells_to_bboxes(
                        out[i], is_preds=True, S=out[i].shape[2], anchors=anchor
                    )[0]

                    gt_boxes += cells_to_bboxes(
                        targets[i], is_preds=False, S=targets[i].shape[2], anchors=anchor
                    )[0]

                boxes = nms(boxes, iou_threshold=1, threshold=0.5, box_format="midpoint")
                gt_boxes = nms(gt_boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")

                cmap = plt.get_cmap("tab20b")
                class_labels = config.COCO_LABELS
                colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
                im = np.array(images[0].permute(1,2,0))
                height, width, _ = im.shape

                # Create figure and axes
                fig, (ax1, ax2) = plt.subplots(1, 2)
                # Display the image
                ax1.imshow(im)
                ax2.imshow(im)

                # box[0] is x midpoint, box[2] is width
                # box[1] is y midpoint, box[3] is height
                axes = [ax1, ax2]
                # Create a Rectangle patch
                boxes = [gt_boxes, boxes]
                for i in range(2):
                    for box in boxes[i]:
                        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
                        class_pred = int(box[0])
                        box = box[2:]
                        # CHECK CHECK CHECK CHECK CHECK
                        upper_left_x = max(box[0] - box[2] / 2, 0)
                        upper_left_x = min(upper_left_x, 1)

                        lower_left_y = max(box[1] - box[3] / 2, 0)
                        lower_left_y = min(lower_left_y, 1)
                        # print(upper_left_x)
                        # print(lower_left_y)
                        rect = patches.Rectangle(
                            (upper_left_x * width, lower_left_y * height),
                            box[2] * width,
                            box[3] * height,
                            linewidth=2,
                            edgecolor=colors[class_pred],
                            facecolor="none",
                        )
                        # Add the patch to the Axes
                        if i == 0:
                            axes[i].set_title("Ground Truth bboxes")
                        else:
                            axes[i].set_title("Predicted bboxes")
                        axes[i].add_patch(rect)
                        axes[i].text(
                            upper_left_x * width,
                            lower_left_y * height,
                            s=class_labels[class_pred],
                            color="white",
                            verticalalignment="top",
                            bbox={"color": colors[class_pred], "pad": 0},
                            fontsize="small"
                        )

                fig.savefig(f'{path}/image_{idx}.png', dpi=300)
                plt.close(fig)
        break


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = max(box[0] - box[2] / 2, 0)
        upper_left_x = min(upper_left_x, 1)

        lower_left_y = max(box[1] - box[3] / 2, 0)
        lower_left_y = min(lower_left_y, 1)
        rect = patches.Rectangle(
            (upper_left_x * width, lower_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            lower_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    plt.show()
    #plt.close(fig)

