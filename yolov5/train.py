import torch
from model import YOLOV5m
from loss import YOLO_LOSS
from torch.optim import Adam
from utils.validation_utils import mean_average_precision, check_class_accuracy, get_evaluation_bboxes
from utils.training_utils import train_loop, get_loaders
from utils.utils import save_checkpoint, load_model_checkpoint, load_optim_checkpoint
from utils.plot_utils import save_predictions
import config


if __name__ == "__main__":

    first_out = 48
    scaler = torch.cuda.amp.GradScaler()

    model = YOLOV5m(first_out=first_out, nc=len(config.COCO_LABELS), anchors=config.ANCHORS,
                    ch=(first_out*4, first_out*8, first_out*16), inference=False).to(config.DEVICE)

    loss_fn = YOLO_LOSS(model, save_logs=True)
    optim = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    if config.CHECKPOINT:
        load_model_checkpoint(config.CHECKPOINT, model)
        load_optim_checkpoint(config.CHECKPOINT, optim)

    train_loader, val_loader = get_loaders(db_root_dir=config.ROOT_DIR, batch_size=8,
                                           train_transform=config.VAL_TRANSFORM,
                                           val_transform=config.VAL_TRANSFORM, num_workers=4)

    for epoch in range(0, config.EPOCHS):

        print(f"Training epoch {epoch+1}/{config.EPOCHS}")

        train_loop(model=model, loader=train_loader, loss_fn=loss_fn,
                   optim=optim, scaler=scaler, epoch=epoch+1)

        # ALADDIN'S
        with torch.no_grad():
            model.eval()
            
            if epoch > 20:    
                print("Computing: class, no-obj and obj accuracies ...")
                check_class_accuracy(model, val_loader, threshold=config.CONF_THRESHOLD)
                
            if epoch > 50:
                print("Computing MAP ...")
                pred_boxes, true_boxes = get_evaluation_bboxes(
                    val_loader,
                    model,
                    iou_threshold=config.NMS_IOU_THRESH,
                    anchors=config.ANCHORS,
                    threshold=config.CONF_THRESHOLD,
                )
                mapval = mean_average_precision(
                    pred_boxes,
                    true_boxes,
                    iou_threshold=config.MAP_IOU_THRESH,
                    box_format="midpoint",
                    num_classes=len(config.COCO_LABELS),
                )
                print(f"MAP: {mapval.item()}")
            

        # NMS WRONGLY MODIFIED TO TEST THIS FEATURE!!
        # save_predictions(model=model, loader=val_loader, epoch=epoch,
        #                 num_images=10, folder="SAVED_IMAGES", device=config.DEVICE)

        model.train()

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
        }

        # save_checkpoint(checkpoint, folder_path=config.SAVE_MODEL_PATH,
        #                filename=f"checkpoint_epoch_{epoch + 1}.pth.tar")
