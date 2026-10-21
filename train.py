from ultralytics import YOLO
import torch


def main():

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(
        r"C:\AIProjects\object-tracking\yolov8n.pt"
    )

    model.train(
        data="dataset/data.yaml",
        epochs=15,
        imgsz=416,
        batch=2,
        device=0,
        workers=0,
        project="runs",
        name="neu_yolov8n_15ep"
    )


if __name__ == "__main__":
    main()