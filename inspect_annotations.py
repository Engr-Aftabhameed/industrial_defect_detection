from pathlib import Path
import cv2

# Dataset paths
image_dir = Path("dataset/train/images")
label_dir = Path("dataset/train/labels")

# Class names from data.yaml
class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# Output directory
output_dir = Path("annotation_check")
output_dir.mkdir(exist_ok=True)

# Get first 10 images
image_files = list(image_dir.glob("*"))[:10]

for image_path in image_files:

    # Read image
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    height, width = image.shape[:2]

    # Corresponding label file
    label_path = label_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"No label found for: {image_path.name}")
        continue

    # Read annotations
    with open(label_path, "r") as file:
        lines = file.readlines()

    for line in lines:

        values = line.strip().split()

        if len(values) != 5:
            continue

        class_id = int(values[0])
        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        # Convert normalized YOLO coordinates to pixels
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # Class name
        class_name = class_names[class_id]

        cv2.putText(
            image,
            class_name,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    # Save inspected image
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)

    print(f"Saved: {output_path}")

print("\nAnnotation inspection complete.")