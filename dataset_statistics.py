from pathlib import Path
from collections import Counter
import yaml

# -----------------------------
# Paths
# -----------------------------
dataset_dir = Path("dataset")
train_labels = dataset_dir / "train" / "labels"
valid_labels = dataset_dir / "valid" / "labels"

# -----------------------------
# Load class names
# -----------------------------
with open(dataset_dir / "data.yaml", "r") as f:
    data = yaml.safe_load(f)

class_names = data["names"]

print("\n==============================")
print("INDUSTRIAL DEFECT DATASET")
print("==============================")

print(f"\nNumber of classes: {len(class_names)}")

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")


# -----------------------------
# Analyze labels
# -----------------------------
def analyze_labels(label_dir):

    class_counter = Counter()
    total_boxes = 0
    empty_labels = 0
    label_files = list(label_dir.glob("*.txt"))

    for label_file in label_files:

        with open(label_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            empty_labels += 1
            continue

        for line in lines:

            values = line.split()

            if len(values) != 5:
                print(f"WARNING: Invalid label: {label_file}")
                continue

            class_id = int(values[0])

            class_counter[class_id] += 1
            total_boxes += 1

    return label_files, class_counter, total_boxes, empty_labels


# -----------------------------
# Training statistics
# -----------------------------
train_files, train_counts, train_boxes, train_empty = analyze_labels(
    train_labels
)

# -----------------------------
# Validation statistics
# -----------------------------
valid_files, valid_counts, valid_boxes, valid_empty = analyze_labels(
    valid_labels
)


# -----------------------------
# Print training statistics
# -----------------------------
print("\n==============================")
print("TRAINING SET")
print("==============================")

print(f"Label files: {len(train_files)}")
print(f"Total defect boxes: {train_boxes}")
print(f"Empty label files: {train_empty}")

print("\nDefect distribution:")

for class_id, class_name in enumerate(class_names):
    count = train_counts[class_id]
    percentage = (count / train_boxes * 100) if train_boxes else 0

    print(
        f"{class_id}: {class_name:<20} "
        f"{count:>5} boxes  ({percentage:>6.2f}%)"
    )


# -----------------------------
# Print validation statistics
# -----------------------------
print("\n==============================")
print("VALIDATION SET")
print("==============================")

print(f"Label files: {len(valid_files)}")
print(f"Total defect boxes: {valid_boxes}")
print(f"Empty label files: {valid_empty}")

print("\nDefect distribution:")

for class_id, class_name in enumerate(class_names):
    count = valid_counts[class_id]
    percentage = (count / valid_boxes * 100) if valid_boxes else 0

    print(
        f"{class_id}: {class_name:<20} "
        f"{count:>5} boxes  ({percentage:>6.2f}%)"
    )


# -----------------------------
# Overall statistics
# -----------------------------
total_images = len(train_files) + len(valid_files)
total_boxes = train_boxes + valid_boxes

print("\n==============================")
print("OVERALL")
print("==============================")

print(f"Total images: {total_images}")
print(f"Total defect boxes: {total_boxes}")

if total_images:
    print(
        f"Average defects per image: "
        f"{total_boxes / total_images:.2f}"
    )

print("\nDataset analysis complete.")