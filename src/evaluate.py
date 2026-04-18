import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import get_model


DATA_DIR = "../data/EuroSAT_RGB"
BATCH_SIZE = 32
IMG_SIZE = 64
MODEL_PATH = "../outputs/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )

    model = get_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
