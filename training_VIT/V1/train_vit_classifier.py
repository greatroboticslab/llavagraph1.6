import os
import torch
import numpy as np
import evaluate
from PIL import Image
from torchvision import transforms
from sklearn.utils import shuffle
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer


train_dir = "/Users/ilminurablikim/Desktop/split_dataset_aug/train"
val_dir = "/Users/ilminurablikim/Desktop/split_dataset_aug/test"
output_dir = "/Users/ilminurablikim/Desktop/vit_output_aug"
os.makedirs(output_dir, exist_ok=True)


label_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}
print("Classes found:", label_names)


image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


weak_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])


def get_image_list(data_path):
    images, labels = [], []
    for label in label_names:
        folder_path = os.path.join(data_path, label)
        if not os.path.exists(folder_path): continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(folder_path, fname))
                labels.append(label2id[label])
    return list(zip(images, labels))

train_data = shuffle(get_image_list(train_dir), random_state=42)
val_data = get_image_list(val_dir)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        return {"pixel_values": weak_transform(image), "labels": torch.tensor(label)}

train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }


model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)


training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    num_train_epochs=15,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=val_dataset,
    data_collator=collate_fn, compute_metrics=compute_metrics,
)


trainer.train()
trainer.save_model(output_dir)
image_processor.save_pretrained(output_dir)
print(f"Model trained and saved to {output_dir}")
