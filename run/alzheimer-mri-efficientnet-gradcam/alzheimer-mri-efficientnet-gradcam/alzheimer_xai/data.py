from collections import Counter
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess(image, label):
    image = preprocess_input(image)
    return image, label

def load_datasets(train_dir, test_dir, image_size, batch_size):
    train_dataset = keras.utils.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="int",
        batch_size=batch_size, image_size=image_size, shuffle=True)

    test_dataset = keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="int",
        batch_size=batch_size, image_size=image_size, shuffle=False)

    class_names = train_dataset.class_names
    class_counts = Counter()
    for _, labels in train_dataset:
        class_counts.update(labels.numpy())

    train_dataset = train_dataset.map(preprocess)
    test_dataset = test_dataset.map(preprocess)

    return train_dataset, test_dataset, class_names, dict(class_counts)
