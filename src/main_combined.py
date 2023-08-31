# Import necessary libraries and classes
import torch
from torch.utils.data import DataLoader
from bi_lstm import ReportGenerator  # Import your text model related code
from model_visual import ResNet50
from settings import *   # Import your image model related code
from combined import *
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32  # Define your batch size

# Train the combined model
trained_combined_model = train_combined_model(
    train_dataset=REPORT_PREPROCESS_DIR + '/train_output.jsonl',
    text_embedding_dim=100,
    text_lstm_units=128,
    image_hid_dim=256,
    image_out_dim=1024,
    dropout=0.5,
    epochs=10,
    batch_size=batch_size
)

# Test the trained combined model
def test_combined_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for text_inputs, image_inputs, labels in test_loader:
            text_inputs = text_inputs.to(device)
            image_inputs = image_inputs.to(device)
            labels = labels.to(device)

            outputs = model(text_inputs, image_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Assuming you have a test dataset defined
test_dataset = MIMIC_DIR(REPORT_PREPROCESS_DIR + '/test_output.jsonl', IMG_DIR_TEST)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

# Test the trained combined model
test_combined_model(trained_combined_model, test_loader)
