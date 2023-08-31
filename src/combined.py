from torch.utils.data import DataLoader, Dataset
from bi_lstm import *
from model_visual import *

class CombinedModel(nn.Module):
    def __init__(self, text_model, image_model, out_dim, image_out_dim):
        super(CombinedModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.fc = nn.Linear(text_model.max_length + image_out_dim, out_dim)

    def forward(self, text_inputs, image_features):
        text_features = self.text_model(text_inputs)
        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.fc(combined_features)
        return output

def train_combined_model(train_dataset, text_embedding_dim=100, text_lstm_units=128,
                         image_hid_dim=256, image_out_dim=1024, dropout=0.5,
                         combined_out_dim=None, epochs=10, batch_size=32):
    text_report_generator = ReportGenerator(train_dataset)
    text_X, text_y = text_report_generator.load_dataset()
    text_report_generator.build_model(text_embedding_dim, text_lstm_units)
    text_report_generator.train_model(text_X, text_y, epochs, batch_size)

    image_model = ResNet50(image_hid_dim, image_out_dim, dropout)
    image_model.train()

    train_dataset = MIMIC_DIR(REPORT_PREPROCESS_DIR + '/train_output.jsonl', IMG_DIR_TRAIN, train_flag=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # train_image_dataloader = DataLoader(train_image_dataset, batch_size=batch_size, shuffle=True)
    combined_out_dim = len(text_report_generator.word_index) + 1

    combined_model = CombinedModel(text_report_generator, image_model, combined_out_dim)

    optimizer = torch.optim.Adam(combined_model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for text_inputs, labels in train_loader:
            image_inputs, _ = next(iter(train_loader))
            image_features, _ = image_model(image_inputs)

            optimizer.zero_grad()

            outputs = combined_model(text_inputs, image_features)  # Use image_features here
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    return combined_model
