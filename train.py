#!/usr/bin/env python3

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


epochs = 3
print_every = 40
steps = 0

model.to(device)
for epoch in range(epochs):
    running_loss = 0

    model.train()
    for inputs, labels in iter(training_loader):
        steps += 1
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                validation_loss, accuracy = validation(model, validation_loader, criterion)

            print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                  "Validation Loss: {:.3f}.. ".format(validation_loss / len(validation_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy / len(validation_loader)))

            running_loss = 0
            model.train()


# TODO: Do validation on the test set
model.eval()
test_loss, test_accuracy = validation(model, testing_loader, criterion)
print("Test Loss: {:.3f}.. ".format(test_loss / len(testing_loader)),
      "Test Accuracy: {:.3f}".format(test_accuracy / len(testing_loader)))


model.class_to_idx = training_data.class_to_idx
# TODO: Save the checkpoint
checkpoint = {
    'model': models.vgg16(),
    'classifier_input_size': input_size,
    'classifier_output_size': output_size,
    'classifier_hidden_layers': hidden_layers,
    'classifier_dropout': drop_p,
    'model_state_dict': model.state_dict(),
    'training_epochs': epochs,
    'optimizer': optimizer,
    'learning_rate': learning_rate,
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_index': training_data.class_to_idx,
}

torch.save(checkpoint, 'checkpoint.pth')
