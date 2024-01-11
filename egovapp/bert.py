import pandas as pd


class BertAlgo:
    def bert():
        global df
        df = pd.read_csv('egovapp/questions.csv', sep=',')
        df = df.sample(n=100)
        return df
    def bertm():
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification, AdamW
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        import pandas as pd

        # Load your CSV dataset
        data = df
        user_messages = data["title"]
        chatbot_responses = data["answer"]

        # Initialize the BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

        # Tokenize and encode your dataset
        input_ids = []
        attention_masks = []

        for i in range(len(user_messages)):
            encoded_dict = tokenizer.encode_plus(
                user_messages[i],
                chatbot_responses[i],
                add_special_tokens=True,
                max_length=128,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])

        # Convert lists to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor([1] * len(data))  # Assuming positive examples

        # Split data into training and validation sets
        train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
            input_ids, attention_masks, labels, test_size=0.2, random_state=42
        )

        # Create data loaders
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

        # Define optimizer and loss function
        optimizer = AdamW(model.parameters(), lr=1e-5)
        criterion = torch.nn.BCEWithLogitsLoss()  # Assuming binary classification

        # Training loop
        model.train()
        for epoch in range(3):
            for batch in train_dataloader:
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()