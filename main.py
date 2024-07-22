from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from uuid import uuid4

dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes')
llm = "HuggingFaceTB/SmolLM-1.7B-Instruct"
generated_size = 4

label_map = {0: 'negative', 1: 'positive'}


device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(llm)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(llm).to(device)

id = 0
for example in dataset['train']:
    
    new_instance = dict()

    message = [{"role": "user", 
                "content": f"""TASK: According to PHRASE and SENTIMENT, list {generated_size} similar but different phrases\n
                                PHRASE: \"{example['text']}\"
                                SENTIMENT: {label_map[example['label']]}
                            """}]

    input_text = tokenizer.apply_chat_template(message, tokenize=False)
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(device)
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=600, temperature=0.6, top_p=0.92, do_sample=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)



    new_instance['original_text'] = example['text']
    new_instance['generated_text'] = output_text
    new_instance['label'] = example['label']
    
    with open(f'{id}.json', 'w') as f: 
        json.dump(new_instance, f, indent=4)

    id += 1
# from transformers import AutoTokenizer

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# # Tokenize the dataset
# def preprocess_function(examples):
#     return tokenizer(examples['text'], padding="max_length", truncation=True)

# # Apply the preprocessing function to the dataset
# tokenized_datasets = dataset.map(preprocess_function, batched=True)

# # Prepare the data for the model
# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# tokenized_datasets.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])


# print(tokenized_datasets)



# from transformers import AutoModelForSequenceClassification

# # Load the model
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# # Move the model to the specified device
# model.to(device)

# from transformers import TrainingArguments, Trainer

# # Set up training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     eval_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['validation']
# )

# # Train the model
# trainer.train()

# results = trainer.evaluate(tokenized_datasets['test'])

# # Print the evaluation results
# print(results)