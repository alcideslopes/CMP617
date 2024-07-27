from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from uuid import uuid4
import torch

dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes')

generated_size = 4

label_map = {0: 'negative', 1: 'positive'}


device = "cuda" # for GPU usage or "cpu" for CPU usage

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
)


id = 0
for example in dataset['train']:
    
    messages = [
        {"role": "system", "content": "You are a movie reviewer who always evaluate movies based on a sentiment!"},
        {"role": "user", "content": f"""TASK: According to folowing REVIEW and SENTIMENT, list {generated_size} similar but different movie reviews.\n
                                        REVIEW: \"{example['text']}\"\n
                                        SENTIMENT: {label_map[example['label']]}\n
                                        SPECIFIC REQUIREMENTS: only generate an enumerated list without any other kind of text!\n
                                    """},
        ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    output_text = tokenizer.decode(response, skip_special_tokens=True)

    new_instance = dict()

    new_instance['original_text'] = example['text']
    new_instance['generated_text'] = output_text
    new_instance['label'] = example['label']
    
    with open(f'output\\{id}.json', 'w') as f: 
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