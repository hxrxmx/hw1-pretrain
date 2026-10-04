import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
)
import torch
import time


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000
RUN_NAME = os.getenv('RUN_NAME', 'best_run')


TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/{RUN_NAME}',
    'optim': os.getenv('OPTIM', 'adamw_torch_fused'),
    'num_train_epochs': 1,
    'max_steps': int(os.getenv('MAX_STEPS', -1)),
    'per_device_train_batch_size': int(os.getenv('BATCH_SIZE', 48)),
    'per_device_eval_batch_size': int(os.getenv('EVAL_BATCH_SIZE', 64)),
    'save_strategy': 'no',
    'learning_rate': float(os.getenv('LEARNING_RATE', 4e-4)),
    'lr_scheduler_type': os.getenv('LR_SCHEDULER', 'constant_with_warmup'),
    'weight_decay': float(os.getenv('WEIGHT_DECAY', 0.01)),
    'adam_beta2': float(os.getenv('ADAM_BETA2', 0.95)),
    'warmup_steps': int(os.getenv('WARMUP_STEPS', 50)),
    'logging_steps': 5,
    'eval_strategy': 'no',
    'prediction_loss_only': True,
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': int(os.getenv('GRAD_ACCUM', 1)),
    'dataloader_num_workers': 4,
    'torch_compile': os.getenv('TORCH_COMPILE', '0') == '1',
    'report_to': 'none',
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    """Load the tokenizer and add a padding token."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    """Tokenize a batch into fixed-length training examples."""
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
    )
    tokenized[LABELS] = [ids.copy() for ids in tokenized[INPUT_IDS]]
    return tokenized


def causal_lm_collator(features):
    batch = default_data_collator(features)
    batch[LABELS][batch[ATTENTION_MASK] == 0] = -100
    return batch


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """Save the tokenized dataset in several parquet files."""
    os.makedirs(output_dir, exist_ok=True)

    for index in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        path = os.path.join(output_dir, f'{index:05d}.parquet')
        shard.to_parquet(path)
        print(f'saved {path}')


def prepare_dataset():
    """Download and tokenize the Russian Wikipedia dataset."""
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()
    workers = min(8, os.cpu_count() or 1)
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        num_proc=workers,
        fn_kwargs={'tokenizer': tokenizer},
        remove_columns=dataset.column_names,
        desc='tokenizing wikipedia',
    )
    save_as_parquets(tokenized)



def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    """Load all prepared parquet shards in filename order."""
    files = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith('.parquet')
    )
    return load_dataset('parquet', data_files=files, split='train')


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


def save_loss_plot(history, output_dir):
    import matplotlib.pyplot as plt

    train = [(row['step'], row['loss']) for row in history if 'loss' in row]
    val = [(row['step'], row['eval_loss']) for row in history if 'eval_loss' in row]

    if train:
        x, y = zip(*train)
        plt.plot(x, y, label='train')
    if val:
        x, y = zip(*val)
        plt.scatter(x, y, label='eval', zorder=3)

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=160)
    plt.close()


def save_generations(model, tokenizer, output_dir):
    prompts = [
        'Искусственный интеллект — это',
        'Москва является столицей',
        'В начале научного эксперимента',
    ]
    rows = []
    model.eval()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        rows.append({'prompt': prompt, 'text': text})
        print(text)

    with open(os.path.join(output_dir, 'generations.json'), 'w', encoding='utf-8') as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def train_model():
    """Train for 15 minutes, evaluate and save the report artifacts."""
    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset()
    train_dataset, eval_dataset = split_dataset(dataset)
    model = create_model(tokenizer)
    args = TrainingArguments(**TRAINING_CONFIG)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as file:
        json.dump(TRAINING_CONFIG, file, indent=2)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=causal_lm_collator,
        processing_class=tokenizer,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)]  # don't change
    )
    trainer.train()
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_state()
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as file:
        json.dump(eval_results, file, indent=2)
    save_loss_plot(trainer.state.log_history, args.output_dir)
    save_generations(model, tokenizer, args.output_dir)


if __name__ == "__main__":
    if os.getenv('PREPARE_DATASET') == '1':
        prepare_dataset()
    else:
        train_model()
