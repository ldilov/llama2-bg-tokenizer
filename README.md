# Tokenizer Training Guide

This document provides instructions on how to use the `train.py` script for training a tokenizer. The script allows for the configuration of tokenizer parameters and the loading of various datasets for training.

## Configuration

Before running the training script, you must specify the tokenizer and training configurations in a JSON file. Below is an explanation of the key configurable values:

- `add_bos_token`: Boolean, adds a beginning of sentence token.
- `add_eos_token`: Boolean, adds an end of sentence token.
- `added_tokens_decoder`: Object, specifies additional tokens with their properties (e.g., content, special).
- `additional_special_tokens`: Array, list of additional special tokens.
- `special_tokens_attr`: Array, specifies the attributes for special tokens (e.g., unk_token, bos_token).
- `dropout`: Float, sets the dropout rate for training.
- `min_frequency`: Integer, minimum frequency threshold for tokens to be included in the vocabulary.
- `max_length`: Integer, maximum token length.
- `pad_to_multiple_of`: Null or Integer, pads sequences to a multiple of this value.
- `bos_token`, `eos_token`, `unk_token`: String, specifies the tokens used for beginning of sentence, end of sentence, and unknown tokens, respectively.
- `fuse_unk`: Boolean, determines if unknown tokens should be fused.
- `clean_up_tokenization_spaces`: Boolean, controls space cleanup in tokenization.
- `model_max_length`: Integer, maximum model length.
- `pad_token`: String, token used for padding.
- `pad_type_id`: Integer, type ID used for padding.
- `padding_side`: String, specifies the side for padding ("left" or "right").
- `sp_model_kwargs`: Object, additional keyword arguments for SentencePiece.
- `spaces_between_special_tokens`: Boolean, adds spaces between special tokens.
- `tokenizer_class`: String, class name of the tokenizer.
- `use_default_system_prompt`: Boolean, use the default system prompt.
- `template`: Object, template for single and pair tokenization.
- `metadata`: Object, includes metadata like vocabulary size.
- `replacement_char`: String, character used for replacements.

## Running the Script

1. Ensure you have the necessary Python packages installed. You can install them using pip or conda as shown below.

### pip Commands

```bash
pip install datasets transformers tokenizers sentencepiece pathlib
```

```bash
conda install -c huggingface datasets transformers pathlib sentencepiece tokenizers
```

2. To start training go to `./scripts` and run `train.py`


## Script adjustment

In `scripts/train.py` you will find the line:

```python
  trainer.train([oscar_dataset, dataset_poems, dataset_opus, dataset_reasoning, dataset_fake_news], 550000)
```

The number `550000` shows how many dataset entries you want to include in the training process. 
Inside the list provided as first argument, you can specify which `Dataset` objects you want to include.

In `tokenizer.core.utilities.constants` you can specify your own custom alphabet inside the `ALPHABET` variable.

After the training completes, the model files are located in `tokenizer.scripts.saved_models.llama`. There you should find:
The resulting tokenizer is compatible with `LlamaTokenizerFast` class and to be more specific - `Llama2`-based models.
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json

# TokenizerTrainer Class Developer's Guide

The `TokenizerTrainer` class is a central component for training, evaluating, and managing tokenizers tailored to specific datasets and text processing tasks. It encapsulates the complexities of tokenizer training, including configuration, dataset preparation, training execution, and evaluation, making it an indispensable tool for developers working on natural language processing (NLP) projects.

## Features

- **Flexible Configuration**: Utilizes a JSON-based configuration for easy customization of tokenizer properties, training parameters, and special tokens.
- **Comprehensive Training Process**: Automates the training process from dataset merging, dynamic token adjustments based on frequency, to the actual training of the tokenizer.
- **Built-in Evaluation**: Offers functionality to evaluate the tokenizer's performance on a holdout dataset, providing insights into vocabulary effectiveness and potential round-trip errors.
- **Easy Serialization**: Facilitates the saving of trained tokenizer models and their configurations for later use or deployment.

## Key Methods

- `__init__(config: Dict[str, Union[str, int, Dict]])`: Initializes the `TokenizerTrainer` with a given configuration.
- `_initialize_tokenizer() -> tokenizers.Tokenizer`: Sets up the tokenizer with predefined normalization, pre-tokenization, decoding strategies, and post-processing templates.
- `train(datasets: List[Union[Dataset, DatasetDict]], limit: int = 50000)`: Trains the tokenizer using the provided list of datasets.
- `evaluate_tokenizer(holdout_dataset: List[str])`: Evaluates the tokenizer on a separate dataset to test its performance.
- `save(path: Union[str, Path]) -> None`: Saves the trained tokenizer and its configuration to the specified path.

## Configuration

The configuration for the `TokenizerTrainer` is specified in a JSON file, which includes:

- **Tokenization Settings**: Attributes such as `add_bos_token`, `add_eos_token`, special tokens configurations, and other tokenizer-specific settings.
- **Training Parameters**: Settings like `dropout`, `min_frequency`, `max_length`, which directly influence the training process and outcome.
- **Normalization and Pre-tokenization Rules**: Defines how text should be normalized and pre-tokenized before being processed by the tokenizer model.

## Usage Example

```python
# Load configuration
config_path = "path/to/tokenizer_config.json"
with open(config_path, 'r') as file:
    config = json.load(file)

# Initialize Trainer
trainer = TokenizerTrainer(config)

# Prepare Datasets
datasets = [...]  # List of Dataset or DatasetDict instances

# Train Tokenizer
trainer.train(datasets)

# Evaluate Tokenizer
trainer.evaluate_tokenizer(holdout_dataset)

# Save Tokenizer
trainer.save("path/to/save/tokenizer")
```

### **Advanced Configuration**
    - `decoders.Replace`: Replaces specified characters (e.g., the replacement character `"▁"`) with another character (e.g., a space), aiding in the reconstruction of the original text from tokenized sequences.
    - `decoders.ByteFallback`: Provides a fallback mechanism for handling bytes directly, useful for dealing with unknown or out-of-vocabulary tokens.
    - `decoders.Fuse()`: Fuses consecutive tokens when possible to reduce tokenization granularity, potentially improving model performance by reducing sparsity.
    - `decoders.Strip`: Removes leading or trailing characters (e.g., spaces), cleaning up the tokenized output for further processing.
- **Impact**: Decoders play a crucial role in translating tokenized sequences back into human-readable text, ensuring the tokenizer's output remains faithful to the original input while accommodating the model's needs.

### **Normalizers**

- **Components**:
    - `Prepend("▁")`: Adds a specific character (e.g., the replacement character `"▁"`) to the beginning of the text, marking the start of processing.
    - `Replace(r" ", "▁")`: Replaces spaces with a specified character, aiding in distinguishing between spaces as part of the text and as token separators.
    - `NFKC()`: Applies Unicode normalization (NFKC), standardizing characters and reducing the complexity of text encoding.
- **Impact**: Normalizers standardize and prepare the input text for tokenization, improving the model's robustness and consistency in handling diverse text inputs.

### **Pre-tokenizers**

- **Components**:
    - `pre_tokenizers.Sequence([Punctuation()])`: Applies a sequence of pre-tokenizers, such as identifying and separating punctuation, which helps in parsing the text more accurately before the main tokenization step.
- **Impact**: Pre-tokenizers refine the input text by identifying and isolating components like punctuation, which enhances the tokenizer's ability to accurately segment text into tokens.

### **Post-processing Template**

- **Components**:
    - `TemplateProcessing(single, pair, special_tokens)`: Defines templates for processing single inputs and pairs of inputs, incorporating special tokens at specified positions.
- **Impact**: Post-processing templates dictate how tokenized sequences are structured, ensuring that special tokens are correctly placed. This is crucial for tasks that require understanding the relationship between sequences (e.g., question-answering), as it impacts how the model interprets sequence boundaries and relationships.


### **Dynamic Token Adjustments**

- **Dynamic Token Selection**: Employs statistical analysis to dynamically adjust the minimum frequency of tokens and identify rare but significant tokens (`dynamic_tokens`) for inclusion, improving model performance on specific domains or datasets.

### **Training and Evaluation Mechanism**

- **Efficient Training**: Leverages a custom training loop that merges datasets, applies dynamic token adjustments, and trains the tokenizer on merged datasets, prioritizing efficiency and effectiveness.
- **Evaluation**: Includes a sophisticated evaluation mechanism to assess tokenizer performance using a holdout dataset, focusing on round-trip errors and tokenization loss, ensuring the tokenizer's reliability and accuracy.

### **Advanced Tokenization Techniques**

- **Byte-Pair Encoding (BPE) with Custom Extensions**: Enhances the standard BPE algorithm with byte fallback, dropout, and unknown token fusion, addressing common tokenization challenges and improving token representation.

### **Sophisticated Normalization and Pre-tokenization**

- Implements a sequence of normalization and pre-tokenization steps that prepare text data for tokenization, improving the model's ability to understand and process varied textual inputs.

### **Comprehensive Post-processing**

- **Template Processing**: Utilizes template processing for single and pair tokenization tasks, incorporating special tokens effectively and ensuring consistent tokenization patterns.

### **Advantages Over Regular Approaches**

- **Dynamic Dropout**: Tokenizer training process doesn't use predefined `dropout` but instead calculates on the fly specifically tailored value, based on the current training dataset. This ensures that tokenizer model can generalize better by putting more weight on context rather than specifics. This would be beneficial at later stage when finetuning LLM with this tokenizer.
- **Dynamic Adaptation**: The ability to dynamically adjust tokenization parameters based (like `min_frequency`) on dataset analysis ensures that the tokenizer remains effective across different text domains.
- **Dynamic Tokens**: The dataset is divided into chunks, each chunk of the dataset is analyzed to count the occurrences of each token. This is done across all chunks in parallel, and the results are aggregated. A threshold (e.g., `0.0005`) is applied to identify tokens that constitute a small fraction of the total words in the dataset. Tokens below this threshold are considered rare or dynamic. From these dynamic tokens, the top `k` tokens with the highest counts (but still under the threshold) are selected. We add them manually to tokenizer's vocabulary so that the tokenizer can focus its attention on the most relevant rare tokens. Dynamic tokens often include terminology, names, or concepts specific to a dataset's domain. Their inclusion in the tokenizer's vocabulary allows the LLM to capture and understand these unique elements more effectively, leading to improved performance in tasks requiring deep domain knowledge or contextual nuance.
- **Sophisticated Evaluation**: The inclusion of a detailed evaluation mechanism enables continuous assessment and improvement of the tokenizer's performance, ensuring high accuracy and reliability.
- **Number Bucketing**:  Numbers in the text are categorized into predefined "buckets" based on their value. The bucketing process involves dividing the number space into several ranges (or buckets) and assigning each number to a specific bucket. Each bucket is represented by its own token that follows specific convention. Common years (e.g., 1900-2025) and ages (e.g., 1-100) are exceptions to this rule and they are represented they way they are written. This reduces sparsity and improves generalization without overfitting to specific values
- **URL Replacement**: URLs in the text are identified using a regular expression for common URL patterns and replaced with a special token `<url>`.  Replacing varied URLs with a single token prevents the model from overfitting to specific web addresses, which are usually not relevant to understanding the text's general context.URLs can introduce a vast number of unique tokens into the vocabulary. Replacing them with a single token significantly simplifies the model's vocabulary. By abstracting away the specifics of URLs, models can focus more on the actual textual content.

## Tokenizer Evaluation Methodology

The evaluation of the tokenizer is crucial to ensure its effectiveness and accuracy. The approach used for evaluation relies on assessing the tokenizer's ability to accurately encode and decode textual data, aiming to measure how well the tokenizer can reproduce the original text after a round-trip of tokenization and detokenization. Here's a detailed explanation of how the loss function works and the significance of the evaluation scores:

### **Loss Function Breakdown**

1. **Tokenization**: Each sentence in the dataset is encoded to token IDs using `tokenizer.encode(example).ids`. This step converts text into a sequence of tokens that the model can understand.

2. **Detokenization**: The token IDs are then decoded back into text using `tokenizer.decode(tokenizer.encode(original).ids)`. This step attempts to reconstruct the original text from the token IDs.

3. **Distance Calculation**: For texts that do not match, the Levenshtein distance (a measure of the difference between two sequences) is calculated between the original and detokenized text, normalized by the original text length. This distance provides a quantitative measure of how much the texts differ.

5. **Loss**: The overall loss is computed as the average of these distances (`distance / round_trip_errors`), providing a single metric that reflects the tokenizer's accuracy in reproducing the original text.