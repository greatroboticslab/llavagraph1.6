# MSEC

## Project files summary

### Data Processing (Test Data)
1. **Extract** → `extract_txt_file.py` → Raw text files
2. **Process** → `process_raw_8.py` → PNG + CSV in nanometers
3. **Extract CSVs** → `extract_csv.py` → Isolated CSV files
4. **Augment** → `Augmented_final.py` → Synthetic training data
5. **data folder** → `data_filtered` : train and test data

### Model Training
5. **Prepare** → `gemini_train.py` → `trainingdata_gemini.json`
6. **Configure** → Edit `finetune2.2.sh` paths
7. **Train** → `bash finetune2.2.sh` → Fine-tuned LLaVA model

### Evaluation
8. **Setup** → Download LLaMA 3.2 3B
9. **Caption** → `evaluateLLaVA.sh` → Image descriptions (llava_V6/)
10. **Classify** → `categorizeLLAMA.sh` → Final classifications (llama_V6/)

---



# 🌋📊 LLaVAGraph

![lLLaVAGraph](https://github.com/user-attachments/assets/5db7aca4-443a-40e9-b8e6-18edd7b83b13)

`LLaVAGraph` is a multimodal agentic framework designed for classifying graphs of laser displacement data from piezoelectric actuators.

_Abstract:_ Piezoelectric actuators have been used extensively throughout multiple industries. The characterization of piezoelectric actuators is important to ensure their accuracy. This research not only measures the displacement of piezoelectric actuators but also classifies different patterns of motion using large language models. The use of large language models offers significant advantages by not only being capable of precise motion classification but also answering potential questions related to piezoelectric actuator motion properties, thus making it a useful tool for practical manufacturing process quality control.

## Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Processing Pipeline](#data-processing-pipeline)
  - [Step 1: Extract Raw Data](#step-1-extract-raw-data)
  - [Step 2: Process Raw Data to Physical Units](#step-2-process-raw-data-to-physical-units)
  - [Step 3: Extract CSV Files](#step-3-extract-csv-files)
  - [Step 4: Generate Synthetic Data](#step-4-generate-synthetic-data)
- [Model Training](#model-training)
  - [Prepare Training Data](#prepare-training-data)
  - [Configure Training Parameters](#configure-training-parameters)
  - [Run Training](#run-training)
- [Evaluation](#evaluation)
  - [Setup Evaluation Environment](#setup-evaluation-environment)
  - [Generate Image Captions](#generate-image-captions)
  - [Final Classification](#final-classification)

---

## Overview

LLaVAGraph processes piezoelectric actuator displacement data through a complete pipeline:

1. **Data Processing**: Convert raw instrument measurements to physical units (nanometers)
2. **Data Augmentation**: Generate synthetic training data to improve model robustness
3. **Training**: Fine-tune LLaVA model to caption displacement graphs
4. **Evaluation**: Use LLaMA to classify captions into waveform types (sine, square, random noise)

---

## Installation

### 1. Create Python Environment

```bash
# Create and activate conda environment
conda create -n llava python=3.10
conda activate llava

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Upgrade pip
pip install --upgrade pip

# Install LLaVA in editable mode
pip install -e .
```

### 2. Install Training Dependencies

```bash
# Install training packages
pip install -e ".[train]"

# Install flash attention (required for efficient training)
pip install flash-attn --no-build-isolation

# Install DeepSpeed for distributed training
pip install deepspeed
```

### 3. Install Data Processing Dependencies

```bash
# Install data processing packages
pip install ttkbootstrap matplotlib pandas numpy

# Install time series augmentation library
pip install tsaug
```


## Data Processing Pipeline

This section covers the complete workflow from raw instrument data to training-ready datasets.

### Step 1: Extract Raw Data

**Script**: `extract_txt_file.py`

**Purpose**: Extract original text files from the raw data directory to prepare them for processing.

**Input**: 
- `Issac_data_Aug_27_2024/`: Raw experimental measurements from instruments

**Output**: 
- `Issac_data_Aug_27_2024/Original_only_txt/`: Extracted text files

**Usage**:

```bash
python extract_txt_file.py
```

**What it does**:
- Scans the `Issac_data_Aug_27_2024/` folder
- Extracts all `.txt` files containing raw displacement measurements
- Creates `Original_only_txt` directory with organized text files
- These files serve as the primary data source for the processing pipeline

---

### Step 2: Process Raw Data to Physical Units

**Script**: `process_raw_8.py`

**Purpose**: Convert raw instrument readings to nanometers using calibration formulas.

**Input**: 
- `Issac_data_Aug_27_2024/Original_only_txt/`: Raw text files

**Output**: 
- `output_file/`: Directory containing processed data
  - PNG files: Visualizations of displacement graphs
  - CSV files: Numeric data in nanometers

**Conversion Formula**:
```
nm = (D - baseline) * (wavelength / 8) - correction
```

Where:
- `D`: Raw displacement value
- `baseline`: Calibration baseline
- `wavelength`: Laser wavelength
- `correction`: System correction factor

**Usage**:

```bash
python process_raw_8.py
```

**Configuration**:
1. Open `process_raw_8.py`
2. Ensure input path points to: `Issac_data_Aug_27_2024/Original_only_txt/`
3. Select "Absolute" mode for calibrated output
4. Run the script

**Dependencies**: 
- `ttkbootstrap`: For GUI interface
- `matplotlib`: For visualization

**Reference**: For detailed implementation, see: https://github.com/greatroboticslab/laserai

---

### Step 3: Extract CSV Files

**Script**: `extract_csv.py`

**Purpose**: Isolate CSV files from the processed output directory for augmentation.

**Input**: 
- `output_file/`: Contains both PNG and CSV files

**Output**: 
- Extracted CSV files ready for synthetic data generation

**Usage**:

```bash
python extract_csv.py
```

**What it does**:
- Scans the `output_file/` directory
- Filters and extracts only CSV files
- Organizes CSV files for the next stage (data augmentation)

---

### Step 4: Generate Synthetic Data

**Script**: `Augmented_final.py`

**Purpose**: Generate synthetic training data through time series augmentation to improve model robustness.

**Input**: 
- CSV files containing waveform data points

**Output**: 
For each input CSV file, generates:
- **1 original visualization**: Accurate representation of source data
- **6 synthetic augmentations**: Modified versions for data diversity

**Augmentation Methods**:

1. **Time Warp**: Modifies temporal dynamics by applying speed changes
   - Mild: `max_speed_ratio=1.2`
   - Moderate: `max_speed_ratio=1.5`
   - Strong: `max_speed_ratio=1.8`

2. **Reverse**: Reverses the time series sequence (temporal flipping)

3. **Pool**: Reduces data resolution through downsampling
   - Size 2: Downsample by factor of 2
   - Size 3: Downsample by factor of 3

**Configuration**:

Edit the following parameters at the bottom of `Augmented_final.py`:

```python
CSV_INPUT_PATH = "path/to/your/csv/*.csv"  # Path pattern for CSV files
OUTPUT_FOLDER_NAME = "OutputWave"           # Output folder name
POINTS_TO_USE = 250                         # Number of data points to use
```

**Usage**:

```bash
python Augmented_final.py
```

**Installation Prerequisites**:

```bash
pip install tsaug
```

**References**:
- https://github.com/arundo/tsaug
- https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html

---

## Model Training

### Prepare Training Data

**Script**: `gemini_train.py`

**Purpose**: Generate training data in the required JSON format for LLaVA model training.

**Input**: 
- `data_filtered/`: New train and test data used for training

**Output**: 
- `trainingdata_gemini.json`: Formatted training data for LLaVA

**JSON Format**:

```json
  {
    "id": "noise_noise_2_Run1_1_absolute_reverse_first250",
    "image": "train/noise/noise_2_Run1_1_absolute_reverse_first250.png",
    "conversations": [
      {
        "from": "human",
        "value": "Describe the repetition and density of the signal across the timeline."
      },
      {
        "from": "gpt",
        "value": "** The signal features intermittent, sharp spikes above a relatively constant baseline. These spikes appear in clustered groups separated by periods of baseline activity.  The density of the spikes is not uniform across the timeline; they are clustered around roughly the 0-25ms, 75-125ms, and 175-225ms intervals. Within these intervals the spikes are closely spaced.  The rest of the time range displays the constant baseline.\n\n**"
      },
      {
        "from": "human",
        "value": "Identify the shape of the most prominent 'edges' or 'transitions'."
      },
      {
        "from": "gpt",
        "value": "** The edges or transitions of the spikes are very abrupt and nearly vertical, both on the rising and falling edges. This suggests a very rapid change in displacement when the signal rises and falls. The transitions look almost like step functions, indicating instantaneous jumps in displacement.\n\n**"
      },
      {
        "from": "human",
        "value": "State the vertical range (min/max) and the polarity (positive/negative) of the displacement."
      },
      {
        "from": "gpt",
        "value": "** The vertical range of the displacement spans from a minimum of approximately -320 nm to a maximum of approximately -240 nm. The polarity of the displacement is entirely negative, as the signal values all lie below zero on the y-axis."
      }
```

**Usage**:

```bash
python gemini_train.py
```

**What it does**:
- Reads images from `data_filtered/`
- Generates question-answer pairs for each image
- Outputs `trainingdata_gemini.json` with structured conversation data

---

### Configure Training Parameters

**Script**: `finetune2.2.sh`

**Purpose**: Fine-tune the LLaVA model using LoRA (Low-Rank Adaptation) for efficient training.

**Before running**, modify the following paths in `finetune2.2.sh`:

```bash
#!/bin/bash

deepspeed llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
```

**Key Parameters**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--lora_enable` | True | Enable LoRA for efficient fine-tuning |
| `--lora_r` | 128 | LoRA rank (controls model capacity) |
| `--lora_alpha` | 256 | LoRA scaling factor |
| `--data_path` | path to JSON | Training data file |
| `--image_folder` | path to images | Directory containing training images |
| `--output_dir` | checkpoint path | Where to save model checkpoints |
| `--num_train_epochs` | 1 | Number of training epochs |
| `--learning_rate` | 2e-4 | Learning rate for training |

**Required Files**:
- `trainingdata_gemini.json`: Training data
- `zero3.json`: DeepSpeed configuration
- Training and testing images in `data_filtered/`

**Download Training Data**:

All required training files are available at:
https://drive.google.com/file/d/1amdSPdiPv1uonQpTGUKBgJOA7TGd3wYy/view?usp=drive_link

---

### Run Training

```bash
bash finetune2.2.sh
```

**Training Process**:
1. Model loads pre-trained LLaVA weights
2. LoRA adapters are initialized
3. Training begins with batched image-text pairs
4. Checkpoints are saved every 50,000 steps
5. Final model is saved to `output_dir`

**Expected Output**:
- Training logs showing loss reduction
- Model checkpoints in the specified output directory
- Final fine-tuned model ready for evaluation


---

## Evaluation

### Setup Evaluation Environment

Evaluation requires a separate environment for running LLaMA 3.2 3B.

#### 1. Request Model Access

Visit https://huggingface.co/meta-llama/Llama-3.2-3B and request access to the model.

> **Note**: Approval typically takes less than 1 hour.

#### 2. Create Evaluation Environment

```bash
# Create new virtual environment
conda create -n llava-eval python=3.10
conda activate llava-eval

# Install Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Login to Hugging Face
huggingface-cli login
```

#### 3. Download LLaMA Model

```bash
# Download LLaMA 3.2 3B Instruct model
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
    --exclude "original/*" \
    --local-dir Llama-3.2-3B-Instruct
```


### Generate Image Captions

**Script**: `evaluateLLaVA.sh`

**Purpose**: Use the fine-tuned LLaVA model to generate descriptive captions for all test images.

**Before running**, update the paths in `evaluateLLaVA.sh`:

```bash
#!/bin/bash

# Path to your fine-tuned LLaVA model
MODEL_PATH="<path-to-fine-tuned-model>"

# Path to test images
IMAGE_FOLDER="<path-to-test-images>"

# Output directory
OUTPUT_DIR="eval/results/llava"

```

**Usage**:

```bash
bash evaluateLLaVA.sh
```

**Output**: 

The script generates three JSON files in `eval/results/llava/` (now `llava_V6/`):
- `randomNoise.json`: Captions for random noise graphs
- `sineWave.json`: Captions for sine wave graphs
- `squareWave.json`: Captions for square wave graphs

**Example Output**:

```json
 {
    "image": "noise_8_Run2_1_absolute_timewarp_strong_first250.png",
    "conversation": [
      {
        "question": "Describe the repetition and density of the signal across the timeline.",
        "answer": "<s> The signal exhibits a clear repetitive pattern across the timeline. It appears to be a periodic signal, with a consistent frequency. The density of the signal is high, with numerous cycles occurring within the 250 ms timeframe. The signal is not a constant value, but rather a series of peaks and troughs. The title suggests that the signal is a sine wave, which is known for its periodic nature and smooth transitions."
      },
      {
        "question": "Identify the shape of the most prominent 'edges' or 'transitions'.",
        "answer": "<s> The most prominent 'edges' or 'transitions' are characterized by sharp, near-vertical ascents and descents. The signal rapidly transitions between the minimum and maximum displacement values. The edges are not perfectly vertical, but they are close, indicating a relatively abrupt change in displacement. This suggests a high-frequency component in the signal."
      },
      {
        "question": "State the vertical range (min/max) and the polarity (positive/negative) of the displacement.",
        "answer": "<s>  The vertical range of the displacement is approximately -50 nm to -250 nm. The polarity of the displacement is entirely negative, as all values are below zero."
      }
    ]
  },
```


---

### Final Classification

**Script**: `categorizeLLAMA.sh`

**Purpose**: Use LLaMA model to read the captions and classify each graph into waveform types.

**Reference Implementation**: 
- https://github.com/greatroboticslab/LlaVAGraph/blob/main/eval/categorizeLLAMA.sh
- https://github.com/greatroboticslab/LlaVAGraph/blob/main/eval/categorizeLLAMA.py

**Before running**, update paths in `categorizeLLAMA.sh`:

**note**
When running llama  
```bash
pip install --upgrade transformers  

python -c "import transformers; print(transformers.__version__)"  
For Llama-3.2, it is recommended to use version 4.45.0 or above.
```

**Usage**:

```bash
bash categorizeLLAMA.sh
```

**Output**: 

The script generates classification results in `eval/results/llama/` (now `llama_V6/`):

**Example Classification Result**:

```json
  {
      "image": "square_1hz_1hz_5_absolute_timewarp_mild_first250.png",
      "gt": "square",
      "pred": "square",
      "is_correct": true,
      "reasoning": "Result: C"
    }
```

**Classification Options**:
- A) Random noise
- B) Sine wave
- C) Square wave


---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `per_device_train_batch_size` in `finetune2.2.sh`
- Enable gradient checkpointing (already enabled by default)
- Use smaller image resolution

**2. Module Not Found Errors**
```bash
# Ensure you're in the correct environment
conda activate llava

# Reinstall dependencies
pip install -e ".[train]"
```

**3. DeepSpeed Configuration Issues**
- Verify `zero3.json` path is correct
- Check GPU availability: `nvidia-smi`
- Ensure CUDA version matches PyTorch installation

**4. Evaluation Scripts Not Running**
- Confirm you're on a GPU node: `srun --pty --partition=gpu bash`
- Verify model paths in shell scripts
- Check file permissions: `chmod +x *.sh`

---

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 24GB+ VRAM (for training)
- **CPU**: Multi-core processor
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for models and data

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10
- **CUDA**: 12.1+
- **PyTorch**: Compatible with CUDA version
