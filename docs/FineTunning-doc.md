# Fine-Tuning LLaMA 3.1-8B for Product Price Estimation
## A Comprehensive Technical Documentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JlO1njhRobolOsuwc0ZdbQpFWSA6uF-d#scrollTo=SY_Ctos2SirK)

---

## Executive Summary

This project successfully fine-tuned Meta's LLaMA 3.1-8B model for the specialized task of product price estimation using Amazon product data. Through Parameter-Efficient Fine-Tuning (PEFT) techniques, we achieved a **56% reduction in prediction error** and **90% accuracy** in price predictions, demonstrating the effectiveness of adapting large language models for domain-specific commercial applications.

**Key Achievements:**
- Reduced prediction error from $42.19 to $18.46 (56% improvement)
- Achieved 90% hit rate for accurate price predictions
- Successfully deployed model to Hugging Face Hub for inference
- Implemented memory-efficient training using QLoRA on Google Colab

---

## 1. Project Overview

### 1.1 Objective
The primary goal was to fine-tune a large language model to predict product prices based solely on textual product descriptions, creating an intelligent pricing system that could understand product features and market positioning.

### 1.2 Business Context
Accurate price estimation from product descriptions has significant applications in:
- E-commerce pricing optimization
- Market research and competitive analysis
- Automated product valuation
- Consumer price awareness tools

### 1.3 Technical Approach
We employed Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) combined with 4-bit quantization (QLoRA) to adapt the Meta-Llama-3.1-8B model for price prediction tasks.

---

**Google Colab** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JlO1njhRobolOsuwc0ZdbQpFWSA6uF-d#scrollTo=SY_Ctos2SirK)

## 2. Dataset and Data Processing

### 2.1 Dataset Source
**Primary Dataset:** `McAuley-Lab/Amazon-Reviews-2023` from Hugging Face
- **Focus Area:** Product metadata from various categories
- **Data Type:** Product descriptions, features, and pricing information
- **Scale:** Filtered from  Appliances categorie to ~25,000 training samples

### 2.2 Data Filtering and Quality Control

#### Price Range Constraints
```python
MIN_PRICE = 0.5   # Minimum product price
MAX_PRICE = 999.49 # Maximum product price
```

#### Content Quality Standards
- **Minimum Token Count:** 150 tokens (ensuring sufficient product information)
- **Maximum Token Count:** 160 tokens (optimized for model context)
- **Minimum Character Count:** 300 characters
- **Text Cleaning:** Removal of irrelevant metadata and standardization

#### Data Processing Pipeline
1. **Content Extraction:** Product titles, descriptions, features, and details
2. **Text Cleaning:** Removal of formatting artifacts, product codes, and noise
3. **Tokenization:** Using LLaMA tokenizer for consistent processing
4. **Quality Filtering:** Ensuring adequate information density
5. **Prompt Formatting:** Structured Q&A format for training

### 2.3 Prompt Structure
The training data followed a consistent instructional format:
```
How much does this cost to the nearest dollar?

[Product Title]
[Product Description and Features]

Price is $[Amount].00
```

---

## 3. Model Architecture and Configuration

### 3.1 Base Model
**Model:** `meta-llama/Meta-Llama-3.1-8B`
- **Parameters:** 8 billion parameters
- **Architecture:** Transformer-based causal language model
- **Context Length:** Optimized for sequences up to 182 tokens

### 3.2 Fine-Tuning Methodology

#### Parameter-Efficient Fine-Tuning (PEFT)
We implemented **LoRA (Low-Rank Adaptation)** to efficiently adapt the model:

```python
# LoRA Configuration Parameters
LORA_R = 32              # Rank of adaptation
LORA_ALPHA = 64          # Scaling parameter
LORA_DROPOUT = 0.1       # Dropout rate
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Attention layers
```

#### Quantization Strategy
**4-bit Quantization (QLoRA)** for memory efficiency:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```

### 3.3 Memory Optimization
- **Base Model Memory Footprint:** Significantly reduced through quantization
- **Training Memory:** Optimized for Google Colab GPU constraints
- **Mixed Precision:** BFloat16 training for efficiency

---

## 4. Training Configuration

### 4.1 Hyperparameters

| Parameter | Value | Rationale |
|-----------|--------|-----------|
| **Learning Rate** | 1e-4 (0.0001) | Conservative rate for stable fine-tuning |
| **Training Epochs** | 1 | Sufficient for task adaptation |
| **Batch Size** | 4 | Optimized for memory constraints |
| **Gradient Accumulation** | 1 step | Direct gradient updates |
| **Optimizer** | PagedAdamW 32-bit | Memory-efficient optimization |
| **LR Scheduler** | Cosine | Smooth learning rate decay |
| **Warmup Ratio** | 0.03 | Gradual training initialization |
| **Weight Decay** | 0.001 | Regularization |
| **Max Gradient Norm** | 0.3 | Gradient clipping |

### 4.2 Training Infrastructure
- **Platform:** Google Colab Pro
- **GPU:** Variable (typically T4 16GB)
- **Framework:** PyTorch with Transformers
- **Monitoring:** Weights & Biases integration

### 4.3 Framework Versions
```
PEFT: 0.15.2
Transformers: 4.52.3
PyTorch: 2.6.0+cu124
Datasets: 3.6.0
Tokenizers: 0.21.1
```

---

## 5. Training Results and Analysis

### 5.1 Training Metrics Evolution

The training process was monitored across approximately 1,200 steps, showing:

#### Token Processing
- **Initial Rate:** ~50,000 tokens/step
- **Final Rate:** ~850,000 tokens/step
- **Total Tokens Processed:** Approximately 500M tokens

#### Accuracy Progression
- **Initial Accuracy:** ~68%
- **Peak Accuracy:** ~71%
- **Final Accuracy:** Stabilized around 69-70%

#### Loss Reduction
- **Initial Loss:** ~1.7
- **Final Loss:** ~1.35
- **Convergence:** Smooth reduction indicating stable learning

#### Learning Rate Schedule
- **Warmup Phase:** Gradual increase to peak learning rate
- **Cosine Decay:** Smooth reduction following cosine schedule
- **Final LR:** Near-zero for fine convergence

### 5.2 Gradient Analysis
- **Gradient Norms:** Maintained between 2-6, indicating healthy training dynamics
- **Stability:** No gradient explosions observed
- **Regularization:** Effective gradient clipping at 0.3

---

## 6. Model Evaluation and Performance

### 6.1 Evaluation Methodology

#### Test Dataset
- **Size:** 250 samples from held-out test set
- **Selection:** Random sampling across price ranges
- **Coverage:** Diverse product categories and price points

#### Evaluation Metrics
1. **Mean Absolute Error (MAE):** Average prediction error in dollars
2. **Root Mean Squared Log Error (RMSLE):** Normalized error measurement
3. **Hit Rate:** Percentage of predictions within acceptable range

### 6.2 Performance Results

#### Fine-Tuned Model Performance
```
Predict Error: $18.46
RMSLE: 0.46
Hit Rate: 90.0%
```

#### Comparative Analysis: Base vs Fine-Tuned Model

| Metric | Base Model | Fine-Tuned Model | Improvement |
|--------|------------|------------------|-------------|
| **Prediction Error** | $42.19 | $18.46 | ↓ 56% reduction |
| **RMSLE** | 1.07 | 0.46 | ↓ 57% reduction |
| **Hit Rate** | 76.0% | 90.0% | ↑ 14% increase |

### 6.3 Error Analysis

#### Prediction Accuracy Distribution
- **High Accuracy (Green):** 90% of predictions within ±20% or ±$40
- **Moderate Accuracy (Orange):** 7% with moderate deviations
- **Low Accuracy (Red):** 3% with significant errors

#### Error Characteristics
- **Systematic Bias:** Minimal bias in predictions
- **Variance:** Low variance indicating consistent performance
- **Outliers:** Few extreme predictions, mostly for complex products

---

## 7. Model Architecture Details

### 7.1 Fine-Tuned Model Structure
```
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 4096)
        (layers): ModuleList(
          (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj, k_proj, v_proj, o_proj): lora.Linear4bit
              - LoRA rank: 32
              - LoRA alpha: 64
              - Dropout: 0.1
            )
          )
        )
      )
    )
  )
)
```

### 7.2 Inference Optimization

#### Advanced Prediction Strategy
The final model employs a sophisticated prediction mechanism:

```python
def improved_model_predict(prompt, top_k=3):
    # Generate top-k token probabilities
    # Weight predictions by probability
    # Return weighted average for robust estimation
```

**Benefits:**
- **Robustness:** Considers multiple probable outputs
- **Stability:** Reduces impact of single-token errors
- **Accuracy:** Weighted averaging improves precision

---

## 8. Deployment and Accessibility

### 8.1 Model Repository
**Hugging Face Hub:** `srajal87/pricer-lite-2025-06-04_17.39.50`
- **Accessibility:** Public repository for inference
- **Model Type:** PEFT adapter weights
- **Base Model:** Compatible with Meta-Llama-3.1-8B

### 8.2 Inference Usage
```python
# Load fine-tuned model
from peft import PeftModel
fine_tuned_model = PeftModel.from_pretrained(base_model, model_id)

# Generate prediction
prompt = "How much does this cost to the nearest dollar?\n\n[Product Description]\n\nPrice is $"
prediction = improved_model_predict(prompt)
```

### 8.3 Monitoring and Tracking
**Weights & Biases:** [Project Dashboard](https://wandb.ai/srajalsahu87-madhav-institute-of-technology-science-gwalior/pricer-lite)
- **Real-time Metrics:** Training loss, accuracy, learning rate
- **Resource Monitoring:** GPU utilization, memory usage
- **Experiment Tracking:** Comprehensive logging of all runs

---

## 9. Technical Challenges and Solutions

### 9.1 Memory Constraints
**Challenge:** Training 8B parameter model on limited GPU memory

**Solutions Implemented:**
- **4-bit Quantization:** Reduced memory footprint by ~75%
- **LoRA Fine-tuning:** Only train 0.3% of total parameters
- **Gradient Checkpointing:** Trade computation for memory
- **Mixed Precision:** BFloat16 for efficiency

### 9.2 Data Quality and Curation
**Challenge:** Ensuring high-quality training data from noisy web sources

**Solutions Implemented:**
- **Automated Filtering:** Token count and character length constraints
- **Text Cleaning:** Removal of metadata and formatting artifacts
- **Price Validation:** Filtering unrealistic price ranges
- **Content Verification:** Ensuring adequate product information

### 9.3 Training Stability
**Challenge:** Maintaining stable training with limited resources

**Solutions Implemented:**
- **Conservative Learning Rate:** Preventing divergence
- **Gradient Clipping:** Controlling gradient magnitudes
- **Cosine Scheduling:** Smooth learning rate decay
- **Regular Checkpointing:** Preventing loss of progress

---

## 10. Key Observations and Insights

### 10.1 Model Adaptation Efficiency
- **Rapid Convergence:** Model adapted quickly to pricing task
- **Stable Learning:** Consistent improvement without overfitting
- **Parameter Efficiency:** LoRA achieved excellent results with minimal parameter updates

### 10.2 Task-Specific Performance
- **Domain Understanding:** Model learned to associate product features with pricing
- **Context Awareness:** Effective use of product descriptions for price inference
- **Generalization:** Strong performance across diverse product categories

### 10.3 Technical Insights
- **Quantization Impact:** Minimal performance loss with significant memory savings
- **Architecture Suitability:** LLaMA architecture well-suited for this task
- **Training Efficiency:** Single epoch sufficient for effective adaptation

---

## 11. Future Enhancements and Recommendations

### 11.1 Model Improvements
- **Multi-Category Training:** Expand to additional product categories
- **Ensemble Methods:** Combine multiple fine-tuned models
- **Continual Learning:** Regular updates with new product data
- **Larger Context:** Utilize extended context for richer product information

### 11.2 Data Enhancements
- **Market Context:** Include competitive pricing information
- **Temporal Factors:** Account for seasonal pricing variations
- **Geographic Data:** Regional pricing variations
- **Review Integration:** Customer feedback as pricing signals

### 11.3 Deployment Optimizations
- **Model Compression:** Further reduce inference costs
- **API Development:** Production-ready inference endpoints
- **Batch Processing:** Efficient bulk price estimation
- **Real-time Updates:** Dynamic model updating capabilities

---

## 12. Conclusion

This project successfully demonstrates the practical application of fine-tuning large language models for specialized commercial tasks. The Meta-Llama-3.1-8B model, when adapted using Parameter-Efficient Fine-Tuning techniques, achieved remarkable performance in product price estimation with minimal computational overhead.

### Key Contributions:
1. **Methodological:** Effective application of QLoRA for commercial NLP tasks
2. **Technical:** Memory-efficient training pipeline for large models
3. **Practical:** Deployable solution for real-world pricing applications
4. **Performance:** Significant improvement over baseline approaches

### Impact:
The fine-tuned model represents a practical tool for e-commerce pricing, market analysis, and consumer applications, demonstrating the commercial viability of adapted large language models for specific business domains.

---

## References and Resources

### Technical Documentation
- **Hugging Face Transformers:** https://huggingface.co/docs/transformers
- **PEFT Library:** https://huggingface.co/docs/peft
- **LoRA Paper:** "LoRA: Low-Rank Adaptation of Large Language Models"
- **QLoRA Paper:** "QLoRA: Efficient Finetuning of Quantized LLMs"

### Model and Data Resources
- **Base Model:** Meta-Llama-3.1-8B on Hugging Face
- **Dataset:** McAuley-Lab/Amazon-Reviews-2023
- **Fine-tuned Model:** srajal87/pricer-lite-2025-06-04_17.39.50
- **Training Logs:** Weights & Biases Dashboard

### Development Environment
- **Platform:** Google Colab Pro
- **Framework:** PyTorch + Hugging Face Ecosystem
- **Version Control:** Git with Hugging Face Hub integration
- **Monitoring:** Weights & Biases MLOps platform

---