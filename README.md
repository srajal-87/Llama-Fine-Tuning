# Fine-Tuning LLaMA 3.1-8B for Product Price Estimation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JlO1njhRobolOsuwc0ZdbQpFWSA6uF-d#scrollTo=SY_Ctos2SirK)

A machine learning project that fine-tunes Meta's LLaMA 3.1-8B model to estimate product prices based on product descriptions, features, and details from Amazon product data.

## 🎯 Project Overview

This project demonstrates how to fine-tune a large language model (LLaMA 3.1-8B) for the specific task of product price estimation. The model learns to predict product prices by analyzing product titles, descriptions, and features.

### Key Results
- **Improved Model**: $18.46 RMSE, 0.46 RMSLE, 90.0% accuracy
- **Base Model**: $40.61 RMSE, 1.21 RMSLE, 82.0% accuracy

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/llama-price-estimation.git
   cd llama-price-estimation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run setup script**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

## 📊 Dataset

The project uses the **Amazon Reviews 2023** dataset from McAuley-Lab:
- Source: `McAuley-Lab/Amazon-Reviews-2023`
- Format: Product metadata with titles, descriptions, features, and prices
- Price range: $0.50 - $999.49
- Token range: 150-160 tokens per item

### Data Processing Pipeline

1. **Raw Data Loading**: Load Amazon product metadata
2. **Price Filtering**: Filter products within price range
3. **Text Cleaning**: Remove irrelevant text and normalize
4. **Tokenization**: Convert to model-compatible format
5. **Prompt Generation**: Create training prompts

## 🔧 Model Configuration

**Base Model**: `meta-llama/Meta-Llama-3.1-8B`

**Key Parameters**:
- Min tokens: 150
- Max tokens: 160
- Min characters: 300
- Price range: $0.50 - $999.49

**Training Setup**:
- Fine-tuning approach: LoRA (Low-Rank Adaptation)
- Batch processing: 1000 items per chunk
- Parallel processing: 3 workers default

## 📈 Results

### Performance Metrics

| Model | RMSE | RMSLE | Accuracy |
|-------|------|-------|----------|
| Base Model | $40.61 | 1.21 | 82.0% |
| Fine-tuned | $18.46 | 0.46 | 90.0% |

### Training Progress
- **Tokens processed**: 800K+
- **Training steps**: 1,200+
- **Final loss**: ~1.3
- **Mean token accuracy**: ~70%

## 🔬 Monitoring & Visualization

Training progress is tracked using **Weights & Biases**:
- **Dashboard**: [W&B Project Link](https://wandb.ai/srajalsahu87-madhav-institute-of-technology-science-gwalior/pricer-lite?nw=nwusersrajalsahu87)
- **Metrics**: Loss, accuracy, learning rate, gradient norm
- **Visualizations**: Training curves, prediction scatter plots


## 📋 Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
peft>=0.4.0
wandb>=0.15.0
tqdm>=4.65.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```
## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for LLaMA 3.1-8B model
- **McAuley-Lab** for Amazon Reviews 2023 dataset
- **Hugging Face** for transformers library
- **Weights & Biases** for experiment tracking



**Training Dashboard**: [W&B Pricer-Lite Project](https://wandb.ai/srajalsahu87-madhav-institute-of-technology-science-gwalior/pricer-lite?nw=nwusersrajalsahu87)

---

⭐ **Star this repository if you find it helpful!**
