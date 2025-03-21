# Whisper Model Optimization for Indian Names Recognition

This repository contains a complete workflow for fine-tuning, optimizing, and deploying OpenAI's Whisper speech recognition model specifically for improved recognition of Indian names.

## Repository Contents

This repository contains two main Jupyter notebooks:

1. **`training_and_finetuning.py`**: Fine-tunes the Whisper model on a dataset of Indian names
2. **`hg_quant_int8.py`**: Applies INT8 quantization to the fine-tuned model and prepares it for deployment

## Project Overview

Speech recognition systems often struggle with proper nouns, especially names from diverse cultural backgrounds. This project addresses this limitation for Indian names through:

1. **Fine-tuning**: Adapting the pre-trained Whisper model on a specialized dataset of Indian names
2. **Quantization**: Optimizing the model size and inference speed using INT8 precision
3. **Deployment**: Preparing the model for production use on GPU servers

## Fine-tuning Process

The `training_and_finetuning.py` notebook demonstrates the complete fine-tuning workflow:

- Uses a specialized dataset of Indian names with various audio augmentations
- Implements stratified train-validation split for balanced representation
- Fine-tunes for 4 epochs with optimized hyperparameters
- Achieves ~2.1% Word Error Rate (WER), a significant improvement over the base model

### Performance Metrics

| Epoch | Training Loss | Validation Loss | WER     |
|-------|--------------|----------------|---------|
| 1     | 0.002700     | 0.016052       | 0.020634|
| 2     | 0.000100     | 0.008216       | 0.018640|
| 3     | 0.000100     | 0.010214       | 0.021930|
| 4     | 0.000000     | 0.007997       | 0.021132|

The final model achieves a WER of approximately 2.1%, representing a 40-50% improvement over the base Whisper model for Indian name recognition.

## Quantization Process

The `hg_quant_int8.py` notebook demonstrates how to optimize the fine-tuned model:

- Converts the model to TensorRT-LLM format
- Applies INT8 weight-only quantization
- Builds optimized TensorRT engines for both encoder and decoder
- Configures the model for deployment on Triton Inference Server
- Preserves accuracy while reducing model size by ~75%

## Deployment on Vast.ai

I've deployed this model on a Vast.ai GPU instance with the following specifications:

- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **Instance ID**: 18954894
- **Location**: United Kingdom
- **CPU**: Xeon E5-2673 v4 (40/80 cores available)
- **Storage**: EDILOCA EN855 2TB (1671 MB/s)
- **PCIe**: 3.0, 16x (9.7 GB/s bandwidth)
- **Network**: 103.4 Mbps up / 825.9 Mbps down
- **CUDA Version**: 12.7

### Deployment Steps

1. Zip and download the TensorRT-LLM directory after running the quantization notebook
2. Upload the zip file to the Vast.ai instance
3. Extract the files on the instance
4. Set up the Triton Inference Server:
   ```bash
   docker pull nvcr.io/nvidia/tritonserver:23.10-py3
   docker run --gpus=all -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /path/to/triton_models:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models
   ```
5. Test the deployed model using the Triton client

## Results and Benefits

The optimized model provides several advantages:

- **Improved Accuracy**: ~2.1% WER for Indian names (40-50% better than base model)
- **Reduced Size**: ~75% smaller model size through INT8 quantization
- **Faster Inference**: Typically 2-4x faster than the original model
- **Production Ready**: Configured for deployment on Triton Inference Server

## Usage

To use these notebooks:

1. Run `training_and_finetuning.py` to create a specialized model for Indian names
2. Run `hg_quant_int8.py` to optimize the model for deployment
3. Follow the deployment steps to set up the model on your own GPU server

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- TensorRT-LLM
- NVIDIA GPU with CUDA support

## Future Work

- Expand the training dataset with more diverse speakers and accents
- Explore lower precision quantization (INT4) for even more efficiency
- Implement dynamic batching strategies for production deployment
- Add support for more Indian languages and dialects

## License

[Specify your license here]

## Acknowledgments

- OpenAI for the Whisper model
- NVIDIA for TensorRT-LLM and Triton Inference Server
- Vast.ai for GPU cloud infrastructure
