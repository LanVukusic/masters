# AGENTS.md - Development Guidelines for This Project

## Project Overview

This is a PyTorch-based audio deep learning project for training audio continuation models on DAC codec tokens. It uses `uv` as the package manager and Python 3.14. This is a masters thesis project.

## Build/Lint/Test Commands

### Package Installation
```bash
# Install dependencies with uv
uv sync

# Add a new dependency (avoid pip)
uv add <package>

# Install with specific GPU support (CUDA 13.0)
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```
**Important**: Never use `uv add pip` or `pip install`. Use `uv pip` only for specific index installs.

### Running the Training Script
```bash
# From root directory
uv run model_training/trainModel.py

# With custom arguments
uv run python model_training/trainModel.py --config configs/train.yaml
```

### Running Tests
```bash
# No test framework configured yet - pytest recommended for future
# pytest testiranje/ -v

# For now, test manually by running Python files:
uv run python testiranje/test_model.py
```

### Linting and Formatting
```bash
# Check linting (ruff is installed in the project)
uv run ruff check .

# Fix linting issues automatically
uv run ruff check . --fix

# Format code
uv run ruff format .
```

## Code Style Guidelines

### General Principles
- Write concise, readable code
- Avoid unnecessary comments - let code explain itself
- Use descriptive variable and function names
- Keep functions focused and small (under 50 lines when possible)

### Imports
- Use absolute imports (e.g., `from model_training.dataloader.raw_dataset import RawAudioDataset`)
- Add `sys.path.insert(0, 'model_training')` when running scripts from non-package directories
- Group imports: standard library, third-party, local application
- Avoid wildcard imports (`from X import *`)

### Formatting
- Maximum line length: 100 characters (ruff default)
- Use 4 spaces for indentation (not tabs)
- Use trailing commas in multi-line collections
- Use f-strings for string formatting (not `.format()` or `%`)

### Types
- Use type hints for function parameters and return values when they improve clarity
- Prefer concrete types over `Any` when possible
- Example:
  ```python
  def process_audio(tokens: torch.Tensor, sample_rate: int) -> torch.Tensor:
      ...
  ```

### Naming Conventions
- `snake_case` for functions, variables, and module names
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Prefix private methods with underscore: `_private_method()`

### Error Handling
- Use descriptive error messages
- Catch specific exceptions when possible
- Never silently swallow exceptions without logging
- Example:
  ```python
  try:
      result = tokenizer.encode(audio)
  except ValueError as e:
      raise ValueError(f"Failed to encode audio: {e}") from e
  ```

### PyTorch-Specific Guidelines
- Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` for device management
- Always move tensors to device with `.to(device, non_blocking=True)` for DataLoader batches
- Use `with torch.no_grad():` for inference
- Use `torch.no_grad()` context for validation and generation
- Prefer `nn.Module` subclasses over functional code
- Use `nn.Parameter` for learnable parameters
- Initialize weights properly (e.g., `nn.init.xavier_uniform_`)

### Project Structure
```
/home/lanv/masters/
├── model_training/
│   ├── dataloader/       # Dataset implementations (IterableDataset, etc.)
│   ├── model/            # Model definitions
│   ├── tokenizer/        # Audio tokenizers (DAC, Mimi)
│   ├── simpleModel/      # Working model implementations
│   ├── narTransformer/   # Transformer architectures
│   └── trainModel.py    # Main training script
├── research/             # Research notebooks and experiments
├── dataset_gen/          # Audio datasets
├── checkpoints/          # Model checkpoints
└── pyproject.toml        # Project configuration
```

### Working with Tokenizers
- When using DAC or Mimi tokenizers, process audio one sample at a time in batches (the tokenizer may squeeze batch dimension)
- Always use `original_sampling_rate` parameter when encoding
- Example batch processing:
  ```python
  tokens_list = []
  for i in range(raw_audio_gpu.shape[0]):
      single_audio = raw_audio_gpu[i:i+1]
      codes = tokenizer.encode_from_waveform(single_audio, original_sampling_rate=rate)
      tokens_list.append(codes)
  tokens = torch.cat(tokens_list, dim=0)
  ```

### Dependencies Notes
- **torchcodec**: Requires CUDA NPP libraries. If import fails with `libnppicc.so.X` error, install nvidia-npp or use CPU-only PyTorch
- **ffmpeg**: Required for audio decoding (already in system)
- **DAC codec**: Requires `descript-audio-codec` package

### Training Tips
- Always call `scheduler.step()` after each training step
- Use gradient clipping cautiously - start with `gradient_clip: 0.0` or high values (10.0+), then tune
- Monitor loss per batch during training
- Use `torch.set_grad_enabled(True/False)` appropriately
- Set DataLoader `num_workers=0` when using on-the-fly tokenization to avoid multiprocessing issues

### Logging
- Use tensorboard via `SummaryWriter` for training metrics
- Log audio samples periodically for qualitative evaluation
- Use meaningful metric names: `Train/Loss`, `Train/LR`, `Train/GradNorm`

### Common Pitfalls
1. **Tokenizer batch handling**: Some tokenizers expect `[channels, samples]`, not `[batch, channels, samples]`
2. **Scheduler not stepping**: Remember to call `scheduler.step()` after `optimizer.step()`
3. **Gradient explosion**: If loss stays constant around ~ln(vocab_size), gradients may be clipped too aggressively
4. **CUDA out of memory**: Reduce batch size or sequence length if needed
5. **DataLoader num_workers**: Set to 0 for on-the-fly tokenization to avoid multiprocessing issues
6. **torchcodec import**: Ensure nvidia-npp is installed or use compatible CUDA version

## Testing Guidelines
- Create `testiranje/` directory for tests
- Use pytest framework (add to dependencies if needed)
- Use descriptive test function names: `test_model_predict_returns_correct_shape`
- Mock expensive operations (tokenizers) when possible
- Test both success cases and edge cases
