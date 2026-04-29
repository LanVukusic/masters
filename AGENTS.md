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
**Important**: Never use `uv add pip` or `pip install`.

### Running the Training Script
```bash
uv run python model_training/train.py
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

## Canonical Shape Contract

Use this contract consistently in tokenizer, dataset, logging, and training code.

- Raw waveform input: `[B, 1, samples]`
- Tokenizer.encode input: `[B, 1, samples]`
- Tokenizer.encode output: `[B, num_quantizers, time_steps]`
- Tokenizer.decode input: `[B, num_quantizers, time_steps]`
- Tokenizer.decode output: `[B, 1, samples]`
- Dataset output: `{"past": [B, K, T_past], "future": [B, K, T_future]}`
- Model boundary: transpose once from `[B, K, T]` -> `[B, T, K]`

### Agent guidance

- Preserve the external token shape `[B, K, T]` everywhere outside the model.
- Keep tokenizer code limited to encoding/decoding and dataset code limited to raw waveform chunking and token batching.
- Avoid adding `squeeze`, `unsqueeze`, or extra `transpose` operations unless the canonical contract requires it.
- Use shared conversion helpers in `model_training.model_config` for all sample/token/second/chunk math.
- Perform token-axis transformations only once at the model input/output boundary.

## Minimal Style Guidelines

- Use absolute imports for local modules.
- Keep functions focused and clear.
- Prefer concrete type hints where useful.
- Use 4 spaces for indentation and a max line length of 100.
- Use f-strings for formatting.

## Training Notes

- Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Use `num_workers=0` in DataLoader for on-the-fly tokenization.
- Call `scheduler.step()` after `optimizer.step()`.
- Clip gradients only when needed.
- Use `with torch.no_grad():` for evaluation and decoding.

## Logging and Tokens

- Token tensors should be `[B, K, T]` before the model boundary.
- Model inputs are transposed to `[B, T, K]` for the model.
- Decode tokens with the tokenizer only once per logging pass.
- Log audio with `SummaryWriter` after decoder output.

## Common Pitfalls

- Do not let agents overcomplicate dataloader/tokenizer shape handling.
- Avoid repeated token reshaping outside the documented contract.
- Keep one canonical batch shape and transpose only at the model interface.

## Testing Guidelines
- Create `testiranje/` directory for tests
- Use pytest framework (add to dependencies if needed)
- Use descriptive test function names: `test_model_predict_returns_correct_shape`
- Mock expensive operations (tokenizers) when possible
- Test both success cases and edge cases
