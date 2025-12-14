import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.frame import FrameModel
from model.depth import DepthModel
from model.inputDepth import InputDepthModel


class JointAudioContinuationModel(nn.Module):
    """
    Joint Model: Combines Frame Model, Depth Model, and Input Depth Model
    for end-to-end audio continuation.

    Based on patent sections [00036], [00062]:
    - Jointly trained components operating on shared embedding space
    - Closed-loop generation for multiple future frames
    - Optimized for computational efficiency
    """

    def __init__(
        self,
        history_length: int = 5,  # Historical RVQ tensors to use
        future_frames: int = 2,  # Number of future frames to generate
        rvq_levels: int = 12,  # RVQ codebook levels
        embedding_dim: int = 512,
        frame_model_layers: int = 6,
        depth_model_layers: int = 2,
        input_depth_model_layers: int = 3,
        codebook_size: int = 1024,
    ):
        super().__init__()
        self.history_length = history_length
        self.future_frames = future_frames
        self.rvq_levels = rvq_levels
        self.embedding_dim = embedding_dim

        # Initialize components
        self.frame_model = FrameModel(
            rvq_levels=rvq_levels,
            embedding_dim=embedding_dim,
            num_layers=frame_model_layers,
            future_frames=future_frames,
        )

        self.depth_model = DepthModel(
            rvq_levels=rvq_levels,
            embedding_dim=embedding_dim,
            num_layers=depth_model_layers,
            codebook_size=codebook_size,
        )

        self.input_depth_model = InputDepthModel(
            rvq_levels=rvq_levels,
            embedding_dim=embedding_dim,
            num_layers=input_depth_model_layers,
            codebook_size=codebook_size,
        )

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self, historical_rvq: torch.Tensor, future_rvq: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            historical_rvq: (batch_size, history_length, rvq_levels)
                Historical RVQ tokens (5 frames of history)
            future_rvq: Optional (batch_size, future_frames, rvq_levels)
                Ground truth future RVQ tokens for training

        Returns:
            Dictionary containing:
            - generated_rvq: (batch_size, future_frames, rvq_levels)
            - total_loss: scalar if future_rvq provided
            - frame_token_loss: scalar
            - token_level_losses: List of losses per RVQ level
        """

        # Step 1: Generate frame tokens for future frames
        future_frame_tokens = self.frame_model(
            historical_rvq
        )  # (batch_size, future_frames, embedding_dim)

        # Step 2: Generate RVQ tokens for each future frame
        generated_frames = []
        all_logits = []

        for frame_idx in range(self.future_frames):
            frame_token = future_frame_tokens[
                :, frame_idx
            ]  # (batch_size, embedding_dim)

            if future_rvq is not None:
                # Training mode with teacher forcing
                target_frame = future_rvq[:, frame_idx] if self.training else None
                generated_frame, logits = self.depth_model(frame_token, target_frame)
            else:
                # Inference mode
                generated_frame, logits = self.depth_model(frame_token)

            generated_frames.append(generated_frame)
            all_logits.append(logits)

        # Stack generated frames
        generated_rvq = torch.stack(
            generated_frames, dim=1
        )  # (batch_size, future_frames, rvq_levels)

        output = {
            "generated_rvq": generated_rvq,
        }

        # Calculate losses if ground truth provided
        if future_rvq is not None and self.training:
            total_loss = 0
            token_level_losses = []

            # Calculate loss for each RVQ level and frame
            for frame_idx in range(self.future_frames):
                for level in range(self.rvq_levels):
                    logits = all_logits[frame_idx][level]  # (batch_size, codebook_size)
                    targets = future_rvq[:, frame_idx, level]  # (batch_size,)
                    loss = self.ce_loss(logits, targets)
                    total_loss += loss
                    token_level_losses.append(loss.item())

            # Frame token consistency loss (optional but recommended per patent)
            if self.future_frames > 1:
                # Create input frame tokens from generated frames
                input_frame_tokens = []
                for frame_idx in range(self.future_frames):
                    frame_tokens = generated_rvq[
                        :, frame_idx
                    ]  # (batch_size, rvq_levels)
                    input_frame_token = self.input_depth_model(frame_tokens)
                    input_frame_tokens.append(input_frame_token)

                # Stack input frame tokens
                predicted_input_tokens = torch.stack(
                    input_frame_tokens, dim=1
                )  # (batch_size, future_frames, embedding_dim)

                # Consistency loss between predicted and actual frame tokens
                frame_token_loss = F.mse_loss(
                    predicted_input_tokens[:, :-1], future_frame_tokens[:, 1:].detach()
                )
                total_loss += frame_token_loss * 0.1  # Weight the consistency loss
            else:
                frame_token_loss = torch.tensor(0.0)

            output.update(
                {
                    "total_loss": total_loss,
                    "frame_token_loss": frame_token_loss,
                    "token_level_losses": token_level_losses,
                }
            )

        return output

    def generate(
        self,
        historical_rvq: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inference mode: Generate future RVQ tokens given historical context

        Args:
            historical_rvq: (batch_size, history_length, rvq_levels)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            generated_rvq: (batch_size, future_frames, rvq_levels)
        """
        self.eval()
        with torch.no_grad():
            return self(historical_rvq)["generated_rvq"]


if __name__ == "__main__":
    from torchinfo import summary
    
    # Create model instance with typical parameters
    model = JointAudioContinuationModel(
        history_length=5,
        future_frames=2,
        rvq_levels=12,
        embedding_dim=512,
        frame_model_layers=6,
        depth_model_layers=2,
        input_depth_model_layers=3,
        codebook_size=1024
    )
    
    print("JointAudioContinuationModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass with correct dimensions
    historical_rvq = torch.randint(0, 1024, (1, 5, 12))  # (batch_size=1, history_length=5, rvq_levels=12)
    output = model(historical_rvq)
    print(f"\nForward pass test - Input: {historical_rvq.shape}")
    print(f"Output keys: {list(output.keys())}")
    if 'generated_rvq' in output:
        print(f"Generated RVQ shape: {output['generated_rvq'].shape}")
    
    # Fixed summary call with proper input dimensions
    dummy_input = torch.randint(0, 1024, (1, 5, 12), dtype=torch.long)  # Correct 3D shape
    summary(
        model,
        input_data=(dummy_input,),  # Pass as tuple for positional arguments
        device="cpu",
        verbose=1,
        col_names=["input_size", "output_size", "num_params"]
    )
