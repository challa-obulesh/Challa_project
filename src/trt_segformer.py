"""
TensorRT-accelerated SegFormer inference for Jetson AGX Orin.

Provides:
  - ONNX export from HuggingFace SegFormer
  - TensorRT FP16 engine building
  - Fast GPU inference with torch buffer management
  - Automatic fallback to PyTorch FP16 if TRT is unavailable

Usage:
    model = SegFormerTRT("segformer_b0.engine")
    seg_map = model(bgr_frame)  # returns (H, W) uint8 class IDs
"""

import os
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


# ── Constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_CLASSES = 19  # Cityscapes


class SegFormerTRT:
    """
    TensorRT SegFormer-B0 inference optimised for Jetson.

    Falls back to PyTorch FP16 if no TRT engine is found.
    """

    def __init__(
        self,
        engine_path: str = None,
        input_size: tuple = (512, 512),
        model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    ):
        """
        Args:
            engine_path: Path to serialised TensorRT engine file.
            input_size:  (H, W) input resolution for the model.
            model_name:  HuggingFace model ID (used for fallback).
        """
        self.input_h, self.input_w = input_size
        self.model_name = model_name
        self.use_trt = False

        if engine_path and os.path.exists(engine_path) and TRT_AVAILABLE:
            self._load_trt_engine(engine_path)
        else:
            if engine_path and not os.path.exists(engine_path):
                print(f"[SegFormerTRT] Engine not found: {engine_path}")
            print("[SegFormerTRT] Falling back to PyTorch FP16")
            self._load_pytorch_model()

    # ── TensorRT Loading ─────────────────────────────────────────────────

    def _load_trt_engine(self, engine_path: str):
        """Load a serialised TensorRT engine."""
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        print(f"[SegFormerTRT] Loading TRT engine: {engine_path}")
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialise engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Discover I/O tensor names and shapes
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            print(f"  [{mode.name}] {name}: shape={list(shape)}, dtype={dtype}")
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        # Pre-allocate GPU buffers
        in_shape = self.engine.get_tensor_shape(self.input_name)
        out_shape = self.engine.get_tensor_shape(self.output_name)

        # Determine dtype
        trt_dtype = self.engine.get_tensor_dtype(self.input_name)
        torch_dtype = torch.float16 if trt_dtype == trt.float16 else torch.float32

        self.input_buffer = torch.zeros(
            list(in_shape), dtype=torch_dtype, device="cuda"
        )
        self.output_buffer = torch.zeros(
            list(out_shape), dtype=torch.float32, device="cuda"
        )

        self.stream = torch.cuda.Stream()
        self.use_trt = True
        print(f"[SegFormerTRT] TRT engine loaded — FP16={'16' in str(trt_dtype)}")

    # ── PyTorch Fallback ─────────────────────────────────────────────────

    def _load_pytorch_model(self):
        """Load SegFormer via HuggingFace and convert to FP16."""
        from transformers import SegformerForSemanticSegmentation

        self.pt_model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name
        )
        self.pt_model.eval().half().cuda()
        print(f"[SegFormerTRT] PyTorch FP16 model loaded: {self.model_name}")

    # ── Preprocessing ────────────────────────────────────────────────────

    def preprocess(self, bgr_frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a BGR frame for SegFormer inference using GPU to accelerate.
        """
        # Resize
        resized = cv2.resize(bgr_frame, (self.input_w, self.input_h),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Move to GPU and do operations there
        tensor = torch.from_numpy(rgb).cuda().float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3)
        
        tensor = (tensor - mean) / std
        # HWC -> CHW, add batch dim
        return tensor.permute(2, 0, 1).unsqueeze(0)

    # ── Inference ────────────────────────────────────────────────────────

    def _infer_trt(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run TensorRT inference."""
        # Copy input to pre-allocated buffer
        dtype = self.input_buffer.dtype
        self.input_buffer.copy_(input_tensor.to(dtype))

        # Set tensor addresses
        self.context.set_tensor_address(
            self.input_name, self.input_buffer.data_ptr()
        )
        self.context.set_tensor_address(
            self.output_name, self.output_buffer.data_ptr()
        )

        # Execute
        self.context.execute_async_v3(
            stream_handle=self.stream.cuda_stream
        )
        self.stream.synchronize()

        return self.output_buffer

    def _infer_pytorch(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run PyTorch FP16 inference."""
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.pt_model(pixel_values=input_tensor.half())
        return outputs.logits.float()

    # ── Public API ───────────────────────────────────────────────────────

    def __call__(
        self,
        bgr_frame: np.ndarray,
        original_size: tuple = None
    ) -> np.ndarray:
        """
        Run full segmentation pipeline on a BGR frame.

        Args:
            bgr_frame:     (H, W, 3) uint8 BGR image.
            original_size: (H, W) to upsample output to. Default: input frame size.

        Returns:
            (H, W) uint8 segmentation map with Cityscapes class IDs.
        """
        if original_size is None:
            original_size = bgr_frame.shape[:2]

        # Preprocess
        input_tensor = self.preprocess(bgr_frame)

        # Infer
        if self.use_trt:
            logits = self._infer_trt(input_tensor)
        else:
            logits = self._infer_pytorch(input_tensor)

        # Postprocess: upsample + argmax
        logits_up = F.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        seg_map = logits_up.argmax(dim=1).squeeze(0).byte().cpu().numpy()
        return seg_map


# ── ONNX & TRT Export Utilities ──────────────────────────────────────────────

def export_segformer_onnx(
    onnx_path: str,
    model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    input_size: tuple = (512, 512),
):
    """
    Export SegFormer to ONNX format.

    Args:
        onnx_path:  Output ONNX file path.
        model_name: HuggingFace model ID.
        input_size: (H, W) input resolution.
    """
    from transformers import SegformerForSemanticSegmentation

    print(f"[Export] Loading {model_name}...")
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval().cuda()

    h, w = input_size
    dummy = torch.randn(1, 3, h, w, device="cuda")

    print(f"[Export] Exporting ONNX to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes=None,  # Fixed shape for best TRT performance
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[Export] ONNX saved: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    max_workspace_gb: float = 2.0,
):
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path:        Input ONNX file.
        engine_path:      Output TRT engine file.
        fp16:             Enable FP16 precision.
        max_workspace_gb: Maximum GPU workspace in GB.
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not installed")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    print(f"[TRT] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(max_workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] FP16 mode enabled")

    # Build engine
    print("[TRT] Building engine (this may take several minutes)...")
    t0 = time.time()
    serialised = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if serialised is None:
        raise RuntimeError("Engine build failed")

    # Save
    with open(engine_path, "wb") as f:
        f.write(serialised)

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"[TRT] Engine saved: {engine_path} ({size_mb:.1f} MB) in {elapsed:.1f}s")
