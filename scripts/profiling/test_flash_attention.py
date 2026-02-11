"""
Script to validate if Flash Attention is being used in the transformer encoder.
Run this to check your setup.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# Test if flash attention backend is available
def test_flash_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small test tensors
    batch_size, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    d_model = num_heads * head_dim

    # Create dummy inputs (fp16 for flash attention)
    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)

    # Test with MultiheadAttention
    mha = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True).to(device).half()

    print("\n" + "="*60)
    print("Testing nn.MultiheadAttention backend selection:")
    print("="*60)

    # Test 1: No mask (should use flash attention)
    print("\n1. No mask:")
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        try:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out, _ = mha(q, k, v)
            print("   ✅ Flash Attention ENABLED")
        except RuntimeError as e:
            print(f"   ❌ Flash Attention FAILED: {e}")

    # Test 2: With key_padding_mask (might fall back)
    print("\n2. With key_padding_mask:")
    key_padding_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    key_padding_mask[:, seq_len//2:] = True  # mask second half

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        try:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out, _ = mha(q, k, v, key_padding_mask=key_padding_mask)
            print("   ✅ Flash Attention ENABLED with padding mask")
        except RuntimeError as e:
            print(f"   ⚠️  Flash Attention FAILED with padding mask: {e}")
            print("   → Will fall back to standard attention")

    # Test 3: Check which backends are available
    print("\n3. Available SDPA backends:")
    backends = []
    for backend in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]:
        with sdpa_kernel([backend]):
            try:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = F.scaled_dot_product_attention(
                        q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2),
                        k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2),
                        v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    )
                backends.append(backend.name)
                print(f"   ✅ {backend.name}")
            except:
                print(f"   ❌ {backend.name}")

    return backends

if __name__ == "__main__":
    available_backends = test_flash_attention()

    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    if "FLASH_ATTENTION" in available_backends:
        print("✅ Flash Attention is available on your system!")
        print("\nTo ensure it's used in training:")
        print("  1. Use fp16 or bf16 mixed precision (Detectron2 does this)")
        print("  2. Avoid complex attention masks if possible")
        print("  3. Set dropout=0 during inference")
    else:
        print("❌ Flash Attention is NOT available.")
        print("   Check: PyTorch >= 2.0, CUDA-compatible GPU, proper drivers")
