import torch
from torchinfo import summary

# Import 你的模型
from torchcfm.models.unet.unet import UNetModel_Decompose_Wrapper

def check_model_architecture():
    # 1. 設定參數 (與訓練時相同)
    image_size = 32
    num_channel = 128
    batch_size = 2  # 隨便設一個小 batch
    
    # 2. 實例化模型
    # 這裡的參數對應你訓練腳本中的設定
    model = UNetModel_Decompose_Wrapper(
        dim=(3, image_size, image_size),
        num_res_blocks=2,
        num_channels=num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    )

    # 3. 準備 Dummy Input
    # Flow Matching 的輸入通常是 (t, x)
    # t: (Batch_Size,)
    # x: (Batch_Size, 3, H, W)
    t = torch.rand(batch_size)
    x = torch.randn(batch_size, 3, image_size, image_size)

    print(f"Model Class: {model.__class__.__name__}")
    print("-" * 50)

    # 4. 使用 torchinfo 顯示詳細資訊
    # 注意：我們需要傳入 input_data 而不是 input_size，因為 forward 有多個參數
    summary(
        model, 
        input_data=[t, x], 
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=4, # 深度設為 4 可以看到 BottleneckEnergyHead 內部的細節
        row_settings=["var_names"]
    )

    print("\n" + "="*50)
    print("Sanity Check for Outputs:")
    
    # 5. 實際跑一次 Forward 檢查輸出形狀
    model.eval()
    with torch.no_grad():
        v_pred, energy_pred, shape_pred = model(t, x)
    
    print(f"Input x shape:       {x.shape}")
    print(f"Output v shape:      {v_pred.shape} (Should be B, 3, H, W)")
    print(f"Output Energy shape: {energy_pred.shape} (Should be B, 3)")
    print(f"Output Shape map:    {shape_pred.shape}  (Should be B, 3, H, W)")
    
    # 驗證 Normalization 是否正確
    # 檢查 shape_pred 的 L2 Norm 在 (H, W) 維度上是否約為 1
    norms = torch.norm(shape_pred, p=2, dim=(2, 3))
    print(f"\nVerifying Spatial Normalization:")
    print(f"Shape map norms (first 5 samples): {norms[0].tolist()}") 
    
    is_norm_correct = torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    if is_norm_correct:
        print("✅ PASS: Shape map is correctly normalized to 1.")
    else:
        print("❌ FAIL: Shape map is NOT normalized (Check dim parameters in F.normalize).")

    # 驗證 Energy 是否為非負 (Softplus)
    if (energy_pred >= 0).all():
         print("✅ PASS: Energy scalars are non-negative.")
    else:
         print("❌ FAIL: Energy scalars contain negative values.")

if __name__ == "__main__":
    check_model_architecture()