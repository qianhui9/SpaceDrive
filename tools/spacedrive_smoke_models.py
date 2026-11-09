import torch
from transformers import Qwen2_5_VLConfig, LlavaConfig
from projects.mmdet3d_plugin.models.vlm_utils.custom_qwen import CustomQwen2_5_VLForConditionalGeneration
from projects.mmdet3d_plugin.models.vlm_utils.custom_llava import CustomLlavaForConditionalGeneration


def run(device='cpu'):
    text = dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    qwen_config = Qwen2_5_VLConfig(
        **dict(text, rope_scaling=dict(rope_type='default', mrope_section=[1, 1, 2])),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=32, patch_size=2, spatial_merge_size=2,
                           temporal_patch_size=1, fullatt_block_indexes=[0], window_size=8),
        image_token_id=125, video_token_id=126, vision_start_token_id=124, vision_end_token_id=127,
    )
    qwen_config._attn_implementation = 'eager'
    llava_config = LlavaConfig(
        text_config=dict(text, model_type='llama'),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=4, image_size=8, patch_size=4),
        image_token_index=127, vision_feature_layer=-1, image_seq_length=4,
    )
    llava_config._attn_implementation = 'eager'
    cases = [
        ('Qwen', CustomQwen2_5_VLForConditionalGeneration, qwen_config,
         torch.tensor([[1, 124, 125, 127, 4, 2]], device=device),
         dict(pixel_values=torch.rand(1, 4, 12, device=device),
              image_grid_thw=torch.tensor([[1, 2, 2]], device=device))),
        ('LLaVA', CustomLlavaForConditionalGeneration, llava_config,
         torch.tensor([[1, 127, 127, 127, 127, 4, 2]], device=device),
         dict(pixel_values=torch.rand(1, 1, 3, 8, 8, device=device))),
    ]
    for name, cls, config, ids, vision in cases:
        model = cls(config).to(device=device, dtype=torch.bfloat16).train()
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       labels=ids.clone(), has_gt_planning=torch.zeros(1, dtype=torch.bool, device=device),
                       use_cache=False, return_dict=True, **vision)
        assert output.logits.shape == (*ids.shape, 128), output.logits.shape
        assert torch.isfinite(output.loss), output.loss
        output.loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        print(f'PASS: tiny {name} multimodal forward/backward ({device}), loss={output.loss.item():.4f}', flush=True)
        del model, output, grads


if __name__ == '__main__':
    run()
