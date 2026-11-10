import torch,pytest
from utils.vlm_loader import resolve_model_path,clean_json

def test_offline_resolution_never_requests_network(tmp_path,monkeypatch):
    import huggingface_hub
    calls=[]
    monkeypatch.setattr(huggingface_hub,'try_to_load_from_cache',lambda *a,**k: None)
    def download(*args,**kwargs):
        calls.append(kwargs); raise FileNotFoundError('no snapshot')
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    with pytest.raises(FileNotFoundError,match='No usable local snapshot'):
        resolve_model_path({'name':'Qwen/Qwen3-VL-4B-Instruct','cache_dir':str(tmp_path),'local_files_only':True})
    assert calls and all(x['local_files_only'] for x in calls)

def test_random_tiny_official_qwen3_cpu_forward():
    # No pretrained 4B weights or performance claims; verifies installed architecture.
    transformers=pytest.importorskip('transformers')
    if not hasattr(transformers,'Qwen3VLForConditionalGeneration'): pytest.skip('Use .venv-qwen3 for Qwen3 interfaces')
    from transformers import Qwen3VLForConditionalGeneration,Qwen3VLConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig,Qwen3VLVisionConfig
    text=Qwen3VLTextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,rope_scaling={'rope_type':'default','mrope_section':[1,1,2]},eos_token_id=2,pad_token_id=0)
    vision=Qwen3VLVisionConfig(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,num_position_embeddings=16,deepstack_visual_indexes=[0])
    model=Qwen3VLForConditionalGeneration(Qwen3VLConfig(text_config=text.to_dict(),vision_config=vision.to_dict(),image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)).eval()
    with torch.inference_mode():
        output=model.generate(input_ids=torch.tensor([[1,62,60,63,4,5]]),pixel_values=torch.randn(4,1536),image_grid_thw=torch.tensor([[1,2,2]]),max_new_tokens=2,do_sample=False)
    assert output.shape[0]==1 and output.shape[1]>6
