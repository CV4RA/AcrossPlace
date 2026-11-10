from utils.generation_cache import GenerationCache,image_identity

def test_atomic_resume_and_content_hash(tmp_path):
    p=tmp_path/'image'; p.write_bytes(b'one'); identity=image_identity(p); p.write_bytes(b'two'); assert image_identity(p)!=identity
    cache=GenerationCache(tmp_path/'cache.sqlite'); cache.put('x',{'facts':[1]}); cache.put('bad',{'error':'failed'},status='failed'); cache.close()
    cache=GenerationCache(tmp_path/'cache.sqlite'); assert cache.get('x')=={'facts':[1]}; assert cache.get('bad') is None; cache.close()
