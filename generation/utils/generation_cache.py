import hashlib,json,sqlite3,time
from pathlib import Path
from utils.io import digest

def image_identity(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return {'path':str(Path(path).resolve()),'sha256':h.hexdigest()}

class GenerationCache:
    """Atomic SQLite records; failures are audited but never served as labels."""
    def __init__(self,path,enabled=True):
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); self.enabled=enabled
        self.db=sqlite3.connect(p); self.db.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, value TEXT, status TEXT, updated REAL)'); self.db.commit()
    def get(self,key):
        if not self.enabled: return None
        row=self.db.execute("SELECT value FROM records WHERE key=? AND status='success'",(key,)).fetchone()
        return json.loads(row[0]) if row else None
    def put(self,key,value,status='success'):
        if not self.enabled: return
        with self.db: self.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',(key,json.dumps(value,ensure_ascii=False),status,time.time()))
    def close(self): self.db.close()
