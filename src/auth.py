"""Persistent authentication and role authorization for the web application."""
from __future__ import annotations
import hashlib, hmac, os, secrets, sqlite3, time
from dataclasses import dataclass
from pathlib import Path
ROLES = ("master", "manager", "general")
_BOOTSTRAP_HASH = "df3eb02c840d41769052207545b3693920ba014461195b889427669f69443eae"
@dataclass(frozen=True)
class User: username: str; role: str
class AuthService:
    def __init__(self, db_path="vector_database/auth.sqlite3", secret=None):
        self.db_path=db_path; Path(db_path).parent.mkdir(parents=True, exist_ok=True); self.secret=(secret or os.getenv("AUTH_TOKEN_SECRET") or "change-me-in-production").encode()
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('master','manager','general')), created_at REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, username TEXT NOT NULL, expires_at REAL NOT NULL)")
            if not db.execute("SELECT 1 FROM users WHERE role='master'").fetchone(): self._insert(db,"master",os.getenv("MASTER_PASSWORD_HASH",_BOOTSTRAP_HASH),"master",True)
    def _connect(self): return sqlite3.connect(self.db_path)
    @staticmethod
    def hash_password(password, salt=None):
        salt=salt or secrets.token_bytes(16); return f"pbkdf2_sha256$240000${salt.hex()}${hashlib.pbkdf2_hmac('sha256',password.encode(),salt,240000).hex()}"
    @staticmethod
    def verify_password(password, stored):
        if stored.startswith("pbkdf2_sha256$"):
            _, rounds, salt, digest=stored.split("$"); actual=hashlib.pbkdf2_hmac("sha256",password.encode(),bytes.fromhex(salt),int(rounds)).hex(); return hmac.compare_digest(actual,digest)
        return hmac.compare_digest(stored,hashlib.pbkdf2_hmac("sha256",password.encode(),b"rag-bootstrap",240000).hex())
    def _insert(self,db,username,password,role,hashed=False): db.execute("INSERT INTO users VALUES (?,?,?,?)",(username,password if hashed else self.hash_password(password),role,time.time()))
    def authenticate(self,username,password):
        with self._connect() as db: row=db.execute("SELECT password_hash FROM users WHERE username=?",(username.strip(),)).fetchone()
        if not row or not self.verify_password(password,row[0]): return None
        token=secrets.token_urlsafe(32)
        with self._connect() as db: db.execute("INSERT INTO sessions VALUES (?,?,?)",(token,username.strip(),time.time()+8*3600))
        return token
    def user_for_token(self,token):
        if not token:return None
        with self._connect() as db: row=db.execute("SELECT u.username,u.role FROM sessions s JOIN users u ON u.username=s.username WHERE s.token=? AND s.expires_at>?",(token,time.time())).fetchone()
        return User(*row) if row else None
    def create_user(self,actor,username,password,role):
        if role not in ROLES[1:]: raise ValueError("Invalid role")
        if role=="manager" and actor.role!="master": raise PermissionError("Only the master can create managers")
        if role=="general" and actor.role not in ("master","manager"): raise PermissionError("Not authorized")
        if not username.strip() or len(password)<8: raise ValueError("Username and a password of at least 8 characters are required")
        with self._connect() as db:
            try:self._insert(db,username.strip(),password,role)
            except sqlite3.IntegrityError:raise ValueError("Username already exists")
        return f"Created {role} account: {username.strip()}"
    def can(self,user,capability): return bool(user and (user.role=="master" or capability in {"chat","metadata","vector","create_general"} and user.role=="manager" or capability=="chat" and user.role=="general"))
