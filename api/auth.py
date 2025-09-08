"""
JWT Authentication and API Key Management System
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
import redis

from .cache_manager import AdvancedCacheManager


class JWTAuth:
    """JWT token authentication system"""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", self._generate_secret_key())
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # In-memory user store (in production, use proper database)
        self.users_db = {
            "admin": {
                "username": "admin",
                "hashed_password": self.get_password_hash("admin123"),
                "email": "admin@example.com",
                "role": "admin",
                "permissions": ["read", "write", "admin"],
                "is_active": True
            },
            "analyst": {
                "username": "analyst", 
                "hashed_password": self.get_password_hash("analyst123"),
                "email": "analyst@example.com",
                "role": "analyst",
                "permissions": ["read"],
                "is_active": True
            }
        }
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(32)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username and password"""
        user = self.users_db.get(username)
        if not user:
            return None
        if not self.verify_password(password, user["hashed_password"]):
            return None
        if not user.get("is_active", True):
            return None
        return user
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                return None
            
            # Check if user still exists and is active
            user = self.users_db.get(username)
            if not user or not user.get("is_active", True):
                return None
            
            return {
                "username": username,
                "email": user.get("email"),
                "role": user.get("role"),
                "permissions": user.get("permissions", []),
                "exp": payload.get("exp"),
                "type": payload.get("type", "access")
            }
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token"""
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.get("type") != "refresh":
            return None
        
        # Create new access token
        new_token_data = {"sub": payload["username"]}
        return self.create_access_token(new_token_data)
    
    def create_user(self, username: str, password: str, email: str, role: str = "user") -> bool:
        """Create a new user"""
        if username in self.users_db:
            return False
        
        permissions = {
            "admin": ["read", "write", "admin"],
            "analyst": ["read"],
            "user": ["read"]
        }.get(role, ["read"])
        
        self.users_db[username] = {
            "username": username,
            "hashed_password": self.get_password_hash(password),
            "email": email,
            "role": role,
            "permissions": permissions,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat()
        }
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate user account"""
        if username in self.users_db:
            self.users_db[username]["is_active"] = False
            return True
        return False


class APIKeyAuth:
    """API Key authentication and management system"""
    
    def __init__(self, cache_manager: Optional[AdvancedCacheManager] = None):
        self.cache_manager = cache_manager
        
        # In-memory API key store (in production, use proper database)
        self.api_keys_db = {
            "ak_live_123456789": {
                "key_id": "ak_live_123456789",
                "name": "Production API Key",
                "user_id": "admin",
                "permissions": ["read", "write"],
                "rate_limit": 1000,  # requests per hour
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "last_used": None,
                "usage_count": 0
            },
            "ak_test_987654321": {
                "key_id": "ak_test_987654321",
                "name": "Development API Key",
                "user_id": "analyst",
                "permissions": ["read"],
                "rate_limit": 100,
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "last_used": None,
                "usage_count": 0
            }
        }
    
    def generate_api_key(self, prefix: str = "ak_live") -> str:
        """Generate a new API key"""
        random_part = secrets.token_urlsafe(16)
        return f"{prefix}_{random_part}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def create_api_key(self, 
                           user_id: str, 
                           name: str,
                           permissions: List[str] = ["read"],
                           rate_limit: int = 100,
                           key_type: str = "live") -> Dict[str, Any]:
        """Create a new API key"""
        api_key = self.generate_api_key(f"ak_{key_type}")
        
        key_data = {
            "key_id": api_key,
            "name": name,
            "user_id": user_id,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "usage_count": 0,
            "key_type": key_type
        }
        
        # Store in database
        self.api_keys_db[api_key] = key_data
        
        # Cache in Redis for fast lookup
        if self.cache_manager:
            await self.cache_manager.set(
                f"api_key:{api_key}", 
                key_data, 
                ttl=86400  # 24 hours
            )
        
        return {
            "api_key": api_key,
            "key_id": api_key,
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "created_at": key_data["created_at"]
        }
    
    async def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return user information"""
        # Try cache first
        if self.cache_manager:
            cached_key = await self.cache_manager.get(f"api_key:{api_key}")
            if cached_key and cached_key.get("is_active"):
                # Update usage statistics
                await self._update_key_usage(api_key, cached_key)
                return cached_key
        
        # Fall back to database
        key_data = self.api_keys_db.get(api_key)
        
        if not key_data or not key_data.get("is_active"):
            return None
        
        # Update usage statistics
        await self._update_key_usage(api_key, key_data)
        
        # Cache for future requests
        if self.cache_manager:
            await self.cache_manager.set(
                f"api_key:{api_key}", 
                key_data, 
                ttl=86400
            )
        
        return key_data
    
    async def _update_key_usage(self, api_key: str, key_data: Dict[str, Any]):
        """Update API key usage statistics"""
        key_data["last_used"] = datetime.utcnow().isoformat()
        key_data["usage_count"] = key_data.get("usage_count", 0) + 1
        
        # Update in database
        if api_key in self.api_keys_db:
            self.api_keys_db[api_key].update(key_data)
        
        # Update in cache
        if self.cache_manager:
            await self.cache_manager.set(f"api_key:{api_key}", key_data, ttl=86400)
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke (deactivate) an API key"""
        if api_key in self.api_keys_db:
            self.api_keys_db[api_key]["is_active"] = False
            self.api_keys_db[api_key]["revoked_at"] = datetime.utcnow().isoformat()
            
            # Remove from cache
            if self.cache_manager:
                await self.cache_manager.delete(f"api_key:{api_key}")
            
            return True
        return False
    
    async def list_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """List all API keys for a user"""
        user_keys = []
        
        for key_id, key_data in self.api_keys_db.items():
            if key_data["user_id"] == user_id:
                # Don't return the actual key, only metadata
                user_keys.append({
                    "key_id": key_id[:20] + "...",  # Masked key
                    "name": key_data["name"],
                    "permissions": key_data["permissions"],
                    "rate_limit": key_data["rate_limit"],
                    "is_active": key_data["is_active"],
                    "created_at": key_data["created_at"],
                    "last_used": key_data.get("last_used"),
                    "usage_count": key_data.get("usage_count", 0)
                })
        
        return user_keys
    
    async def get_key_usage_stats(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for an API key"""
        key_data = self.api_keys_db.get(api_key)
        
        if not key_data:
            return None
        
        return {
            "key_id": api_key[:20] + "...",
            "name": key_data["name"],
            "usage_count": key_data.get("usage_count", 0),
            "last_used": key_data.get("last_used"),
            "created_at": key_data["created_at"],
            "is_active": key_data["is_active"],
            "rate_limit": key_data["rate_limit"]
        }


class RolePermissionManager:
    """Role-based access control manager"""
    
    def __init__(self):
        self.permissions = {
            "admin": ["read", "write", "delete", "admin", "manage_users", "manage_models"],
            "analyst": ["read", "write"],
            "viewer": ["read"],
            "api_user": ["read", "api_access"]
        }
    
    def has_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions
    
    def has_any_permission(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """Check if user has any of the required permissions"""
        return any(perm in user_permissions for perm in required_permissions)
    
    def has_all_permissions(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """Check if user has all required permissions"""
        return all(perm in user_permissions for perm in required_permissions)


# Permission checking decorator
def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user from dependency injection
            user = kwargs.get('user') or (args[0] if args else None)
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user_permissions = user.get("permissions", [])
            permission_manager = RolePermissionManager()
            
            if not permission_manager.has_any_permission(user_permissions, required_permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {required_permissions}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator