"""
Tests for authentication system
"""

import pytest
from datetime import datetime, timedelta

from api.auth import JWTAuth, APIKeyAuth, RolePermissionManager


class TestJWTAuth:
    """Test JWT authentication"""
    
    def test_password_hashing(self, jwt_auth):
        """Test password hashing and verification"""
        password = "test_password_123"
        hashed = jwt_auth.get_password_hash(password)
        
        assert hashed != password
        assert jwt_auth.verify_password(password, hashed)
        assert not jwt_auth.verify_password("wrong_password", hashed)
    
    def test_user_authentication(self, jwt_auth):
        """Test user authentication"""
        # Valid user
        user = jwt_auth.authenticate_user("admin", "admin123")
        assert user is not None
        assert user["username"] == "admin"
        assert user["role"] == "admin"
        
        # Invalid password
        user = jwt_auth.authenticate_user("admin", "wrong_password")
        assert user is None
        
        # Non-existent user
        user = jwt_auth.authenticate_user("nonexistent", "password")
        assert user is None
    
    def test_token_creation_and_verification(self, jwt_auth):
        """Test JWT token creation and verification"""
        # Create token
        token_data = {"sub": "admin"}
        token = jwt_auth.create_access_token(token_data)
        
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are quite long
        
        # Verify token
        payload = jwt_auth.verify_token(token)
        assert payload is not None
        assert payload["username"] == "admin"
        assert payload["role"] == "admin"
        assert "exp" in payload
    
    def test_token_expiration(self, jwt_auth):
        """Test token expiration"""
        # Create token with short expiration
        token_data = {"sub": "admin"}
        expires_delta = timedelta(seconds=-1)  # Already expired
        token = jwt_auth.create_access_token(token_data, expires_delta)
        
        # Verify expired token
        payload = jwt_auth.verify_token(token)
        assert payload is None
    
    def test_refresh_token(self, jwt_auth):
        """Test refresh token functionality"""
        token_data = {"sub": "admin"}
        refresh_token = jwt_auth.create_refresh_token(token_data)
        
        # Use refresh token to get new access token
        new_access_token = jwt_auth.refresh_access_token(refresh_token)
        assert new_access_token is not None
        
        # Verify new access token
        payload = jwt_auth.verify_token(new_access_token)
        assert payload["username"] == "admin"
    
    def test_create_user(self, jwt_auth):
        """Test user creation"""
        # Create new user
        success = jwt_auth.create_user("testuser", "password123", "test@example.com", "user")
        assert success is True
        
        # Verify user was created
        user = jwt_auth.authenticate_user("testuser", "password123")
        assert user is not None
        assert user["username"] == "testuser"
        assert user["email"] == "test@example.com"
        assert user["role"] == "user"
        
        # Try to create duplicate user
        success = jwt_auth.create_user("testuser", "password123", "test@example.com")
        assert success is False
    
    def test_deactivate_user(self, jwt_auth):
        """Test user deactivation"""
        # Create and authenticate user
        jwt_auth.create_user("tempuser", "password123", "temp@example.com")
        user = jwt_auth.authenticate_user("tempuser", "password123")
        assert user is not None
        
        # Deactivate user
        success = jwt_auth.deactivate_user("tempuser")
        assert success is True
        
        # Try to authenticate deactivated user
        user = jwt_auth.authenticate_user("tempuser", "password123")
        assert user is None


class TestAPIKeyAuth:
    """Test API key authentication"""
    
    @pytest.mark.asyncio
    async def test_api_key_generation(self, api_key_auth):
        """Test API key generation"""
        key1 = api_key_auth.generate_api_key("ak_test")
        key2 = api_key_auth.generate_api_key("ak_live")
        
        assert key1.startswith("ak_test_")
        assert key2.startswith("ak_live_")
        assert key1 != key2
        assert len(key1) > 20
        assert len(key2) > 20
    
    @pytest.mark.asyncio
    async def test_create_api_key(self, api_key_auth):
        """Test API key creation"""
        key_data = await api_key_auth.create_api_key(
            user_id="testuser",
            name="Test Key",
            permissions=["read", "write"],
            rate_limit=200
        )
        
        assert "api_key" in key_data
        assert key_data["name"] == "Test Key"
        assert key_data["permissions"] == ["read", "write"]
        assert key_data["rate_limit"] == 200
        assert "created_at" in key_data
    
    @pytest.mark.asyncio
    async def test_verify_api_key(self, api_key_auth):
        """Test API key verification"""
        # Create API key
        key_data = await api_key_auth.create_api_key(
            user_id="testuser",
            name="Test Key",
            permissions=["read"]
        )
        
        api_key = key_data["api_key"]
        
        # Verify valid key
        verified_data = await api_key_auth.verify_api_key(api_key)
        assert verified_data is not None
        assert verified_data["user_id"] == "testuser"
        assert verified_data["permissions"] == ["read"]
        assert verified_data["is_active"] is True
        assert verified_data["usage_count"] == 1  # First use
        
        # Verify invalid key
        invalid_data = await api_key_auth.verify_api_key("invalid_key")
        assert invalid_data is None
    
    @pytest.mark.asyncio
    async def test_revoke_api_key(self, api_key_auth):
        """Test API key revocation"""
        # Create and verify key
        key_data = await api_key_auth.create_api_key(user_id="testuser", name="Test Key")
        api_key = key_data["api_key"]
        
        verified = await api_key_auth.verify_api_key(api_key)
        assert verified is not None
        
        # Revoke key
        success = await api_key_auth.revoke_api_key(api_key)
        assert success is True
        
        # Try to verify revoked key
        verified = await api_key_auth.verify_api_key(api_key)
        assert verified is None
    
    @pytest.mark.asyncio
    async def test_list_user_api_keys(self, api_key_auth):
        """Test listing user API keys"""
        user_id = "testuser"
        
        # Create multiple keys
        await api_key_auth.create_api_key(user_id, "Key 1", ["read"])
        await api_key_auth.create_api_key(user_id, "Key 2", ["write"])
        
        # List keys
        keys = await api_key_auth.list_api_keys(user_id)
        assert len(keys) == 2
        
        # Check key data (should be masked)
        for key in keys:
            assert "key_id" in key
            assert key["key_id"].endswith("...")  # Masked
            assert "name" in key
            assert "permissions" in key
            assert "usage_count" in key
    
    @pytest.mark.asyncio
    async def test_api_key_usage_tracking(self, api_key_auth):
        """Test API key usage tracking"""
        # Create key
        key_data = await api_key_auth.create_api_key(user_id="testuser", name="Usage Test")
        api_key = key_data["api_key"]
        
        # Use key multiple times
        for i in range(3):
            verified = await api_key_auth.verify_api_key(api_key)
            assert verified["usage_count"] == i + 1
            assert verified["last_used"] is not None
        
        # Get usage stats
        stats = await api_key_auth.get_key_usage_stats(api_key)
        assert stats["usage_count"] == 3
        assert stats["last_used"] is not None


class TestRolePermissionManager:
    """Test role-based access control"""
    
    def test_has_permission(self):
        """Test permission checking"""
        rpm = RolePermissionManager()
        
        user_permissions = ["read", "write"]
        
        assert rpm.has_permission(user_permissions, "read")
        assert rpm.has_permission(user_permissions, "write")
        assert not rpm.has_permission(user_permissions, "admin")
        assert not rpm.has_permission(user_permissions, "delete")
    
    def test_has_any_permission(self):
        """Test any permission checking"""
        rpm = RolePermissionManager()
        
        user_permissions = ["read", "write"]
        required_permissions = ["admin", "write"]
        
        assert rpm.has_any_permission(user_permissions, required_permissions)
        
        required_permissions = ["admin", "delete"]
        assert not rpm.has_any_permission(user_permissions, required_permissions)
    
    def test_has_all_permissions(self):
        """Test all permissions checking"""
        rpm = RolePermissionManager()
        
        user_permissions = ["read", "write", "admin"]
        required_permissions = ["read", "write"]
        
        assert rpm.has_all_permissions(user_permissions, required_permissions)
        
        required_permissions = ["read", "delete"]
        assert not rpm.has_all_permissions(user_permissions, required_permissions)