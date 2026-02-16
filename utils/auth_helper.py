import jwt
import uuid
import hashlib
import os
from urllib.parse import urlencode

def generate_auth_header(access_key, secret_key, query_params=None):
    """
    Upbit API Authorization Header Generator.
    Adheres to: https://docs.upbit.com/kr/reference/auth

    Args:
        access_key (str): Create at Upbit > Support > Open API
        secret_key (str): Secret key
        query_params (dict): Request query parameters (for GET requests with params)

    Returns:
        str: Authorization header string ("Bearer <jwt_token>")
    """
    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
    }

    if query_params:
        query_string = urlencode(query_params).encode()

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload['query_hash'] = query_hash
        payload['query_hash_alg'] = 'SHA512'

    # Create JWT Token
    # Note: access_key is in payload, secret_key is used for signing.
    jwt_token = jwt.encode(payload, secret_key, algorithm='HS256')
    
    # In more recent pyjwt versions, encode returns a string. 
    # In older versions, it might return bytes. Ensuring it's string.
    if isinstance(jwt_token, bytes):
        jwt_token = jwt_token.decode('utf-8')

    return f'Bearer {jwt_token}'

def validate_keys(access_key, secret_key):
    """
    Simple validation of key format.
    """
    if not access_key or not secret_key:
        return False
    if len(access_key) != 40 or len(secret_key) != 40:
        # Warning: Upbit keys are usually 40 chars, but this might change.
        pass 
    return True
