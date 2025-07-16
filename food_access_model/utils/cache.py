import threading
import time

class SimpleCache:
    """
    A simple thread-safe in-memory cache with optional time-to-live (TTL) support.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SimpleCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, default_ttl: float = 600):
        self._store = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl

    def set(self, key, value, ttl: float = None):
        """
        Set a value in the cache with an optional TTL (in seconds).
        """
        expire_at = None
        if ttl is None:
            ttl = self._default_ttl
        if ttl is not None:
            expire_at = time.time() + ttl
        with self._lock:
            self._store[key] = (value, expire_at)

    def get(self, key, default=None):
        """
        Get a value from the cache. Returns default if not found or expired.
        """
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return default
            value, expire_at = item
            if expire_at is not None and time.time() > expire_at:
                # Expired
                del self._store[key]
                return default
            return value

    def delete(self, key):
        """
        Remove a value from the cache.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]

    def clear(self):
        """
        Clear the entire cache.
        """
        with self._lock:
            self._store.clear()

    def cleanup(self):
        """
        Remove all expired items from the cache.
        """
        now = time.time()
        with self._lock:
            keys_to_delete = [k for k, (_, expire_at) in self._store.items()
                              if expire_at is not None and now > expire_at]
            for k in keys_to_delete:
                del self._store[k]

def get_cache():
    return SimpleCache()