import os
import env
import shutil
import platform
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MllmEnv:
    # System info
    system_type: str
    machine_arch: str
    cpu_count: int

    # Mllm related stuff
    mllm_root_path: Path
    mllm_cache_path: Path
    mllm_rt_max_threads: int
    mllm_cpu_memory_pool_default_size: int
    mllm_gpu_memory_pool_default_size: int

    # CUDA related stuff
    _mllm_cuda_enabled: bool = False
    cuda_toolkit_path: Path
    nvcc_path: Path

    # QNN related stuff
    _mllm_qnn_enabled: bool = False

    # DSL backend
    tilelang_enabled: bool = True
    triton_enabled: bool = False

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MllmEnv, cls).__new__(cls)
            cls._mllm_basic_env_init()

            # Find devices or dirs
            cls._instance._find_or_create_mllm_cache_path()
            cls._instance._find_cuda()
            cls._instance._find_qnn()
        return cls._instance

    def _system_env_init(self):
        self.cpu_count = os.cpu_count()
        self.system_type = platform.system().lower()
        self.machine_arch = platform.machine()

    def _mllm_basic_env_init(self):
        self.mllm_rt_max_threads = os.cpu_count() // 2
        self.mllm_cpu_memory_pool_default_size = 1024 * 1024 * 256  # 256MB
        self.mllm_gpu_memory_pool_default_size = 1024 * 1024 * 256  # 256MB

    def _find_cuda(self):
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            self.nvcc_path = Path(nvcc_path).resolve()
            self.cuda_toolkit_path = self.nvcc_path.parent.parent
            self._mllm_cuda_enabled = True

    def _find_qnn(self):
        self._mllm_qnn_enabled = (
            "QNN_ENABLED" in os.environ and os.environ["QNN_ENABLED"].lower() == "true"
        )

    def _find_or_create_mllm_cache_path(self):
        self.mllm_root_path = Path.home() / ".mllm"
        if not self.mllm_root_path.exists():
            self.mllm_root_path.mkdir(parents=False, exist_ok=True)
        self.mllm_cache_path = self.mllm_root_path / "cache"
        if not self.mllm_cache_path.exists():
            self.mllm_cache_path.mkdir(parents=False, exist_ok=True)

    @property
    def mllm_qnn_enabled(self):
        return self._mllm_qnn_enabled

    @property
    def mllm_cuda_enabled(self):
        return self._mllm_cuda_enabled
