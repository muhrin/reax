import subprocess
import sys


def probe_local_device_count(platform: str) -> int:
    """Use a subprocess to import JAX and ask it the number of local devices.  This is necessary
    if we want to avoid calling any jax functions in this process which would already set in stone
    the device setup"""
    if platform == "auto":
        platform = "None"
    else:
        platform = f"'{platform}'"

    code = [
        "import jax",
        f"jax.config.update('jax_platforms', {platform})",
        "print(jax.local_device_count())",
    ]

    code = ";".join(code)
    result = subprocess.check_output([sys.executable, "-c", code])  # nosec
    return int(result.decode().strip())
