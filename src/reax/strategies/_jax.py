import logging
import os
import random
import socket
import subprocess  # nosec  # Suppresses Bandit's B404 subprocess warning
import sys
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax
from jax import distributed, tree
from jax.experimental import multihost_utils
import jax.numpy as jnp
import jaxtyping as jt
from typing_extensions import override

from . import _parallel

if TYPE_CHECKING:
    import reax


__all__ = ("JaxDdpStrategy",)

_LOGGER = logging.getLogger(__name__)


_OutT = TypeVar("_OutT")
Children = list[subprocess.Popen]


class JaxDdpStrategy(_parallel.ParallelStrategy):
    """This strategy uses multi-processing and the JAX library for communication."""

    def __init__(self, platform: str = None, devices: int | str = "auto"):
        _LOGGER.info("Starting JAX DDP strategy...")
        res = self._init(platform, devices)
        self._process_id = res[0]
        self._num_processes: int = res[1]
        self._children = res[2]
        self._device = jax.local_devices()[0]

    @override
    def teardown(self):
        # Wait for all children to finish
        if self._process_id is not None:
            jax.distributed.shutdown()
            for child in self._children:
                child.wait()
            self._children = None
            self._process_id = None

    @staticmethod
    def probe_local_device_count() -> int:
        """Use a subprocess to import JAX and ask it the number of local devices"""
        code = "import jax; print(jax.local_device_count())"
        result = subprocess.check_output([sys.executable, "-c", code])  # nosec # Disable Bandit
        return int(result.decode().strip())

    @staticmethod
    def get_available_port(min_port=49152, max_port=65535, max_attempts=100) -> int:
        """Generates a random port and checks if it's available.  Retries if necessary."""
        for _ in range(max_attempts):
            port = random.randint(min_port, max_port)  # nosec
            try:
                # Attempt to bind to the port.  If it's in use, this will raise an exception.
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("localhost", port))  # Bind to localhost for testing
                s.close()
                return port
            except OSError:
                # Port is already in use.  Try again.
                pass  # Continue to the next iteration of the loop

        raise OSError(f"Could not find an available port after {max_attempts} attempts.")

    def _init(self, platform: str | None, devices: int | str) -> tuple[int, int, Children]:
        if platform:
            jax.config.update("jax_platforms", platform)

        children = []
        if "REAX_PROCESS_ID" in os.environ:
            env = os.environ
            kwargs = dict(
                coordinator_address=env["REAX_COORDINATOR_ADDRESS"],
                num_processes=int(env["REAX_NUM_PROCESSES"]),
                process_id=int(env["REAX_PROCESS_ID"]),
                local_device_ids=int(env["REAX_PROCESS_ID"]),
            )
            res = self._distributed_init(**kwargs)
            return *res, children

        try:
            # First, let's try the automated method which uses automated cluster detection
            res = self._distributed_init(num_processes=None if devices == "auto" else devices)
        except ValueError:
            # Not in a cluster environment, so open our own processes
            port = self.get_available_port()
            coordinator_address = f"localhost:{port}"
            if devices == "auto":
                num_processes = self.probe_local_device_count()
            else:
                num_processes = devices

            # Launch the other processes
            children = self._launch_children(platform, num_processes, coordinator_address)

            # and now join them
            res = self._distributed_init(
                coordinator_address=coordinator_address,
                num_processes=num_processes,
                process_id=0,
                local_device_ids=0,
            )

        return *res, children

    def _distributed_init(self, **kwargs) -> tuple[int, int]:
        distributed.initialize(**kwargs)
        return jax.process_index(), jax.process_count()

    def _launch_children(
        self, platform: str | None, num: int, coordinator_address: str
    ) -> Children:
        children = []
        for process_id in range(1, num):
            kwargs = {
                "REAX_COORDINATOR_ADDRESS": coordinator_address,
                "REAX_NUM_PROCESSES": str(num),
                "REAX_PROCESS_ID": str(process_id),
            }
            if platform is not None:
                kwargs["JAX_PLATFORMS"] = platform

            # Copy the current environment
            env = os.environ.copy()
            # Add or override variables
            env.update(kwargs)

            # Relaunch the same script with the same Python executable
            if "REAX_COMMAND" in os.environ:
                cmd = os.environ["REAX_COMMAND"].split()
            else:
                cmd = [sys.executable] + sys.argv
            _LOGGER.info("Launching subprocess with: %s", " ".join(cmd))
            children.append(
                subprocess.Popen(  # pylint: disable=consider-using-with
                    cmd, env=env
                )  # nosec # Disable bandit
            )

        return children

    @property
    def device(self):
        return self._device

    @property
    def process_count(self) -> int:
        """Return the total number of ranks."""
        return self._num_processes

    @property
    def process_index(self) -> int:
        return self._process_id

    @override
    def to_device(self, value: Any) -> Any:
        """To device."""
        return jax.device_put(value, self._device)

    @override
    def from_device(self, value: Any) -> Any:
        """From device."""
        return jax.device_get(value)

    @property
    @override
    def is_global_zero(self) -> bool:
        """Is global zero."""
        return self._process_id == 0

    @override
    def broadcast(self, obj: jt.PyTreeDef, src: int = 0) -> Any:
        """Broadcasts an object to all processes.

        Args:
            obj: The pytree to broadcast.
            src (int, optional): Source rank, defaults to 0.
        """
        is_str = isinstance(obj, str)
        if is_str:
            # Encode
            with jax.default_device(self._device):
                obj = jnp.array(list(obj.encode("utf-8")), dtype=jnp.uint8)

        res = multihost_utils.broadcast_one_to_all(obj, src == self.process_index)
        if is_str:
            # Decode
            res = bytes(res.tolist()).decode("utf-8")

        return res

    @override
    def all_gather(self, obj: jt.PyTreeDef) -> Any:
        return multihost_utils.process_allgather(obj)

    @override
    def all_reduce(self, obj: jt.PyTree, reduce_op: str = "mean") -> jt.PyTree:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            obj: the pytree to sync and reduce
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value
        """
        return getattr(jnp, reduce_op)(self.all_gather(obj))

    @override
    def barrier(self, name: str | None = None) -> None:
        """Synchronizes all processes which blocks processes until the whole group enters this
        function.

        Args:
            name: an optional name to pass into barrier.
        """
        multihost_utils.sync_global_devices(name)

    @override
    def compute(self, metric: "reax.typing.MetricInstance[_OutT]") -> _OutT:
        if self.process_count == 1:
            return metric.compute()

        dynamic, static = eqx.partition(metric, eqx.is_array)
        gathered = multihost_utils.process_allgather(dynamic)
        unbatched: "list[reax.typing.MetricInstance[_OutT]]" = unbatch_pytree(gathered, dynamic)
        # Rejoin with the static data
        unbatched = [eqx.combine(entry, static) for entry in unbatched]

        metric = unbatched[0]
        for entry in unbatched[1:]:
            metric = metric.merge(entry)

        return metric.compute()


def unbatch_pytree(batched: jt.PyTree, original: jt.PyTree) -> list[jt.PyTree]:
    """Splits a single batched Pytree into a list of unbatched Pytrees,
    using the 'original_tree' structure to ensure correct unbatched leaf shapes.

    Args:
        batched: A Pytree where every leaf array has shape (B, ...).
        original: A Pytree with the *expected* leaf shapes after unbatching.

    Returns:
        A list of Pytrees, each corresponding to one element from the batch (B).
    """
    # 1. Separate the structure (aux_data) from the data (batched_leaves)
    batched_leaves, tree_structure = tree.flatten(batched)

    if not batched_leaves:
        # Handle empty or purely structural trees
        return [batched]

    # Get the target unbatched leaf shapes from the original tree
    original_leaves, original_structure = tree.flatten(original)

    # Input validation: Ensure the batched and original tree structures match
    if tree_structure != original_structure:
        raise ValueError("Structure of batched_tree must match original_tree.")

    # Extract target shapes
    target_shapes = [
        leaf.shape if isinstance(leaf, jnp.ndarray) else None for leaf in original_leaves
    ]

    # 2. Determine the batch size (B) from the leading dimension of the first leaf
    try:
        batch_size = batched_leaves[0].shape[0]
        if batch_size == 0:
            return []

    except IndexError:
        # Handle case where the leaf is a scalar JAX array (shape ())
        return [batched]

    # Input validation: Ensure all leaves have the same batch size
    for leaf in batched_leaves:
        if leaf.shape[0] != batch_size:
            raise ValueError(
                f"All leaves must have the same leading batch dimension. "
                f"Found shapes: {leaf.shape} and {batched_leaves[0].shape}"
            )

    # 3. Transpose/Split the leaves list by the batch dimension (B)
    # The inner function splits one array (leaf) into 'B' components (each of shape (1, ...)).
    split_leaves = [jnp.split(leaf, batch_size, axis=0) for leaf in batched_leaves]

    # 'zip' now groups the k-th elements from all split lists together
    unbatched_leaves_list = zip(*split_leaves)

    # 4. Reconstruct the Pytree for each set of unbatched leaves
    unbatched_trees = []
    for leaves_for_batch_k in unbatched_leaves_list:

        reshaped_leaves = []
        for leaf_idx, split_leaf in enumerate(leaves_for_batch_k):

            # The split_leaf is guaranteed to have shape (1, D, W, H) or (1,)
            target_shape = target_shapes[leaf_idx]

            # Use jnp.reshape to force the leaf back to its original shape.
            # This correctly handles the transformation:
            # From (1, D, W, H) -> to (D, W, H)
            # From (1,) -> to ()
            # From (1, 1) -> to (1,)
            # If target_shape is None then we have a scalar (not an array)
            reshaped_leaf = (
                jnp.reshape(split_leaf, target_shape) if target_shape is not None else split_leaf[0]
            )
            reshaped_leaves.append(reshaped_leaf)

        # Recreate the Pytree structure
        unbatched_tree = tree.unflatten(tree_structure, reshaped_leaves)
        unbatched_trees.append(unbatched_tree)

    return unbatched_trees
