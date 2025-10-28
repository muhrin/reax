from collections.abc import Callable
import functools
import inspect
import json
import logging
import subprocess
import tempfile
import textwrap
from typing import Final

_LOGGER = logging.getLogger(__name__)

# --- Error Marker Definitions ---
ERROR_START_MARKER: Final[str] = "---SUBPROCESS_ERROR_START---"
ERROR_END_MARKER: Final[str] = "---SUBPROCESS_ERROR_END---"


class SubprocessFunctionError(Exception):
    """
    Custom exception raised in the parent process when a failure
    occurs in the target function executed by the subprocess.
    """

    def __init__(self, original_type, original_message, traceback_details):
        self.original_type = original_type
        self.original_message = original_message
        self.traceback_details = traceback_details

        # Format the message for clean printing in the parent process
        super().__init__(
            f"Error in subprocess function: {original_type}: {original_message}\n"
            f"\n--- Subprocess Traceback Start ---\n"
            f"{traceback_details}"
            f"--- Subprocess Traceback End ---"
        )


def run_function_in_subprocess(target_func: Callable, *func_args, **func_kwargs):
    """
    Writes the source code of the file containing `target_func` to a
    temporary file, appends an execution block that handles exceptions
    by serializing them, and executes it in a new Python subprocess.

    Args:
        target_func (function): The function to be executed in the subprocess.

    Raises:
        SubprocessFunctionError: If the target function raises an exception.
    """
    func_name = target_func.__name__

    # Serialize Arguments (Parent Process)
    try:
        # Package args and kwargs into a single, JSON-serializable structure
        argument_package = {"args": func_args, "kwargs": func_kwargs}
        # Dump to JSON string, ensuring it's compact for command line
        serialized_args = json.dumps(argument_package, separators=(",", ":"))
    except TypeError as exc:
        raise TypeError(f"Error: Function arguments are not JSON serializable: {exc}") from None

    # 1. Inspect and Read Original Code
    original_file_path = inspect.getsourcefile(target_func)
    if not original_file_path:
        raise RuntimeError(f"Could not determine source file for {func_name}.")

    # 2. Inspect and Read Original Code
    original_file_path = inspect.getsourcefile(target_func)
    if not original_file_path:
        raise RuntimeError(f"Could not determine source file for {func_name}.")

    with open(original_file_path, encoding="utf-8") as f:
        original_code = f.read()

    # 3. Define the Argument and Exception-Handling Execution Block
    # The child will read the serialized arguments from sys.argv[1]
    execution_block = textwrap.dedent(
        f"""

    # --- Code injected by run_function_in_subprocess ---
    import sys
    import traceback
    import json

    # Redefine markers for the subprocess
    ERROR_START_MARKER = "{ERROR_START_MARKER}"
    ERROR_END_MARKER = "{ERROR_END_MARKER}"

    if __name__ == "__main__":
        print(f"\\n--- Subprocess Start: Running {{__file__}} as main script ---")

        # 4. Deserialize Arguments (Child Process)
        try:
            serialized_data = sys.argv[1]
            arg_data = json.loads(serialized_data)

            # Extract args and kwargs
            sub_args = arg_data.get("args", [])
            sub_kwargs = arg_data.get("kwargs", {{}})

        except IndexError:
            # Handle case where no argument data was passed (shouldn't happen here)
            sub_args = []
            sub_kwargs = {{}}
        except json.JSONDecodeError as e:
            print(f"Subprocess failed to decode arguments from command line: {{e}}")
            sys.exit(1)

        try:
            # Call the target function with the deserialized arguments
            print(
                f"Subprocess calling {{__name__}} with args: {{sub_args}} and kwargs: "
                f"{{sub_kwargs}}"
            )
            {func_name}(*sub_args, **sub_kwargs)

        except Exception as e:
            # Catch the exception, serialize it, and print it using markers

            tb_details = traceback.format_exc()

            error_data = {{
                "type": type(e).__name__,
                "message": str(e),
                "traceback": tb_details
            }}

            # Print serialized error data with markers to STDOUT
            print(ERROR_START_MARKER)
            print(json.dumps(error_data))
            print(ERROR_END_MARKER)

            # Ensure the subprocess exits with an error code
            sys.exit(1)

        print("--- Subprocess End ---")
    """
    )
    # Combine the original code and the execution block
    combined_code = original_code + execution_block

    # 5. Create and write to a temporary file.
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(combined_code)
        temp_file.close()

        _LOGGER.debug("Generated temporary file: %s", temp_file_path)

        # 6. Launch the file in a new Python subprocess, passing arguments via command line.
        try:
            # Pass the serialized args as the third command-line argument (index 1 in sys.argv)
            cmd = ["python", temp_file_path, serialized_args]

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # nosec

            # 7. Check for Serialized Exception (Custom Logic)
            if result.returncode != 0:

                if ERROR_START_MARKER in result.stdout and ERROR_END_MARKER in result.stdout:

                    start_index = result.stdout.find(ERROR_START_MARKER) + len(ERROR_START_MARKER)
                    end_index = result.stdout.find(ERROR_END_MARKER)

                    json_string = result.stdout[start_index:end_index].strip()

                    try:
                        error_data = json.loads(json_string)
                        raise SubprocessFunctionError(
                            error_data["type"], error_data["message"], error_data["traceback"]
                        )
                    except json.JSONDecodeError:
                        pass

                # Fallback for generic non-zero exit code
                raise subprocess.CalledProcessError(
                    result.returncode, result.args, output=result.stdout, stderr=result.stderr
                )

            _LOGGER.debug(
                "Subprocess finished successfully with return code: %d", result.returncode
            )

        except FileNotFoundError:
            raise FileNotFoundError(
                "\nError: Python executable not found. Make sure 'python' is in your PATH."
            ) from None

    _LOGGER.debug("Cleaned up temporary file: %s", temp_file_path)


def in_subprocess(fn: Callable):
    """Run a test within a subprocess, necessary for certain strategies."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        run_function_in_subprocess(fn, *args, **kwargs)

    return wrapper
