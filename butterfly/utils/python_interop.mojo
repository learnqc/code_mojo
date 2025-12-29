from python import Python, PythonObject

# --- Strictly Agnostic Python Interop ---
# Standardized according to Code Mojo Principle 89:
# "The Explicit Intermediate Variable Pattern for PythonObject Conversion"


fn _ensure_path() raises:
    """Ensures current directory is in sys.path."""
    var sys = Python.import_module("sys")
    # Only append if not already there to be clean
    var path: PythonObject = sys.path
    var found = False
    for i in range(len(path)):
        if String(path[i]) == ".":
            found = True
            break
    if not found:
        sys.path.append(".")


fn get_python_func(module: String, func: String) raises -> PythonObject:
    """Imports a module and returns a function object."""
    _ensure_path()
    var py_mod = Python.import_module(module)
    return py_mod.__getattr__(func)


fn python_call(
    module: String, func: String, n: Int, value: Float64
) raises -> PythonObject:
    """Calls a Python function func(n, value) and returns the raw PythonObject.
    """
    var py_func = get_python_func(module, func)
    return py_func(n, value)


fn python_call(
    module: String, func: String, n: Int, value: Float64, iters: Int
) raises -> PythonObject:
    """Calls a Python function func(n, value, iters) and returns the raw PythonObject.
    """
    var py_func = get_python_func(module, func)
    return py_func(n, value, iters)


fn python_to_float64(obj: PythonObject) raises -> Float64:
    """Principle 89: Explicit conversion to Float64 via String/atof."""
    var py_val: PythonObject = obj
    return atof(String(py_val))
