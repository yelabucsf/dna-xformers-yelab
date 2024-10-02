# Guide to Adding New Code to src/dnaformer

This guide outlines the steps to add new code to the `src/dnaformer` directory and ensure proper installation and importability.

## Steps for Adding New Code

1. **Adding a New File:**
   - Create the new file in the appropriate subdirectory of `src/dnaformer/` (e.g., `src/dnaformer/models/new_model.py`).
   - Include a docstring at the top of the file explaining its purpose.

2. **Adding New Classes or Functions:**
   - Write your new class or function in the appropriate file.
   - Include proper type hints and docstrings for clarity and maintainability.

3. **Updating `__init__.py` Files:**
   - If you've added a new file, update the `__init__.py` file in the corresponding directory to import the new module.
   - For example, if you added `src/dnaformer/models/new_model.py`, update `src/dnaformer/models/__init__.py`:
     ```python
     from .new_model import NewModel
     ```
   - If you want the new class/function to be directly importable from the top-level package, also update `src/__init__.py`.

4. **Updating `setup.py`:**
   - If you've added a new subdirectory, make sure it's included in the `packages` parameter of the `setup()` function in `setup.py`.
   - The current setup using `find_packages(where="src")` should automatically include new subdirectories, but double-check to be sure.

5. **Updating Requirements:**
   - If your new code requires additional dependencies, add them to `environment/requirements.txt`.

6. **Testing the Installation:**
   - After making changes, reinstall the package in editable mode:
     ```
     pip install -e .
     ```
   - Try importing your new module/class/function in a Python interpreter to ensure it's accessible.

7. **Updating Documentation:**
   - If you have separate documentation, update it to include information about the new code.
   - Consider adding examples of how to use the new functionality.

8. **Version Control:**
   - If you're using version control (e.g., git), commit your changes, including the new file and any modified `__init__.py` files.

## Example Workflow

Let's say you want to add a new model called `AdvancedModel` in a new file `src/dnaformer/models/advanced_model.py`:

1. Create `src/dnaformer/models/advanced_model.py`:
   ```python
   class AdvancedModel:
       """An advanced model for DNA sequence analysis."""
       def __init__(self, param1, param2):
           self.param1 = param1
           self.param2 = param2
       
       def process(self, data):
           """Process the input data."""
           # Implementation here
   ```

2. Update `src/dnaformer/models/__init__.py`:
   ```python
   from .roformer import RoformerModel, SimpleTransformerModel
   from .advanced_model import AdvancedModel
   ```

3. If you want it to be directly importable, update `src/__init__.py`:
   ```python
   from .dnaformer.models import AdvancedModel
   ```

4. If `AdvancedModel` requires a new dependency, add it to `environment/requirements.txt`.

5. Reinstall the package:
   ```
   pip install -e .
   ```

6. Test the import:
   ```python
   from dnaformer.models import AdvancedModel
   # or
   from dnaformer import AdvancedModel
   ```

By following these steps, you ensure that your new code is properly integrated into the package structure and is importable after installation. This process maintains the organization of your project and makes it easy for users (including yourself) to access and use the new functionality.
