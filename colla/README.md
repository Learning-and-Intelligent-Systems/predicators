# MiniBehavior Gridworld for CoLLA 2026

This directory contains the MiniBehavior gridworld environment setup for CoLLA 2026.

## Setup Instructions

### 1. Install Requirements

Navigate to the `colla/` directory and install the required packages:

```bash
cd colla/
pip install -r requirements.txt
```

### 2. Run the Main Script

From the `colla/` directory, run:

```bash
python main.py
```
(should take ~3min to complete)

### 3. Fix Known Package Issues

After installation, you may encounter two errors that require manual fixes in the installed packages:

#### Error 1: Gymnasium Package

**Location:** In your gymnasium package installation (typically in `site-packages/gymnasium/envs/registration.py`)

**Find this code:**
```python
# Update the env spec kwargs with the `make` kwargs
    env_spec_kwargs = copy.deepcopy(env_spec.kwargs)
    env_spec_kwargs.update(kwargs)
```

**Replace with:**
```python
# Update the env spec kwargs with the `make` kwargs
    env_spec_kwargs = copy.deepcopy(env_spec.kwargs)
    env_spec_kwargs = {}
    env_spec_kwargs.update(kwargs)
```

#### Error 2: Minigrid Environment

**Location:** In your minigrid package installation (typically in `site-packages/minigrid/minigrid_env.py`)

**Find this code:**
```python
        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps
```

**Replace with:**
```python
max_steps = int(max_steps)
        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps
```

## Notes

- These fixes are temporary workarounds for compatibility issues between the packages
- Make sure to apply these fixes in your Python environment's site-packages directory
- You can find your site-packages location by running: `python -c "import site; print(site.getsitepackages())"`

## Output and Results

- The most recent frame is saved to `output_image.jpeg`
- Results including NSRTs, CSVs, and logs are saved in the `results/` directory
- Demos are located in `../demos/` (relative to the `colla/` directory)
  - To view demos, run: `python view_demos.py`

## Next Steps

1. Fix remaining MiniBehavior environments (should be 20 total)
2. Implement LLM baseline operator learner
3. Add predicate invention
4. Generate results for CoLLA
