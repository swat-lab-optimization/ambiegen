# Usage

## Generating Tests

Run the following to generate behavior-rich test cases:

```bash
python generate_tests.py --config configs/default.yaml
```

## Comparing Outputs

Use the `compare.py` script to compare results:

```bash
python compare.py --reference results/ref --test results/test
```

