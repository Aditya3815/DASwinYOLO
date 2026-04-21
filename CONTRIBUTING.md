# Contributing to DASwin-YOLO

Thank you for your interest in contributing to DASwin-YOLO! We welcome contributions from the community to help improve the architecture, fix bugs, or expand its utility in remote sensing and dense object detection.

## 🛠️ Submitting a Pull Request (PR)

1. **Fork the repository** and create your branch from `master`.
2. **Make your changes**. If you are modifying the architectural components (`dc3swt.py`, `coord_attention.py`, `bifpn.py`, `yolo.py`), please ensure your modifications align with the overall design philosophy (e.g., maintaining efficient tensor operations natively in PyTorch).
3. **Run the Test Suite**: Before submitting your PR, ensure that all 36 architectural unit tests pass. We rely heavily on these tests to prevent regression in the deformable attention module and coordinate attention components.
   ```bash
   python tests/test_components.py
   ```
4. **Submit your PR** with a clear description of the problem and the implemented solution.

## 🐛 Submitting a Bug Report

If you encounter an issue, please submit a bug report via GitHub Issues. To help us reproduce the issue, please include:
- A minimum reproducible example (code, dataset snippet, or config YAML).
- Ensure your issue is reproducible on the latest `master` branch.
- Log traces of any errors, especially if they occur during the `DC3SWT` or `DeformConv2d` forward passes.

## 📄 License

By contributing, you agree that your contributions will be licensed under the [AGPL-3.0 license](https://choosealicense.com/licenses/agpl-3.0/).
