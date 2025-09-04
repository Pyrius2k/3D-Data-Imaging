# 🧠 3D Data Imaging with Python

This repository provides a collection of Python scripts for 3D data imaging, point cloud processing, and mesh reconstruction using `Open3D` and other Python tools. It includes:

- ✅ Ball Pivoting Algorithm for mesh reconstruction  
- ✅ Poisson Surface Reconstruction  
- ✅ 2D to 3D conversion using monocular depth estimation and point cloud generation

---

## 📁 Contents

- [`Ballpivot.py`](./Ballpivot.py) - Generate surface meshes from point clouds using Ball Pivoting.
- [`Poisson.py`](./Poisson.py) - Reconstruct watertight meshes using Poisson Surface Reconstruction.
- [`2dto3d.py`](./2dto3d.py) - Convert 2D RGB images to 3D point clouds using depth estimation.

---

## 🔧 Requirements

Install dependencies using pip:

```bash
pip install open3d numpy matplotlib
