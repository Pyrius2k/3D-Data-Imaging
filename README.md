# 🧠 3D Data Imaging with Python

This repository provides a collection of Python scripts for 3D data imaging, point cloud processing, mesh reconstruction, and 3D model generation using `Open3D`, `PyTorch`, `Diffusers`, and other powerful Python tools. It includes:

- ✅ Ball Pivoting Algorithm for mesh reconstruction
- ✅ Poisson Surface Reconstruction
- ✅ 2D to 3D conversion using monocular depth estimation and point cloud generation
- ✅ Prompt-to-3D point cloud generation with Point-E
- ✅ Prompt-to-3D model generation with Shap-E

---

## 📁 Contents

- [`Ballpivot.py`](./Ballpivot.py) — Generate surface meshes from point clouds using Ball Pivoting.
- [`Poisson.py`](./Poisson.py) — Reconstruct watertight meshes using Poisson Surface Reconstruction.
- [`2dto3d.py`](./2dto3d.py) — Convert 2D RGB images to 3D point clouds using depth estimation.
- [`PointE.py`](./PointE.py) — Generate 3D point clouds from text prompts using OpenAI's Point-E.
- [`ShapE.py`](./ShapE.py) — Generate 3D textured meshes from text prompts using OpenAI's Shap-E.
- [`shapE.ipynb`](./shapE.ipynb) — A Jupyter Notebook demonstrating Shap-E text-to-3D generation.

---

## 🔧 Requirements

Install dependencies using pip:

```bash
pip install open3d numpy matplotlib
```

> Optionally, for 2D-to-3D depth estimation, install MiDaS or another depth model:

```bash
pip install torch torchvision
pip install timm  # Required for MiDaS
```
> For Point-E and Shap-E, install diffusers and transformers, along with torch if not already installed:

```bash
pip install torch transformers diffusers accelerate
```
---

## 📌 Features

### 1. Ball Pivoting Mesh Reconstruction

The Ball Pivoting Algorithm (BPA) reconstructs a mesh from a point cloud by "rolling" a ball across the surface and connecting points that lie on the ball’s surface.

**Run:**

```bash
python Ballpivot.py --input input_point_cloud.ply
```

---

### 2. Poisson Surface Reconstruction

Poisson reconstruction creates watertight surfaces from oriented point clouds, producing high-quality meshes even with noisy input data.

**Run:**

```bash
python Poisson.py --input input_point_cloud.ply
```

---

### 3. 2D to 3D with Depth Estimation

This script takes a 2D image and transforms it into a 3D point cloud using monocular depth estimation and Open3D for point cloud generation.

#### 🔄 Transformation Example

<table>
  <tr>
    <td><strong>2D Input Image</strong></td>
    <td><strong>3D Output Visualization</strong></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/Pyrius2k/3D-Data-Imaging/blob/main/gemini.image2.png?raw=true" width="300">
    </td>
    <td>
      <img src="https://github.com/Pyrius2k/3D-Data-Imaging/blob/main/mushroom.png?raw=true" width="300">
    </td>
  </tr>
</table>

**Run:**

```bash
python 2dto3d.py --image path_to_your_image.jpg
```

**Steps:**

1. Load the input image  
2. Estimate depth using a pre-trained model (e.g., MiDaS)  
3. Generate a 3D point cloud using the estimated depth and camera intrinsics  
4. Visualize and optionally export the 3D output  

---

### 4. Point-E: Prompt-to-3D Point Cloud Generation

Point-E is a system for generating 3D point clouds from text prompts. It first generates a synthetic view and then uses a point cloud diffusion model to create the 3D representation.

**Run:**

```bash
python PointE.py --prompt "a high-resolution 3D point cloud of a futuristic spaceship"
```

Key Capabilities:
- Generate diverse 3D point clouds directly from textual descriptions.
- Ideal for quick conceptualization and prototyping of 3D forms.

---

### 5. Shap-E: Prompt-to-3D Textured Mesh Generation

Shap-E is a conditional diffusion model that generates 3D assets, including textured implicit functions and explicit meshes, from various inputs like text or images.

**Run:**

```bash
python ShapE.py --prompt "a detailed model of an antique teapot with a delicate pattern"
```

Key Capabilities:
- Generate high-quality 3D textured meshes from text prompts.
- Supports richer detail and surface properties compared to raw point clouds.
- Suitable for creating ready-to-use 3D models.
- An interactive demonstration is available in shapE.ipynb.

# 🤖 Generated Example: War Robot
Here's an example of a 3D war robot generated using Shap-E:

<img src="https://github.com/Pyrius2k/3D-Data-Imaging/blob/main/robot.gif?raw=true" alt="War Robot 3D Model" width="300">

## 🧪 Demo Data

- Sample point clouds: `data/sample_pointcloud.ply`
- Sample images: `data/sample_image.jpg`

---

## 🚀 Future Improvements

- [ ] GUI integration 
- [ ] Integration with LiDAR datasets
- [ ] Implement advanced mesh optimization techniques for Point-E and Shap-E outputs

---

## 🤝 Contributions

Feel free to open issues or pull requests to contribute new methods, bug fixes, or enhancements!

---

## 📜 License

This project is licensed under the MIT License.

---

## 📚 References

📄 [![Open3D Documentation](https://img.shields.io/badge/Open3D%20Documentation-blue?style=for-the-badge)](https://www.open3d.org/)

📄 [![MiDaS Depth Estimation](https://img.shields.io/badge/MiDaS%20Depth%20Estimation-green?style=for-the-badge)](https://github.com/isl-org/MiDaS)

📄 [![Poisson Surface Reconstruction (Kazhdan et al.)](https://img.shields.io/badge/Poisson%20Surface%20Reconstruction-orange?style=for-the-badge)](https://www.cs.jhu.edu/~misha/Code/PoissonRecon/)

📄 [![Ball Pivoting Algorithm (Bernardini et al.)](https://img.shields.io/badge/Ball%20Pivoting%20Algorithm-red?style=for-the-badge)](https://www.researchgate.net/publication/220494622_The_Ball-Pivoting_Algorithm_for_Surface_Reconstruction)

📄 [![Point-E (Chen et al.)](https://img.shields.io/badge/Ball%20Pivoting%20Algorithm-red?style=for-the-badge)]([https://www.researchgate.net/publication/220494622_The_Ball-Pivoting_Algorithm_for_Surface_Reconstruction](https://arxiv.org/abs/2212.08751))


