# 🧠 3D Data Imaging with Python

This repository provides a collection of Python scripts for 3D data imaging, point cloud processing, and mesh reconstruction using `Open3D` and other Python tools. It includes:

- ✅ Ball Pivoting Algorithm for mesh reconstruction  
- ✅ Poisson Surface Reconstruction  
- ✅ 2D to 3D conversion using monocular depth estimation and point cloud generation

---

## 📁 Contents

- [`Ballpivot.py`](./Ballpivot.py) — Generate surface meshes from point clouds using Ball Pivoting.
- [`Poisson.py`](./Poisson.py) — Reconstruct watertight meshes using Poisson Surface Reconstruction.
- [`2dto3d.py`](./2dto3d.py) — Convert 2D RGB images to 3D point clouds using depth estimation.

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

## 🧪 Demo Data

- Sample point clouds: `data/sample_pointcloud.ply`
- Sample images: `data/sample_image.jpg`

---

## 🚀 Future Improvements

- [ ] Add Marching Cubes algorithm  
- [ ] GUI integration with Gradio or Streamlit  
- [ ] Support for video-to-3D frame-by-frame  
- [ ] Integration with LiDAR datasets  

---

## 🤝 Contributions

Feel free to open issues or pull requests to contribute new methods, bug fixes, or enhancements!

---

## 📜 License

This project is licensed under the MIT License.

---

## 📚 References

- [Open3D Documentation](https://www.open3d.org/)  
- [MiDaS Depth Estimation](https://github.com/isl-org/MiDaS)  
- Kazhdan et al. — Poisson Surface Reconstruction  
- Bernardini et al. — Ball Pivoting Algorithm  
