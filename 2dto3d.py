import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Benutzerdefinierte Funktion für die orthographische Projektion
def depth_to_pointcloud_orthographic(depth_map, image, scale_factor=255):
    height, width = depth_map.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    z = (depth_map / scale_factor) * height / 2

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    mask = points[:, 2] != 0
    points = points[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = image.reshape(-1, 3)[mask] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    _, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1)
    inlier_cloud = pcd.select_by_index(ind)

    return inlier_cloud, z, height, width

# --- Hauptskript ---
folder_path = Path("D:/Graphics/Bilder")
export_path = Path("D:/Graphics/Results")
exporting = True

if not folder_path.exists():
    print(f"Fehler: Der Pfad {folder_path} existiert nicht.")
    exit()

if not export_path.exists():
    os.makedirs(export_path)

num_samples = 2
selection = random.sample(os.listdir(folder_path), num_samples)

selected_images = []
for i in range(num_samples):
    pathi = str(folder_path / selection[i])
    selected_image = cv2.imread(pathi)
    selected_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB)
    selected_images.append(selected_image)

checkpoint = "LiheYoung/depth-anything-large-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Verwendetes Gerät: {device}")

processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)

depth_samples = []
for i in range(num_samples):
    depth_input = processor(images=selected_images[i], return_tensors="pt").to(device)
    with torch.no_grad():
        inference_outputs = model(**depth_input)
        output_depth = inference_outputs.predicted_depth
    output_depth = output_depth.squeeze().cpu().numpy()
    depth_samples.append([selected_images[i], output_depth])

# --- Verarbeite alle Bilder in einer einzigen Schleife ---
for i in range(num_samples):
    print(f"\nVerarbeite Bild {i+1}...")
    
    depth_image_raw = depth_samples[i][1]
    color_image = depth_samples[i][0]
    width, height = depth_image_raw.shape

    # 1. Plots anzeigen
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(color_image)
    axs[0].set_title('Originalbild')
    axs[1].imshow(depth_image_raw, cmap='plasma')
    axs[1].set_title('Tiefenschätzung')
    plt.tight_layout()
    plt.show()

    # 2. Bilder für die Visualisierung speichern
    depth_image_display = (depth_image_raw * 255 / np.max(depth_image_raw)).astype('uint8')
    color_image_resized = cv2.resize(color_image, (height, width))
    cv2.imwrite(str(export_path / f'Results{i}.png'), cv2.cvtColor(color_image_resized, cv2.COLOR_BGR2RGB))
    cv2.imwrite(str(export_path / f'Results{i}_depth.png'), depth_image_display)

    # 3. Perspektivische Projektion (Pinhole-Kamera)
    depth_o3d_persp = o3d.geometry.Image(depth_image_raw)
    image_o3d_persp = o3d.geometry.Image(color_image_resized)
    rgbd_image_persp = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d_persp, depth_o3d_persp, convert_rgb_to_intensity=False)
    
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    fx = fy = width * 0.8
    cx, cy = width / 2, height / 2
    camera_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    pcd_persp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_persp, camera_intrinsic)
    pcd_persp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 4. Orthographische Projektion
    pcd_ortho, _, _, _ = depth_to_pointcloud_orthographic(depth_image_raw, color_image_resized)

    # 5. Visualisierung und Export der Ergebnisse
    print(f"Zeige perspektivische Ansicht für Bild {i+1} an.")
    o3d.visualization.draw_geometries([pcd_persp])
    
    print(f"Zeige orthographische Ansicht für Bild {i+1} an.")
    o3d.visualization.draw_geometries([pcd_ortho])

    if exporting:
        o3d.io.write_point_cloud(str(export_path / f'pcd_pinhole_{i}.ply'), pcd_persp)
        o3d.io.write_point_cloud(str(export_path / f'pcd_ortho_{i}.ply'), pcd_ortho)
        print("Ergebnisse gespeichert.")

    # --- Mesh-Erstellung ---
    # Normalen-Schätzung und Poisson-Rekonstruktion für beide Punktwolken
    
    # Mesh für Perspektivische Projektion
    pcd_persp.estimate_normals()
    pcd_persp.orient_normals_to_align_with_direction()
    mesh_persp, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_persp, depth=9)
    print('run Poisson surface reconstruction for Pinhole')
    o3d.visualization.draw_geometries([mesh_persp])
    if exporting:
        o3d.io.write_triangle_mesh(str(export_path / f'mesh_pinhole_{i}.obj'), mesh_persp, write_triangle_uvs=True)

    # Mesh für Orthographische Projektion
    pcd_ortho.estimate_normals()
    pcd_ortho.orient_normals_to_align_with_direction()
    mesh_ortho, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_ortho, depth=9)
    print('run Poisson surface reconstruction for Orthographic')
    o3d.visualization.draw_geometries([mesh_ortho])
    if exporting:
        o3d.io.write_triangle_mesh(str(export_path / f'mesh_ortho_{i}.obj'), mesh_ortho, write_triangle_uvs=True)