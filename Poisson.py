import open3d as o3d
import numpy as np

def point_cloud_to_mesh_poisson(point_cloud_data):
    """
    Takes a point cloud object, visualizes it,
    creates a closed mesh using Poisson reconstruction,
    and visualizes the result.
    """
    
    # 1. Check and visualize the original point cloud
    if not point_cloud_data.has_points():
        print("Error: The point cloud object is empty.")
        return
        
    print(f"-> The point cloud has {len(point_cloud_data.points)} points.")
    o3d.visualization.draw_geometries([point_cloud_data], window_name="Point Cloud")
    
    # 2. Compute normal vectors
    # The normals are essential for Poisson reconstruction.
    print("-> Computing normal vectors for the point cloud...")
    point_cloud_data.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    
    o3d.visualization.draw_geometries([point_cloud_data], 
                                  window_name="Point Cloud with Normals",
                                  point_show_normal=True) 
    
    # Color the normal vectors for better visualization
    point_cloud_data.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([point_cloud_data], window_name="Point Cloud with Normals")

    # 3. Create the mesh using Poisson reconstruction
    print("-> Creating mesh from the point cloud with the Poisson algorithm...")
    # The "depth" determines the level of detail. A value of 9 is a good compromise.
    # Higher values like 10 or 11 increase detail but are more computationally intensive.
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud_data, depth=11)
    
    # 4. Crop the mesh at the edges to remove artifacts
    # Since Poisson creates a closed volume, unwanted
    # remnants can arise, which are removed with the bounding box.
    print("-> Removing edges and artifacts...")
    bbox = point_cloud_data.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # 5. Visualize the finished mesh
    print("-> Visualizing the reconstructed mesh...")
    o3d.visualization.draw_geometries([mesh], window_name="Generated Mesh (Poisson)")
    
    # 6. Save the mesh as an OBJ file (optional)
    output_path = "output_bunny_poisson_mesh.obj"
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"-> The mesh was saved as '{output_path}'.")
    
# ---- Main function ----
if __name__ == "__main__":
    
    # Here the bunny mesh is loaded and converted into a point cloud
    print("-> Loading the Stanford Bunny mesh and converting it to a point cloud...")
    data = o3d.data.BunnyMesh()
    bunny_mesh = o3d.io.read_triangle_mesh(data.path)
    
    # We create a point cloud with 20000 points to get a better result
    # A higher point density is advantageous for Poisson
    bunny_pcd = bunny_mesh.sample_points_poisson_disk(number_of_points=20000)
    
    # Call the main function with the created point cloud
    point_cloud_to_mesh_poisson(bunny_pcd)
