import open3d as o3d
import numpy as np

def point_cloud_to_mesh(point_cloud_data):
    """
    Takes a point cloud object, visualizes it,
    creates a mesh, and visualizes the result.
    """
    
    # 1. Visualize the original point cloud
    print(f"-> The point cloud has {len(point_cloud_data.points)} points.")
    o3d.visualization.draw_geometries([point_cloud_data], window_name="Punktwolke")
    
    # 2. Compute normal vectors
    print("-> Computing normal vectors for the point cloud...")
    point_cloud_data.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    # 3. Create the mesh using the ball_pivoting method
    print("-> Creating mesh from the point cloud...")
    distances = point_cloud_data.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    
    # Define the radii for the Ball Pivoting algorithm
    # Corrected line: a simple list of floats is all that's needed
    radii = [avg_dist, avg_dist * 2, avg_dist * 4]
    
    # Perform the reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud_data, o3d.utility.DoubleVector(radii))
    
    # 4. Visualize the finished mesh
    print("-> Visualizing the reconstructed mesh...")
    o3d.visualization.draw_geometries([mesh], window_name="Erzeugtes Mesh")
    
    # 5. Save the mesh as an OBJ file (optional)
    output_path = "output_bunny_mesh.obj"
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"-> The mesh was saved as '{output_path}'.")
    
# ---- Main function ----
if __name__ == "__main__":
    
    # Here, the bunny mesh is loaded and converted into a point cloud.
    print("-> Loading the Stanford Bunny mesh and converting it to a point cloud...")
    data = o3d.data.BunnyMesh()
    bunny_mesh = o3d.io.read_triangle_mesh(data.path)
    
    # We generate a point cloud with 5000 points from the mesh.
    bunny_pcd = bunny_mesh.sample_points_poisson_disk(number_of_points=40000)
    
    # Call the main function with the created point cloud
    point_cloud_to_mesh(bunny_pcd)