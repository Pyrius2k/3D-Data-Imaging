import open3d as o3d
import numpy as np

def point_cloud_to_mesh(point_cloud_data):
    """
    Takes a point cloud object, visualizes it,
    creates a mesh, and visualizes the result.
    """
    

    print(f"-> The point cloud has {len(point_cloud_data.points)} points.")
    o3d.visualization.draw_geometries([point_cloud_data], window_name="Punktwolke")
    

    print("-> Computing normal vectors for the point cloud...")
    point_cloud_data.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))


    print("-> Creating mesh from the point cloud...")
    distances = point_cloud_data.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    
 
    radii = [avg_dist, avg_dist * 2, avg_dist * 4]
    

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud_data, o3d.utility.DoubleVector(radii))
    

    print("-> Visualizing the reconstructed mesh...")
    o3d.visualization.draw_geometries([mesh], window_name="Erzeugtes Mesh")
    

    output_path = "output_bunny_mesh.obj"
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"-> The mesh was saved as '{output_path}'.")
    

if __name__ == "__main__":
    
   
    print("-> Loading the Stanford Bunny mesh and converting it to a point cloud...")
    data = o3d.data.BunnyMesh()
    bunny_mesh = o3d.io.read_triangle_mesh(data.path)
    

    bunny_pcd = bunny_mesh.sample_points_poisson_disk(number_of_points=40000)


    point_cloud_to_mesh(bunny_pcd)
