import open3d as o3d
import numpy as np

def point_cloud_to_mesh_poisson(point_cloud_data):
    """
    Takes a point cloud object, visualizes it,
    creates a closed mesh using Poisson reconstruction,
    and visualizes the result.
    """
   
    if not point_cloud_data.has_points():
        print("Error: The point cloud object is empty.")
        return
        
    print(f"-> The point cloud has {len(point_cloud_data.points)} points.")
    o3d.visualization.draw_geometries([point_cloud_data], window_name="Point Cloud")
  
    print("-> Computing normal vectors for the point cloud...")
    point_cloud_data.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    
    o3d.visualization.draw_geometries([point_cloud_data], 
                                  window_name="Point Cloud with Normals",
                                  point_show_normal=True) 
    

    point_cloud_data.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([point_cloud_data], window_name="Point Cloud with Normals")

    print("-> Creating mesh from the point cloud with the Poisson algorithm...")

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud_data, depth=11)

    print("-> Removing edges and artifacts...")
    bbox = point_cloud_data.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)


    print("-> Visualizing the reconstructed mesh...")
    o3d.visualization.draw_geometries([mesh], window_name="Generated Mesh (Poisson)")
    

    output_path = "output_bunny_poisson_mesh.obj"
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"-> The mesh was saved as '{output_path}'.")
    

if __name__ == "__main__":
    
    print("-> Loading the Stanford Bunny mesh and converting it to a point cloud...")
    data = o3d.data.BunnyMesh()
    bunny_mesh = o3d.io.read_triangle_mesh(data.path)
    

    bunny_pcd = bunny_mesh.sample_points_poisson_disk(number_of_points=20000)
   
    point_cloud_to_mesh_poisson(bunny_pcd)

