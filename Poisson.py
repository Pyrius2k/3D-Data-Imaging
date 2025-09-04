import open3d as o3d
import numpy as np

def point_cloud_to_mesh_poisson(point_cloud_data):
    """
    Nimmt ein Punktwolken-Objekt entgegen, visualisiert es,
    erzeugt ein geschlossenes Mesh mit Poisson-Rekonstruktion
    und visualisiert das Ergebnis.
    """
    
    # 1. Überprüfen und Visualisieren der ursprünglichen Punktwolke
    if not point_cloud_data.has_points():
        print("Fehler: Das Punktwolken-Objekt ist leer.")
        return
        
    print(f"-> Die Punktwolke hat {len(point_cloud_data.points)} Punkte.")
    o3d.visualization.draw_geometries([point_cloud_data], window_name="Punktwolke")
    
    # 2. Berechnung der Normalenvektoren
    # Die Normalen sind für die Poisson-Rekonstruktion unerlässlich.
    print("-> Berechne Normalenvektoren für die Punktwolke...")
    point_cloud_data.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    
    o3d.visualization.draw_geometries([point_cloud_data], 
                                 window_name="Punktwolke mit Normalen",
                                 point_show_normal=True) 
    
    # Normalenvektoren zur besseren Darstellung färben
    point_cloud_data.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([point_cloud_data], window_name="Punktwolke mit Normalen")

    # 3. Erzeugen des Meshes mit der Poisson-Rekonstruktion
    print("-> Erzeuge Mesh aus der Punktwolke mit Poisson-Algorithmus...")
    # Die "depth" bestimmt die Detailgenauigkeit. Ein Wert von 9 ist ein guter Kompromiss.
    # Höhere Werte wie 10 oder 11 erhöhen die Detailtreue, sind aber rechenintensiver.
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud_data, depth=11)
    
    # 4. Abschneiden des Meshes an den Rändern, um Artefakte zu entfernen
    # Da Poisson ein geschlossenes Volumen erzeugt, können unerwünschte
    # Reste entstehen, die mit der Bounding Box entfernt werden.
    print("-> Entferne Ränder und Artefakte...")
    bbox = point_cloud_data.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # 5. Visualisieren des fertigen Meshes
    print("-> Visualisiere das rekonstruierte Mesh...")
    o3d.visualization.draw_geometries([mesh], window_name="Erzeugtes Mesh (Poisson)")
    
    # 6. Speichern des Meshes als OBJ-Datei (optional)
    output_path = "output_bunny_poisson_mesh.obj"
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"-> Das Mesh wurde als '{output_path}' gespeichert.")
    
# ---- Hauptfunktion ----
if __name__ == "__main__":
    
    # Hier wird das Hasen-Mesh geladen und in eine Punktwolke umgewandelt
    print("-> Lade das Stanford Bunny Mesh und konvertiere es in eine Punktwolke...")
    data = o3d.data.BunnyMesh()
    bunny_mesh = o3d.io.read_triangle_mesh(data.path)
    
    # Wir erzeugen eine Punktwolke mit 50000 Punkten, um ein besseres Ergebnis zu erzielen
    # Eine höhere Punktdichte ist für Poisson vorteilhaft
    bunny_pcd = bunny_mesh.sample_points_poisson_disk(number_of_points=20000)
    
    # Rufe die Hauptfunktion mit der erstellten Punktwolke auf
    point_cloud_to_mesh_poisson(bunny_pcd)