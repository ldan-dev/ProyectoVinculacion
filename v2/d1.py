import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import csv
import os
import time

# Configuración del usuario
PART_NAME = "switch"          # Nombre de la pieza
NUM_CAPTURES = 12             # Número de capturas (12 = cada 30°)
MIN_DIST = 0.3                # Distancia mínima (metros)
MAX_DIST = 0.8                # Distancia máxima (metros)
WAIT_TIME = 2                 # Segundos de espera entre capturas
OUTLIER_REMOVAL = 0.5         # Nivel de filtrado de ruido (0-1)

# Iniciar cámara
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    # Esperar estabilización de la cámara
    for _ in range(30):
        pipeline.wait_for_frames()
    
    # Crear carpeta para resultados
    os.makedirs(PART_NAME, exist_ok=True)
    
    for i in range(NUM_CAPTURES):
        print(f"\n Preparando captura {i+1}/{NUM_CAPTURES}")
        print(f" Tienes {WAIT_TIME} segundos para girar la pieza...")
        
        # Espera para girar la pieza
        time.sleep(WAIT_TIME)
        
        # Capturar frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("⚠ Error en captura. Reintentando...")
            continue
        
        # Procesar nube de puntos
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(points.get_texture_coordinates())
        
        # Filtrar por distancia y obtener colores
        valid_points = []
        colors = []
        color_image = np.asanyarray(color_frame.get_data())
        width, height = color_image.shape[1], color_image.shape[0]
        
        for vertex, tex_coord in zip(vtx, tex):
            if MIN_DIST <= vertex[2] <= MAX_DIST:
                u = int(tex_coord[0] * width)
                v = int(tex_coord[1] * height)
                u = min(max(u, 0), width - 1)
                v = min(max(v, 0), height - 1)
                bgr = color_image[v, u]
                rgb = [bgr[2], bgr[1], bgr[0]]
                valid_points.append(vertex)
                colors.append(rgb)
        
        # Crear nube de puntos Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(valid_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors)/255.0)
        
        # Filtrar outliers
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=20, 
            std_ratio=OUTLIER_REMOVAL
        )
        filtered_pcd = pcd.select_by_index(ind)
        
        # Guardar en CSV
        csv_path = f"{PART_NAME}/{PART_NAME}_{i}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x','y','z','r','g','b'])
            points = np.asarray(filtered_pcd.points)
            colors = (np.asarray(filtered_pcd.colors)*255).astype(int)
            for pt, clr in zip(points, colors):
                writer.writerow([pt[0], pt[1], pt[2], clr[0], clr[1], clr[2]])
        
        # Guardar en PLY
        ply_path = f"{PART_NAME}/{PART_NAME}_{i}.ply"
        o3d.io.write_point_cloud(ply_path, filtered_pcd)
        
        # Visualizar captura actual
        print("👁 Visualizando captura (cierra ventana para continuar)...")
        o3d.visualization.draw_geometries([filtered_pcd], 
                                          window_name=f"Captura {i+1}/{NUM_CAPTURES}")
        
        print(f"✅ Captura {i+1} guardada:")
        print(f" - CSV: {csv_path}")
        print(f" - PLY: {ply_path}")

finally:
    pipeline.stop()

print("\n🎉 Todas las capturas completadas!")
print(f"📁 Carpeta de resultados: {os.path.abspath(PART_NAME)}")
