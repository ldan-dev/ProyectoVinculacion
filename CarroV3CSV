# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:03:52 2025

@author: dsplab_1
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
import pandas as pd

def capture_point_cloud():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # Create a point cloud object
    pc = rs.pointcloud()
    
    # Create a list to store point clouds
    point_clouds = []
    
    # Wait for a coherent pair of frames: depth and color
    print("Comenzando captura en 5 segundos...")
    print("Mueve la cámara alrededor del objeto lentamente para capturar todos los lados")
    time.sleep(5)
    
    # Definir los límites de distancia en metros
    # Se ajusta para que el objeto a 10 cm sea el principal, y se ignoren los puntos más lejanos
    dist_min = 0.05  # 5 cm
    dist_max = 0.10  # 50 cm
    n_frames = 30    # n frames
    
    try:
        # Capture 60 frames (about 2 seconds at 30 fps)
        for i in range(n_frames):
            print(f"Capturando frame {i+1}/60")
            
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Map color texture to each point
            pc.map_to(color_frame)
            
            # Generate point cloud
            points = pc.calculate(depth_frame)
            
            # Get vertices and texture coordinates
            v = points.get_vertices()
            t = points.get_texture_coordinates()
            
            # Convert to numpy arrays
            vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)
            texture_coords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
            
            # Convert the pointcloud to Open3D format
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(vertices)
            
            # Get color image as numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Map colors to points
            colors = []
            
            # Iterate through the texture coordinates array
            for u_coord, v_coord in texture_coords:
                u, v = int(u_coord * color_frame.get_width()), int(v_coord * color_frame.get_height())
                if u < color_image.shape[1] and v < color_image.shape[0]:
                    # Get RGB (reversed because OpenCV uses BGR)
                    colors.append(color_image[v, u][::-1] / 255.0)  # Still normalized for Open3D
                else:
                    colors.append([0, 0, 0])
            
            o3d_pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
            
            # --- NORMA AGREGADA: Filtrar puntos por distancia ---
            # Se crea un filtro para mantener solo los puntos que están entre dist_min y dist_max
            # El eje Z representa la profundidad o la distancia de la cámara
            distances = np.asarray(vertices)[:, 2]
            valid_indices = np.where((distances > dist_min) & (distances < dist_max))[0]
            
            # Aplicar el filtro al PointCloud de Open3D
            o3d_pcd = o3d_pcd.select_by_index(valid_indices)
            
            # --- FIN DE LA NORMA AGREGADA ---
            
            # Add to our collection
            if len(o3d_pcd.points) > 0:
                point_clouds.append(o3d_pcd)
            
            time.sleep(0.1)  # Wait a bit between frames to allow camera movement
        
        print("¡Captura completada!")
        
        if not point_clouds:
            print("¡No se capturaron nubes de puntos!")
            return
            
        # Combine all point clouds
        print("Procesando nubes de puntos...")
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in point_clouds:
            combined_pcd += pcd
        
        # Voxel downsampling to reduce file size and redundancy
        print("Reduciendo la densidad de la nube de puntos...")
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.01)
        
        # Statistical outlier removal to clean the point cloud
        print("Eliminando puntos atípicos...")
        combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Save the combined point cloud to PLY
        ply_filename = f"complete_object_pointcloud_{time.strftime('%Y%m%d_%H%M%S')}.ply"
        print(f"Guardando nube de puntos en {ply_filename}...")
        o3d.io.write_point_cloud(ply_filename, combined_pcd)
        print(f"Nube de puntos guardada exitosamente en {ply_filename}")
        
        # Extract points and colors for CSV
        points = np.asarray(combined_pcd.points)
        colors = np.asarray(combined_pcd.colors)
        
        # Create DataFrame and save as CSV
        if len(colors) > 0:  # If colors are available
            df = pd.DataFrame({
                'x': points[:, 0],
                'y': points[:, 1],
                'z': points[:, 2],
                'r': colors[:, 0] * 255,  # Convertir a rango 0-255
                'g': colors[:, 1] * 255,
                'b': colors[:, 2] * 255
            })
        else:  # If colors are not available
            df = pd.DataFrame({
                'x': points[:, 0],
                'y': points[:, 1],
                'z': points[:, 2]
            })
        
        # Save to CSV
        csv_filename = f"complete_object_pointcloud_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Nube de puntos guardada en formato CSV: {csv_filename}")
        
        # Visualize the result
        print("Visualizando nube de puntos...")
        o3d.visualization.draw_geometries([combined_pcd])
        
    finally:
        pipeline.stop()
        print("Pipeline detenido")

if __name__ == "__main__":
    capture_point_cloud()
