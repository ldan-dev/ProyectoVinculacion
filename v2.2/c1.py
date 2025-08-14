"""
Captura nubes de puntos con una sola cámara Intel RealSense (RGB-D),
filtra basura (puntos lejanos/aislados), y guarda cada captura en CSV y PLY (XYZ+RGB).
Incluye pausas entre capturas para mover a mano el objeto.

Requisitos:
    pip install pyrealsense2 open3d numpy

Sugerencias de uso:
- Usa cartulina lisa (idealmente azul/verde opaco) como fondo para reducir ruido visual.
- Mantén la pieza a una distancia estable (p. ej., 0.30–0.70 m). Ajusta los umbrales abajo.
- Evita brillos (cromados) y superficies muy negras; inclina ligeramente la pieza si hay huecos.
- Ilumina de forma uniforme (luz difusa).

Salida por captura:
    - <output_dir>/<object_name>_<nn>.ply    (XYZ+RGB)
    - <output_dir>/<object_name>_<nn>.csv    (x,y,z,r,g,b)
Al final: visualiza cada PLY capturado.

Autor: tú :)
"""

import os
import time
import csv
from pathlib import Path
import numpy as np
import pyrealsense2 as rs
import open3d as o3d


# =========================
# Parámetros editables
# =========================
object_name = "switch"              # Nombre de la pieza u objeto (se usa en el nombre de archivos)
output_dir = "./captures"           # Carpeta donde se guardarán las capturas
captures_per_object = 10            # Número de capturas que harás por objeto
pause_seconds = 2.0                 # Pausa entre capturas (para mover el objeto a mano)

# Resolución/FPS (válidos para la mayoría de D4xx). Si te va lento, baja a 640x480.
color_width, color_height, color_fps = 640, 480, 30
depth_width, depth_height, depth_fps = 640, 480, 30

# Filtros por distancia (metros) para quitar “basura” muy lejana/cercana
z_min_m = 0.30                      # distancia mínima
z_max_m = 0.70                      # distancia máxima (ajústalo a tu montaje)
max_xy_radius_m = None              # opcional: recortar por radio lateral sqrt(x^2 + y^2); None para desactivar (p. ej. 0.30)

# Limpieza de outliers (Open3D) para quitar puntos aislados
sor_nb_neighbors = 20               # vecinos para Statistical Outlier Removal
sor_std_ratio = 2.0                 # cuanto menor, más agresivo

# Warmup: descarta frames iniciales (autoexposición/autoajuste)
warmup_frames = 30

# Activar filtros de post-proceso de RealSense (suavizan/llenan agujeros)
use_realsense_filters = True
decimation_magnitude = 2            # 2, 3... reduce resolución de profundidad (menos ruido)
spatial_alpha = 0.5                 # 0–1
spatial_delta = 20                  # 1–50
spatial_magnitude = 2               # 1–5
temporal_alpha = 0.4                # 0–1
temporal_delta = 20                 # 1–100
hole_filling_mode = 1               # 0:desact, 1:nearest, 2:farther


# =========================
# Utilidades
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def setup_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    # Color en BGR8 (más universal); luego convertimos a RGB
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)
    profile = pipeline.start(config)

    # Alinearemos la profundidad al color
    align = rs.align(rs.stream.color)

    # Filtros de post-proceso
    filters = {}
    if use_realsense_filters:
        filters["decimation"] = rs.decimation_filter()
        filters["decimation"].set_option(rs.option.filter_magnitude, decimation_magnitude)

        filters["spatial"] = rs.spatial_filter()
        filters["spatial"].set_option(rs.option.filter_magnitude, spatial_magnitude)
        filters["spatial"].set_option(rs.option.filter_smooth_alpha, spatial_alpha)
        filters["spatial"].set_option(rs.option.filter_smooth_delta, spatial_delta)

        filters["temporal"] = rs.temporal_filter()
        filters["temporal"].set_option(rs.option.filter_smooth_alpha, temporal_alpha)
        filters["temporal"].set_option(rs.option.filter_smooth_delta, temporal_delta)

        filters["hole_filling"] = rs.hole_filling_filter(hole_filling_mode)

    return pipeline, profile, align, filters


def get_aligned_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    return color_frame, depth_frame


def apply_rs_filters(depth_frame, filters):
    if not use_realsense_filters or not filters:
        return depth_frame
    f = depth_frame
    f = filters["decimation"].process(f)
    f = filters["spatial"].process(f)
    f = filters["temporal"].process(f)
    f = filters["hole_filling"].process(f)
    return f


def rs_pointcloud_with_color(depth_frame, color_frame):
    """
    Calcula nube de puntos (XYZ) y toma los colores RGB mapeando las texturas del frame de color.
    Retorna:
        xyz: np.ndarray [N,3] en metros
        rgb: np.ndarray [N,3] en uint8 (0..255)
    """
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)

    # XYZ
    vtx = np.asanyarray(points.get_vertices())  # array de rs.vertex
    xyz = np.array(vtx).view(np.float32).reshape(-1, 3)

    # UV y extracción de color
    tex = np.asanyarray(points.get_texture_coordinates())
    uv = np.array(tex).view(np.float32).reshape(-1, 2)

    color_image_bgr = np.asanyarray(color_frame.get_data())  # H x W x 3 (BGR)
    h, w, _ = color_image_bgr.shape
    # UV en [0,1] -> pixel coords
    u = np.clip((uv[:, 0] * w).astype(np.int32), 0, w - 1)
    v = np.clip((uv[:, 1] * h).astype(np.int32), 0, h - 1)
    rgb = color_image_bgr[v, u, ::-1].copy()  # conv BGR->RGB

    return xyz, rgb


def filter_by_distance_and_radius(xyz, rgb, z_min, z_max, max_xy_radius=None):
    """
    Filtra por rango de profundidad y opcionalmente por radio lateral (en metros).
    Quita puntos con z <= 0 (inválidos).
    """
    z = xyz[:, 2]
    mask = (z > 0) & (z >= z_min) & (z <= z_max)
    if max_xy_radius is not None:
        r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        mask = mask & (r <= max_xy_radius)
    return xyz[mask], rgb[mask]


def remove_outliers_o3d(xyz, rgb, nb_neighbors=20, std_ratio=2.0):
    """
    Usa Open3D Statistical Outlier Removal para quitar puntos aislados.
    Retorna:
        xyz_f, rgb_f, pcd_f (Open3D PointCloud con colores en [0,1])
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    colors = (rgb.astype(np.float32) / 255.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd_f, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    ind = np.array(ind, dtype=np.int64)
    xyz_f = np.asarray(pcd.points)[ind]
    rgb_f = (np.asarray(pcd.colors)[ind] * 255.0).astype(np.uint8)
    return xyz_f, rgb_f, pcd_f


def save_csv(csv_path, xyz, rgb):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "r", "g", "b"])
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", int(r), int(g), int(b)])


def save_ply(ply_path, xyz, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((rgb.astype(np.float32) / 255.0))
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False, print_progress=False)


def visualize_sequential(ply_paths):
    for ply in ply_paths:
        print(f"Visualizando: {ply}")
        pcd = o3d.io.read_point_cloud(str(ply))
        o3d.visualization.draw_geometries([pcd])  # cierra la ventana para pasar al siguiente


def main():
    out_dir = Path(output_dir) / object_name
    ensure_dir(out_dir)

    pipeline, profile, align, filters = setup_pipeline()

    try:
        print("Calentando cámara (autoexposición/autoajuste)...")
        for _ in range(warmup_frames):
            _ = get_aligned_frames(pipeline, align)

        saved_plys = []

        for i in range(captures_per_object):
            # Pausa para que muevas el objeto a mano
            print(f"\nPrepárate para la captura {i+1}/{captures_per_object}...")
            for t in range(int(pause_seconds), 0, -1):
                print(f"Capturando en {t} s...")
                time.sleep(1.0)
            if pause_seconds - int(pause_seconds) > 1e-3:
                time.sleep(pause_seconds - int(pause_seconds))

            # Frame alineado
            color_frame, depth_frame = get_aligned_frames(pipeline, align)
            if color_frame is None or depth_frame is None:
                print("No se obtuvo frame válido. Reintentando...")
                continue

            # Post-proceso de profundidad
            depth_frame = apply_rs_filters(depth_frame, filters)

            # Nube de puntos XYZ + RGB
            xyz, rgb = rs_pointcloud_with_color(depth_frame, color_frame)

            # Filtrado por distancia
            xyz, rgb = filter_by_distance_and_radius(xyz, rgb, z_min_m, z_max_m, max_xy_radius_m)

            # Quitar outliers aislados
            xyz_f, rgb_f, pcd_f = remove_outliers_o3d(xyz, rgb, nb_neighbors=sor_nb_neighbors, std_ratio=sor_std_ratio)

            n_before = xyz.shape[0]
            n_after = xyz_f.shape[0]
            print(f"Puntos antes: {n_before:,} | después de filtros: {n_after:,}")

            # Guardar
            base = f"{object_name}_{i+1:02d}"
            csv_path = out_dir / f"{base}.csv"
            ply_path = out_dir / f"{base}.ply"
            save_csv(csv_path, xyz_f, rgb_f)
            save_ply(ply_path, xyz_f, rgb_f)
            saved_plys.append(ply_path)
            print(f"Guardado:\n- {csv_path}\n- {ply_path}")

        # Visualización final
        if saved_plys:
            print("\nMostrando cada nube capturada (cierra la ventana para pasar a la siguiente)...")
            visualize_sequential(saved_plys)
        else:
            print("No se guardaron nubes.")

    finally:
        pipeline.stop()
        print("Transmisión detenida.")


if __name__ == "__main__":
    main()
