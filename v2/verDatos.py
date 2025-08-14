def get_part_data(part_name):
    """Obtiene metadatos y datos 3D de una pieza"""
    # Obtener metadatos de SQLite
    conn = sqlite3.connect('auto_parts.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM parts WHERE name = ?", (part_name,))
    metadata = cursor.fetchone()
    conn.close()
    
    # Obtener datos 3D de Parquet
    df = pd.read_parquet(metadata[5])  # 5 = file_path
    
    # Convertir a nube de puntos
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[['x','y','z']].values)
    pcd.colors = o3d.utility.Vector3dVector(df[['r','g','b']].values / 255)
    
    return {
        'metadata': metadata,
        'point_cloud': pcd,
        'dataframe': df
    }

# Ejemplo de uso:
part_data = get_part_data('ignition_switch')
o3d.visualization.draw_geometries([part_data['point_cloud']])
