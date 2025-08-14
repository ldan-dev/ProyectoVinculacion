import pandas as pd
import sqlite3
import pyarrow.parquet as pq

def add_to_database(ply_path, metadata):
    """Añade una pieza a la base de datos"""
    # Leer PLY y convertir a DataFrame
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    
    df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'r': colors[:, 0].astype(int),
        'g': colors[:, 1].astype(int),
        'b': colors[:, 2].astype(int)
    })
    
    # Guardar en Parquet (datos 3D)
    parquet_path = f"database/{metadata['name']}.parquet"
    df.to_parquet(parquet_path, compression='gzip')
    
    # Guardar metadatos en SQLite
    conn = sqlite3.connect('auto_parts.db')
    cursor = conn.cursor()
    
    # Crear tabla si no existe
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS parts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT,
        date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        point_count INTEGER,
        file_path TEXT
    )
    ''')
    
    # Insertar metadatos
    cursor.execute('''
    INSERT INTO parts (name, category, point_count, file_path)
    VALUES (?, ?, ?, ?)
    ''', (
        metadata['name'],
        metadata['category'],
        len(points),
        parquet_path
    ))
    
    conn.commit()
    conn.close()
    print(f"Pieza {metadata['name']} añadida a la base de datos")

# Ejemplo de uso:
metadata = {
    'name': 'ignition_switch',
    'category': 'electronic',
    'description': 'Switch de encendido modelo 2023'
}

add_to_database("switch_completo.ply", metadata)
