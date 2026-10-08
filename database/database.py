"""
Database helper functions and connection manager for SQLite
"""

import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH
from database.schema import CREATE_TABLES_SQL

def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns a connection to the SQLite database with Row factory enabled."""
    path = db_path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Optional[Path] = None) -> None:
    """Initializes tables in the SQLite database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.executescript(CREATE_TABLES_SQL)
    conn.commit()
    conn.close()

def query_df(query: str, params: tuple = (), db_path: Optional[Path] = None) -> pd.DataFrame:
    """Executes a SQL query and returns the results as a Pandas DataFrame."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()

def execute_sql(sql: str, params: tuple = (), db_path: Optional[Path] = None) -> int:
    """Executes a single SQL statement and returns rowcount."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()

def execute_many(sql: str, param_list: List[tuple], db_path: Optional[Path] = None) -> int:
    """Executes a parameterized SQL statement on a list of tuples."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.executemany(sql, param_list)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
