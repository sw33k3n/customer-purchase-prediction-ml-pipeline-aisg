import sqlite3
import pandas as pd
from pathlib import Path

def load_data(db_path="data/online_shopping.db", table_name="online_shopping"):

    db_path = Path(db_path)

    if not db_path.exists():
        hint = (
            "Set a valid path via `./run.sh /path/to/your.db` "
            "or env var `ONLINE_SHOPPING_DB_PATH=/path/to/your.db`."
        )
        raise FileNotFoundError(
            f"Database file not found at: {db_path.resolve()}. {hint}"
        )

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Provided database path is not a file: {db_path.resolve()}"
        )

    conn = None

    try:
        conn = sqlite3.connect(db_path)

        # Check whether the requested table exists
        table_check_query = """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name = ?
        """
        table_result = pd.read_sql_query(
            table_check_query,
            conn,
            params=(table_name,)
        )

        if table_result.empty:
            available_tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                conn
            )["name"].tolist()

            raise ValueError(
                f"Table '{table_name}' not found in database. "
                f"Available tables: {available_tables}"
            )

        # Load the table
        query = f'SELECT * FROM "{table_name}"'
        df = pd.read_sql_query(query, conn)

        if df.empty:
            raise ValueError(f"Table '{table_name}' exists but contains no rows.")

        return df

    except sqlite3.Error as e:
        raise RuntimeError(
            f"SQLite error while reading '{db_path}': {e}"
        ) from e

    except pd.errors.DatabaseError as e:
        raise RuntimeError(
            f"Pandas failed to read from database '{db_path}': {e}"
        ) from e

    finally:
        if conn is not None:
            conn.close()
