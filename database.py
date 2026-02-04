import pandas as pd
import numpy as np
import streamlit as st
import psycopg2
from psycopg2 import OperationalError

def get_db_connection(host, database, user, password, port=5432):
    """
    Create and return a PostgreSQL database connection using psycopg2.

    :param host: Database host
    :param database: Database name
    :param user: Database user
    :param password: User password
    :param port: Database port (default: 5432)
    :return: psycopg2 connection object
    :raises: OperationalError if connection fails
    """
    try:
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port,
        )
        return connection
    except OperationalError as e:
        raise OperationalError(f"Failed to connect to the database: {e}")


def fetch_het_data(connection=None):
    """
    Fetches HET data from the database.
    If connection is provided, it runs a SQL query.
    If fails or no connection, returns dummy data.
    """
    if connection:
        try:
            query = "SELECT * FROM public.het_measurements"
            df = pd.read_sql(query, connection)
            return df
        except Exception as e:
            st.warning(f"Failed to fetch data from DB, using dummy data: {e}")
            pass

    # Generating Dummy Data
    materials = ['HC340LA', 'CR4', 'DP600', 'DP800', 'DX51D']
    suppliers = ['Supplier A', 'Supplier B', 'Supplier C']
    clearances = ['12345', 'Keine Angabe', '54321']
    timestamps = ['1h', '24h', 'Keine Angabe']
    
    data = []
    for i in range(100):
        mat = np.random.choice(materials)
        sup = np.random.choice(suppliers)
        clr = np.random.choice(clearances)
        thick = np.random.uniform(0.5, 3.0)
        lab_nr = f"LAB-{i:05d}"
        
        # 20 HET measurements (%)
        # HET values typically range from 20% to 100% depending on material
        base_het = np.random.uniform(30, 80)
        measurements = np.random.normal(base_het, 5, 20).tolist()
        
        row = {
            'material': mat,
            'supplier': sup,
            'clearance': clr,
            'CoilNr': f"COIL-{np.random.randint(1000, 9999)}",
            'thick': thick,
            'date': '2025-01-20',
            'lab': 'Lab Name',
            'LabProt': lab_nr,
            'timeStampMeas': np.random.choice(timestamps),
            'comm': 'Sample Comment',
            'measdata_HET': measurements
        }
        # Major Strain = ln(1 + HEC/100)
        row['measdata_Strain'] = [np.log(1 + m/100) for m in measurements]
        data.append(row)
        
    return pd.DataFrame(data)
