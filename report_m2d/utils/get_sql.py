import re

import MySQLdb
import pandas as pd
from sqlalchemy import create_engine


def get_sql(con, SQL_statement, sort_on=None, index_cols=None, index=None, columns=None, values=None):
    if type(con) == str:
        con = create_engine(con)
    schema_name = con.url.database
    db = MySQLdb.connect(host=con.url.host, user=con.url.username, password=con.url.password, database=con.url.database, port=con.url.port, charset='utf8')
    cursor = db.cursor()
    
    if re.fullmatch(r'\w+', SQL_statement):
        table_name = SQL_statement
        SQL_statement = f'SELECT * FROM {table_name}'
    else:
        try:
            table_name = re.search('from (\w+)', SQL_statement, re.IGNORECASE).group(1)
        except AttributeError:
            raise ValueError('The table name could not be found in the SQL statement.')
    
    cursor.execute(f'''
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name="{table_name}" AND table_schema="{schema_name}"
        ORDER BY ORDINAL_POSITION
    ''')
    real_columns = list(map(lambda x: x[0], cursor.fetchall()))
    cursor.execute(SQL_statement)
    db.commit()
    data = pd.DataFrame(cursor.fetchall(), columns=real_columns)

    if sort_on is not None:
        data = data.sort_values(by=sort_on)

    if index_cols is not None:
        data = data.set_index(index_cols)
        
    if index is not None:
        data = data.pivot(index=index, columns=columns, values=values)
        
    if data.shape[1] == 1:
        data = data.iloc[:, 0]

    return data