import dolphindb as ddb
import dolphindb.settings as keys
import numpy as np
import pandas as pd
from utils.datepair import generate_monthly_date_pairs_corrected


def to_dolphindb_all(df, dbPath, tableName):

    df['date'] = pd.to_datetime(df['date'])

    s = ddb.session(protocol=keys.PROTOCOL_DDB)
    s.connect("10.8.3.37", 8848, "admin", "123456") 

    dates = np.array(pd.date_range(start=df['date'].min().strftime('%Y%m%d'), end=df['date'].max().strftime('%Y%m%d')), dtype="datetime64[D]")
    # if s.existsDatabase(dbPath): s.dropDatabase(dbPath)
    
    db = s.database(dbName='tsinghua', partitionType=keys.VALUE, partitions=dates, dbPath=dbPath)

    for i in range(0, len(df), 1000000):
        if i == 0:
            t = s.table(data=df.iloc[i: i+1000000])
            db.createPartitionedTable(table=t, tableName=tableName, partitionColumns='date').append(t)
        else:
            s.run("tableInsert{{loadTable('{db}', `{tb})}}".format(db=dbPath,tb=tableName), df.iloc[i: i+1000000])
    
    s.close()
    
    print(f"[+] finish load {tableName} to dolphindb")


def to_dolphindb_update(df, dbPath, tableName):

    df['date'] = pd.to_datetime(df['date'])

    s = ddb.session(protocol=keys.PROTOCOL_DDB)
    s.connect("10.8.3.37", 8848, "admin", "123456")
    
    for i in range(0, len(df), 1000000):
        s.run("tableInsert{{loadTable('{db}', `{tb})}}".format(db=dbPath,tb=tableName), df.iloc[i: i+1000000])
    
    s.close()

    print(f"[+] finish update {tableName} to dolphindb")


def get_data_from_dolphindb(dbPath, tableName, start_date, end_date, columns='*'):

    s = ddb.session(protocol=keys.PROTOCOL_DDB)
    s.connect("10.8.3.37", 8848, "admin", "123456")

    date_pairs = generate_monthly_date_pairs_corrected(start_date, end_date)
    result = pd.DataFrame()

    for start, end in date_pairs:
        params = {'dt_s': pd.to_datetime(start).strftime("%Y.%m.%d"),
                'dt_e': pd.to_datetime(end).strftime("%Y.%m.%d")}
        
        query = f"select {columns} from {tableName} where date >= {params['dt_s']} and date <= {params['dt_e']}"
        re = s.loadTableBySQL(tableName=tableName, dbPath=dbPath, sql=query).toDF()

        result = pd.concat([result, re], axis=0)
    
    s.close()
    print(f"[+] finish get {tableName} from dolphindb")

    return result