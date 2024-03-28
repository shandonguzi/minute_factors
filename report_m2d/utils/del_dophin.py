import dolphindb as ddb
import dolphindb.settings as keys
s = ddb.session(protocol=keys.PROTOCOL_DDB)
s.connect("10.8.3.37", 8848, "admin", "123456")

# factors returns prices
dbPath = "dfs://prices"
if s.existsDatabase(dbPath):
    s.dropDatabase(dbPath)
s.close()