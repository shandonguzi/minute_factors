'''This module provides database information'''

import urllib.parse

from sqlalchemy import create_engine


USER0 = 'root'
PASSWD0 = urllib.parse.quote_plus("xxx")

HOST0 = '10.8.3.37'

PORT0 = 33308
PORT1 = 33309

DATABASE0 = 'csmar'
DATABASE3 = 'joinquant'

level0_csmar = create_engine(f'mysql://{USER0}:{PASSWD0}@{HOST0}:{PORT0}/{DATABASE0}')
level0_joinquant = create_engine(f'mysql://{USER0}:{PASSWD0}@{HOST0}:{PORT0}/{DATABASE3}')
level1_csmar = create_engine(f'mysql://{USER0}:{PASSWD0}@{HOST0}:{PORT1}/{DATABASE0}')