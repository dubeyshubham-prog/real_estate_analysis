# =====================================================
# REQUIRED LIBRARIES
# =====================================================
import pandas as pd

# =====================================================
# PREPROCESSING LEVEL 2 CLASS
# ====================================================
class PreprocessorLevel2:
    def __init__(self):
        self.sector_mapping = {
            'dharam colony': 'sector 12',
            'krishna colony': 'sector 7',
            'suncity': 'sector 54',
            'prem nagar': 'sector 13',
            'mg road': 'sector 28',
            'gandhi nagar': 'sector 28',
            'laxmi garden': 'sector 11',
            'shakti nagar': 'sector 11',

            'baldev nagar': 'sector 7',
            'shivpuri': 'sector 7',
            'garhi harsaru': 'sector 17',
            'imt manesar': 'manesar',
            'adarsh nagar': 'sector 12',
            'shivaji nagar': 'sector 11',
            'bhim nagar': 'sector 6',
            'madanpuri': 'sector 7',

            'saraswati vihar': 'sector 28',
            'arjun nagar': 'sector 8',
            'ravi nagar': 'sector 9',
            'vishnu garden': 'sector 105',
            'bhondsi': 'sector 11',
            'surya vihar': 'sector 21',
            'devilal colony': 'sector 9',
            'valley view estate': 'gwal pahari',

            'mehrauli  road': 'sector 14',
            'jyoti park': 'sector 7',
            'ansal plaza': 'sector 23',
            'dayanand colony': 'sector 6',
            'sushant lok phase 2': 'sector 55',
            'chakkarpur': 'sector 28',
            'greenwood city': 'sector 45',
            'subhash nagar': 'sector 12',

            'sohna road road': 'sohna road',
            'malibu town': 'sector 47',
            'surat nagar 1': 'sector 104',
            'new colony': 'sector 7',
            'mianwali colony': 'sector 12',
            'jacobpura': 'sector 12',
            'rajiv nagar': 'sector 13',
            'ashok vihar': 'sector 3',

            'dlf phase 1': 'sector 26',
            'nirvana country': 'sector 50',
            'palam vihar': 'sector 2',
            'dlf phase 2': 'sector 25',
            'sushant lok phase 1': 'sector 43',
            'laxman vihar': 'sector 4',
            'dlf phase 4': 'sector 28',
            'dlf phase 3': 'sector 24',

            'sushant lok phase 3': 'sector 57',
            'dlf phase 5': 'sector 43',
            'rajendra park': 'sector 105',
            'uppals southend': 'sector 49',
            'sohna': 'sohna road',
            'ashok vihar phase 3 extension': 'sector 5',
            'south city 1': 'sector 41',
            'ashok vihar phase 2': 'sector 5',

            'sector 95a': 'sector 95',
            'sector 23a': 'sector 23',
            'sector 12a': 'sector 12',
            'sector 3a': 'sector 3',
            'sector 110 a': 'sector 110',
            'patel nagar': 'sector 15',
            'a block sector 43': 'sector 43',
            'maruti kunj': 'sector 12',
            'b block sector 43': 'sector 43',

            'sector-33 sohna road': 'sector 33',
            'sector 1 manesar': 'manesar',
            'sector 4 phase 2': 'sector 4',
            'sector 1a manesar': 'manesar',
            'c block sector 43': 'sector 43',
            'sector 89 a': 'sector 89',
            'sector 2 extension': 'sector 2',
            'sector 36 sohna road': 'sector 36'
        }

    # =====================================================
    # CREATE SECTOR COLUMN
    # =====================================================
    def create_sector_column(self, df):

        df = df.copy()

        df.insert(
            loc=3,
            column='sector',
            value=df['property_name']
            .str.split('in')
            .str.get(1)
            .str.replace('Gurgaon', '')
            .str.strip()
            .str.lower()
        )

        return df

    # =====================================================
    # CLEAN SECTOR COLUMN
    # =====================================================
    def clean_sector(self, df):

        df = df.copy()

        df['sector'] = df['sector'].replace(self.sector_mapping)

        return df

    # =====================================================
    # REMOVE RARE SECTORS
    # =====================================================
    def remove_rare_sectors(self, df):

        df = df.copy()

        valid_sectors = (
            df['sector']
            .value_counts()[lambda x: x >= 3]
            .index
        )

        return df[df['sector'].isin(valid_sectors)]

    # =====================================================
    # MANUAL FIXES
    # =====================================================
    def apply_manual_fixes(self, df):

        df = df.copy()

        fixes = {
            955: 'sector 37',
            2800: 'sector 92',
            2838: 'sector 90',
            2857: 'sector 76'
        }

        for idx, sector in fixes.items():
            if idx in df.index:
                df.loc[idx, 'sector'] = sector

        multi_fix = [311, 1072, 1486, 3040, 3875]

        existing = [x for x in multi_fix if x in df.index]

        if existing:
            df.loc[existing, 'sector'] = 'sector 110'

        return df

    # =====================================================
    # DROP UNUSED COLUMNS
    # =====================================================
    def drop_columns(self, df):

        df = df.copy()

        cols = [
            'property_name',
            'description',
            'rating'
        ]

        return df.drop(columns=cols)

    # =====================================================
    # COMPLETE PIPELINE
    # =====================================================
    def run_pipeline(self, df):

        df = self.create_sector_column(df)
        df = self.clean_sector(df)
        df = self.remove_rare_sectors(df)
        df = self.apply_manual_fixes(df)
        df = self.drop_columns(df)

        return df