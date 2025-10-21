import numpy as np
import pandas as pd
import re

class CleaningPipeline:

    X: pd.DataFrame



    def run(self, X ,*, create_interactions=True):
        self.X = X

        self.fix_dtypes()
        self.drop_bad_cols()
        self.bin_categories()
        self.drop_previous_category_cols()
        self.one_hot_cats()
        self.handle_drug_cols()
        self.drop_reundant_cols()

        if (create_interactions):
            self.create_numerical_interactions()
            self.add_dummy_interactions()

            

    def add_dummy_interactions(self):
        def add_dummy_interactions(df, left_cols, right_cols, prefix='i_'):
            out = df.copy()
            L = [c for c in left_cols  if c in out.columns]
            R = [c for c in right_cols if c in out.columns]
            count = 0
            for lc in L:
                for rc in R:
                    col_name = f'{prefix}{lc}__{rc}'
                    out[col_name] = out[lc] * out[rc]
                    count += 1
                    print(f'added col: {col_name}')

            print(f'added {count} interaction terms from')
            print(left_cols)
            print('x')
            print(right_cols)

            return out
        
        # setup column lists
        specialty_cols = self.X.filter(regex=r'^specialty_cat_').columns.to_list()
        age_cols = self.X.filter(regex=r'^age_group_').columns.to_list()
        diag_cols = self.X.filter(regex=r'^diag1_group_').columns.to_list()
        race_cols = self.X.filter(regex=r'^race_cat_').columns.to_list()
        discharge_cols = self.X.filter(regex=r'^discharge_loc_').columns.to_list()


        self.X = add_dummy_interactions(
            self.X,
            left_cols=specialty_cols,
            right_cols=age_cols
        )

        # discharge dispo x race
        self.X = add_dummy_interactions(
            self.X,
            left_cols= discharge_cols,
            right_cols=race_cols
        )

        # discharge x specialty

        self.X = add_dummy_interactions(
            self.X,
            left_cols=discharge_cols,
            right_cols=specialty_cols
        )



    def create_numerical_interactions(self):
        for col in self.X.filter(like='diag1_group_'):
            self.X[f'i_hosp_time__{col}'] = self.X[col] * self.X['time_in_hospital']
            
        # medical specialty of admitting physician x time in hospital
        for col in self.X.filter(like='specialty_cat_'):
            self.X[f'i_hosp_time__{col}'] = self.X[col] * self.X['time_in_hospital']
        
        # Discharge Disposition x time in hospital
        for col in self.X.filter(regex=r'^discharge_loc_'):
            self.X[f'i_hosp_time__{col}'] = self.X[col] * self.X['time_in_hospital']





    def drop_reundant_cols(self):

        # drop one col of one_hot cat for each category
        self.X = self.X.drop(
            columns=[
                'diag1_group_circulatory', 
                'specialty_cat_missing', 
                'age_group_30-60', # mid age group
                'race_cat_caucasian', 
                'gender_male', # most common base for medical studies
                'discharge_loc_home', 
                'admission_source_other',
                'a1c_group_no_test',
                'glucose_group_no_test',
                'admit_type_group_emergency'
                
            ]
        )


        # keep insulin and metformin columns (binary yes no prescribed)
        drug_cols = [
            'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        # Create the list of flag columns
        flag_cols = [f"{c}_flag" for c in drug_cols]

        self.X = self.X.drop(
            columns=[
                'diabetesMed',
                'change'
            ]
        )
        # drop drug columns
        self.X = self.X.drop(
            columns=drug_cols
        )
        self.X = self.X.drop(columns=flag_cols)





    def handle_drug_cols(self):
        # swap to binary flags
        self.X['diabetesMed_flag'] = (self.X['diabetesMed'].str.lower() == 'yes').astype(int)
        self.X['change_flag'] = (self.X['change'].str.lower() == 'ch').astype(int)

        drug_cols = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]

        for col in drug_cols:
            self.X[f'{col}_flag'] = (self.X[col].str.lower() != 'no').astype(int)

        # sum across medicine for amount of medicine prescribed
        self.X['num_drugs'] = self.X[[f'{c}_flag' for c in drug_cols]].sum(axis=1)

        


    def one_hot_cats(self):
        # one hot categorical columns
        self.X = pd.get_dummies(self.X, 
                            columns=[  'diag1_group', 
                                        'diag2_group', 
                                        'diag3_group', 
                                        'admission_source',
                                        'discharge_loc',
                                        'specialty_cat',
                                        'race_cat',
                                        'age_group',
                                        'gender',
                                        'a1c_group',
                                        'glucose_group',
                                        'admit_type_group'
                                    ],
                                    dtype=int)
        
        self.X = self.X.rename(columns={'gender_Female' : 'gender_female', 'gender_Male' : 'gender_male'})

    def drop_previous_category_cols(self):
        self.X = self.X.drop(
            columns=['diag_1', 
                    'diag_2', 
                    'diag_3', 
                    'admission_source_id',
                    'discharge_disposition_id',
                    'medical_specialty',
                    'race',
                    'age',
                    'A1Cresult',
                    'max_glu_serum',
                    'admission_type_id'
                    ])

    def bin_categories(self):
        # bin diagnoses into groups 
        _icd_num_re = re.compile(r'^(\d{3})(?:\.\d+)?$')   # e.g., '250.13' -> '250'

        def _parse_icd9(code):
            """
            Returns (prefix, num) where:
            - prefix is 'E', 'V', or '' for numeric codes
            - num is an integer 3-digit number if numeric; else None
            """
            if code is None or (isinstance(code, float) and np.isnan(code)):
                return '', None
            s = str(code).strip()
            if not s:
                return '', None
            first = s[0].upper()
            if first in ('E', 'V'):
                return first, None
            m = _icd_num_re.match(s)
            if m:
                return '', int(m.group(1))
            # Try float -> int of floor integer part
            try:
                return '', int(float(s))
            except Exception:
                return '', None

        def icd9_to_group(code):
            """
            Map a single ICD-9 code to the study's diagnosis group.
            """
            prefix, num = _parse_icd9(code)

            # External causes (E or V) 
            if prefix in ('E', 'V'):
                return 'other'

            if num is None:
                return 'other'

            # Special case: Diabetes 250.xx
            if 250 <= num <= 250:
                return 'diabetes'

            # Primary named groups
            if (390 <= num <= 459) or (num == 785):
                return 'circulatory'
            if (460 <= num <= 519) or (num == 786):
                return 'respiratory'
            if (520 <= num <= 579) or (num == 787):
                return 'digestive'
            if 800 <= num <= 999:
                return 'injury'
            if 710 <= num <= 739:
                return 'musculoskeletal'
            if (580 <= num <= 629) or (num == 788):
                return 'genitourinary'
            if 140 <= num <= 239:
                return 'neoplasms'

            return 'other'
        
        # Admission source -> 3 bins
        def bin_admit_source(id: int) -> str:
            if id == 7:
                return 'emergency'
            if id == 1 or id == 2:
                return 'refer'
            return 'other'
        
        # Medical Specialty of admitting physician -> 6 bins including missing
        def bin_medical_specialty(value: str) -> str:
            if pd.isna(value) or value in ("Missing", "Unknown", "PhysicianNotFound", "OutreachServices", "DCPTEAM"):
                return "missing"
            
            # normalize casing and spacing just in case
            val = str(value).strip().lower()
            
            # Internal Medicine
            if "internal" in val:
                return "internal_medicine"
            
            # Cardiology
            if "cardio" in val:
                return "cardiology"
            
            # Surgery (catch-all for surgical specialties)
            if "surg" in val or "orthopedic" in val or "urology" in val or "gyneco" in val or "neuro" in val or "vascular" in val or "thoracic" in val:
                return "surgery"
            
            # Family / General Practice
            if "family" in val or "general" in val or "gp" in val or "obstetric" in val or "pediatr" in val:
                return "pcp"
            
            # Everything else
            return "other"
        # Discharge Dispostion -> 

        def bin_discharge(id: int) -> str:
            if id in [1, 6, 8, 13]:
                return 'home'
            if id in [2, 3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30]:
                return 'transfer_inpatient'
            if id in [16, 17, 12]:
                return 'outpatient_followup'
            if id == 7:
                return 'left_ama'
            if id in [18, 25, 26]:
                return 'unknown'
            return 'unknown'
        
        # bin race -> Hispanic/Asian -> other since they have few values
        def bin_race(race: str) -> str:
            if (pd.isna(race)):
                return "other"
            race = race.strip().lower()
            if race == "caucasian":
                return "caucasian"
            if race == "africanamerican":
                return "african_american"
            return "other"
    

        # bin age -> 3 groups
        def bin_age(age: str) -> str:
            # grabs [0-10], [10-20], [20-30]
            if ('10' in age or '20' in age):
                return '<30'
            # grabs (30-40), (40-50), (50-60)
            if ('40' in age or '50' in age):
                return '30-60'
            return '>60'
        # bin a1c results
        def bin_a1c(val: str) -> str:
            if pd.isna(val):
                return 'no_test'
            if val in ('>7', '>8'):
                return 'high'
            if val.lower().startswith('norm'):
                return 'normal'
            return 'no_test'
        def bin_glucose(val: str) -> str:
            if pd.isna(val):
                return 'no_test'
            if val in ('>200', '>300'):
                return 'high'
            if val.lower().startswith('norm'):
                return 'normal'
            return 'no_test'
        
        # bin admission type id 
        def bin_admit_type(id: int) -> str:
            if id in [5, 6, 8]: # Not Available, NULL, Not mapped
                return 'na'
            if id in [1,7]: # Emergency, Trauma Center
                return 'emergency'
            if id == 2:
                return 'urgent'
            if id == 3:
                return 'elective'
            return 'other'
        


        self.X['diag1_group'] = self.X['diag_1'].apply(icd9_to_group)
        self.X['diag2_group'] = self.X['diag_2'].apply(icd9_to_group)
        self.X['diag3_group'] = self.X['diag_3'].apply(icd9_to_group)
        self.X['admission_source'] = self.X['admission_source_id'].apply(bin_admit_source)
        self.X['discharge_loc'] = self.X['discharge_disposition_id'].apply(bin_discharge)
        self.X['specialty_cat'] = self.X['medical_specialty'].apply(bin_medical_specialty)
        self.X['race_cat'] = self.X['race'].apply(bin_race)
        self.X['age_group'] = self.X['age'].apply(bin_age)
        self.X['a1c_group'] = self.X['A1Cresult'].apply(bin_a1c)
        self.X['glucose_group'] = self.X['max_glu_serum'].apply(bin_glucose)
        self.X['admit_type_group'] = self.X['admission_type_id'].apply(bin_admit_type)

    


    def fix_dtypes(self):
        # fix diag column types
        self.X['diag_1'] = self.X['diag_1'].astype('string')
        self.X['diag_2'] = self.X['diag_2'].astype('string')
        self.X['diag_3'] = self.X['diag_3'].astype('string')

        # fix specialty column type
        self.X['medical_specialty'] = self.X['medical_specialty'].astype('string')

        # fix race
        self.X['race'] = self.X['race'].astype('string')

        # fix age
        self.X['age'] = self.X['age'].astype('string')

        self.X['change'] = self.X['change'].astype('string')



    def drop_bad_cols(self):
        self.X = self.X.drop(columns=['weight', 'payer_code'])



