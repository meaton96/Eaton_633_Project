import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from typing import List




class CleaningPipeline(BaseEstimator, TransformerMixin):
    """
    Represents an sklearn compatible pipeline to prepocess the dataset.
    Handles Categorical binning, one hot encoding, log transformations, and interaction term creation
    """

    X: pd.DataFrame


    def __init__(self, 
                 create_interactions=True, 
                 log_transform_cols:None | np.ndarray=None,
                 one_hot_ordinal:bool = True
                 ) -> None:
        self.create_interactions = create_interactions
        self.log_transform_cols = log_transform_cols
        self.one_hot_ordinal = one_hot_ordinal

    # wrapper for all steps
    def run_pipe(self):
        self.bin_categories()
        self.drop_previous_category_cols()
        self.one_hot_cats()
        self.handle_drug_cols()

        if self.create_interactions:
            self.create_numerical_interactions()
            self.add_dummy_interactions()

        if self.log_transform_cols is not None and len(self.log_transform_cols) > 0:
            self.log_transform_continous_features()

        if not self.one_hot_ordinal:
            # transform binned ordinal categories into numeric encodings for tree models
            mapping = {
                'none': 0,
                'one': 1,
                'few': 3,
                'several': 6,
                'frequent': 10
            }

            ordinal_cols = ['v_outpatient_group', 'v_emergency_group']
            
            for col in ordinal_cols:
                num_col = f"{col}_num"
                self.X[num_col] = self.X[col].map(mapping).astype('Int64')

            self.X.drop(columns=ordinal_cols, inplace=True)
            

    # fit to supplied data, saves data as class field for manipulation
    def fit(self, X, y=None):
        X = self._ensure_dataframe(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.X = X.copy()

        self.run_pipe() # transform

   
        self._ref_dummies_ = self._choose_ref_dummies(self.X)

       
        self.X = self.X.drop(columns=list(self._ref_dummies_.values()), errors="ignore")

       
        self._safe_drop_non_features()

        # Freeze final schema
        self.feature_names_out_ = np.array(self.X.columns, dtype=object)
        self._out_dtypes_ = self.X.dtypes.to_dict()

        return self
    

    def log_transform_continous_features(self):
        self.X[self.log_transform_cols] = np.log1p(self.X[self.log_transform_cols])

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ["feature_names_out_", "_out_dtypes_"])
        Xw = self._ensure_dataframe(X, like=self.feature_names_in_).copy()
        X_prev = getattr(self, "X", None)
        try:
            self.X = Xw

            self.run_pipe()

            # Drop the same refs chosen at fit, if they exist in this X
            if hasattr(self, "_ref_dummies_"):
                self.X = self.X.drop(columns=list(self._ref_dummies_.values()), errors="ignore")

            self._safe_drop_non_features()

            Xw = self.X
        finally:
            if X_prev is not None:
                self.X = X_prev

        # align to training schema
        for col in self.feature_names_out_:
            if col not in Xw.columns:
                Xw[col] = 0
        Xw = Xw.loc[:, self.feature_names_out_]

        for c, dt in self._out_dtypes_.items():
            if c in Xw.columns:
                try: Xw[c] = Xw[c].astype(dt, copy=False)
                except: pass

        return Xw
    
    def _safe_drop_non_features(self):
        # Raw strings we never want in model space
        base_drop = ['diabetesMed', 'change', 'patient_nbr', 'encounter_id']

        drug_cols = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]

        # Drop drug strings and all flags EXCEPT keep metformin_flag and insulin_flag
        drop_flags = [f"{c}_flag" for c in drug_cols if c not in ('metformin', 'insulin')]

        to_drop = [c for c in (base_drop + drug_cols + drop_flags) if c in self.X.columns]
        if to_drop:
            self.X = self.X.drop(columns=to_drop, errors="ignore")


    # sklearn api
    def get_feature_names_out(self, input_features=None):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    
    def _ensure_dataframe(self, X, like: np.ndarray | None = None) -> pd.DataFrame:
        """
        Accept pandas or numpy; if numpy, wrap in a DataFrame with either the
        original training column names (when available) or generic names.
        """
        if isinstance(X, pd.DataFrame):
            return X
        X_arr = check_array(X, accept_sparse=False, dtype=None, force_all_finite="allow-nan")
        if like is not None:
            cols = list(like)
        else:
            cols = [f"x{i}" for i in range(X_arr.shape[1])]
        return pd.DataFrame(X_arr, columns=cols)

    # creates interaction terms
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


    def _choose_ref_dummies(self, X: pd.DataFrame) -> dict[str, str]:
        # Map each dummy family to a regex
        families = {
            'diag1_group_':       r'^diag1_group_',
            'diag2_group_':       r'^diag2_group_',
            'diag3_group_':       r'^diag3_group_',
            'admission_source_':  r'^admission_source_',
            'discharge_loc_':     r'^discharge_loc_',
            'specialty_cat_':     r'^specialty_cat_',
            'race_cat_':          r'^race_cat_',
            'age_group_':         r'^age_group_',
            'gender_':            r'^gender_',
            'a1c_group_':         r'^a1c_group_',
            'glucose_group_':     r'^glucose_group_',
            'admit_type_group_':  r'^admit_type_group_',
        }

        if self.one_hot_ordinal:
            families['v_outpatient_group'] = r'^v_outpatient_group'
            families['v_emergency_group'] = r'^v_emergency_group'

        refs = {}
        for fam, pat in families.items():
            cols = X.filter(regex=pat).columns
            if len(cols) >= 2:
                # Choose the most frequent level as reference (sum of one-hot = count)
                counts = X[cols].sum(axis=0)
                ref_col = counts.idxmax()
                refs[fam] = ref_col
        return refs

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
        columns = [ 'diag1_group', 
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
                    'admit_type_group',
                ]
        
        if self.one_hot_ordinal:
            columns.append('v_outpatient_group')
            columns.append('v_emergency_group')


        

        # one hot categorical columns
        self.X = pd.get_dummies(self.X, columns=columns, dtype=int)
        
        self.X = self.X.rename(columns={'gender_Female' : 'gender_female', 'gender_Male' : 'gender_male'})

    def drop_previous_category_cols(self):
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
                'admission_type_id',
                'number_outpatient',
                'number_emergency'
                ]
        

        self.X = self.X.drop(columns=columns)

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
                raise ValueError('Inpatient-fransferred patient detected')
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
                return 'under_30'
            # grabs (30-40), (40-50), (50-60)
            if ('40' in age or '50' in age):
                return '30-60'
            return 'over_60'
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
        
        # bin number outpatient
        def bin_procedure_cols(num: int) -> str:
            if num == 0:
                return 'none'
            if num == 1:
                return 'one'
            if num in [2,3]:
                return 'few'
            if num < 8:
                return 'several'
            return 'frequent'
        


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
        self.X['v_outpatient_group'] = self.X['number_outpatient'].apply(bin_procedure_cols)
        self.X['v_emergency_group'] = self.X['number_emergency'].apply(bin_procedure_cols)
        #self.X['v_inpatient_group'] = self.X['number_inpatient'].apply(bin_procedure_cols)
    


    





