import numpy as np
import pandas as pd


def import_data() -> pd.DataFrame:
    fedfunds_rate = pd.read_csv('./data/FEDFUNDS.csv')
    mortgage_rate = pd.read_csv('./data/MORTGAGE30US.csv')
    median_sales_price = pd.read_csv('./data/MSPUS.csv')
    rental_vacancy = pd.read_csv('./data/RRVRUSQ156N.csv')
    lumber_index = pd.read_csv('./data/WPU081.csv')

    # set date
    fedfunds_rate['DATE'] = pd.to_datetime(fedfunds_rate['DATE'])
    mortgage_rate['DATE'] = pd.to_datetime(mortgage_rate['DATE'])
    median_sales_price['DATE'] = pd.to_datetime(median_sales_price['DATE'])
    rental_vacancy['DATE'] = pd.to_datetime(rental_vacancy['DATE'])
    lumber_index['DATE'] = pd.to_datetime(lumber_index['DATE'])

    # resample to quarterly
    fedfunds_rate = fedfunds_rate.set_index("DATE").resample("Q").mean().reset_index()
    mortgage_rate = mortgage_rate.set_index("DATE").resample("Q").mean().reset_index()
    median_sales_price = median_sales_price.set_index("DATE").resample("Q").mean().reset_index()
    rental_vacancy = rental_vacancy.set_index("DATE").resample("Q").mean().reset_index()
    lumber_index = lumber_index.set_index("DATE").resample("Q").mean().reset_index()

    # merge
    df = mortgage_rate.merge(median_sales_price, on="DATE").merge(fedfunds_rate, on="DATE").merge(rental_vacancy, on="DATE").merge(lumber_index, on="DATE")

    df = df.set_index('DATE')

    df['MSPUS'] = np.log(df['MSPUS'])
    df['WPU081'] = np.log(df['WPU081'])

    df_s1 = df.shift(1).rename(
        columns={'MORTGAGE30US': 'MORTGAGE30US_lag1', 'MSPUS': 'MSPUS_lag1', 'FEDFUNDS': 'FEDFUNDS_lag1',
                 'RRVRUSQ156N': 'RRVRUSQ156N_lag1', 'WPU081': 'WPU081_lag1'})

    df_s2 = df.shift(2).rename(
        columns={'MORTGAGE30US': 'MORTGAGE30US_lag2', 'MSPUS': 'MSPUS_lag2', 'FEDFUNDS': 'FEDFUNDS_lag2',
                 'RRVRUSQ156N': 'RRVRUSQ156N_lag2', 'WPU081': 'WPU081_lag2'})

    dfi = pd.concat([df, df_s1, df_s2], axis=1).dropna()

    return dfi


def merge_counterfactual_empirical_data(df_cf, df_emp):
    counterfactual = "Alternate"
    empirical = "Reality"

    dff = pd.concat([df_cf, df_emp], axis=1).rename(columns={"FEDFUNDS": f"Federal Rate ({counterfactual})",
                                                     "FEDFUNDS_EMP": f"Federal Rate ({empirical})",
                                                     "MORTGAGE30US": f"Mortgage Rate ({counterfactual})",
                                                     "MORTGAGE30US_EMP": f"Mortgage Rate ({empirical})",
                                                     "SALES": f"Sales Price ({counterfactual})",
                                                     "SALES_EMP": f"Sales Price ({empirical})",
                                                     "RENTAL": f"Rental Vacancies ({counterfactual})",
                                                     "RENTAL_EMP": f"Rental Vacancies ({empirical})",
                                                     "LUMBER": f"Lumber Index ({counterfactual})",
                                                     "LUMBER_EMP": f"Lumber Index ({empirical})"
                                                     })
    return dff
