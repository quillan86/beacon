import numpy as np
import pandas as pd
import dowhy.gcm as gcm
from dowhy.gcm._noise import compute_noise_from_data
import networkx as nx
import joblib


def create_causal_model(dfi: pd.DataFrame) -> gcm.InvertibleStructuralCausalModel:
    """
    Create a causal model from the training data.
    :param dfi: - training data
    :return:
    """
    causal_graph = nx.DiGraph([('FEDFUNDS_lag2', 'FEDFUNDS'),
                               ('FEDFUNDS_lag1', 'FEDFUNDS'),
                               ('MSPUS_lag2', 'FEDFUNDS'),
                               ('MSPUS_lag1', 'FEDFUNDS'),
                               ('FEDFUNDS_lag2', 'MORTGAGE30US'),
                               ('FEDFUNDS_lag1', 'MORTGAGE30US'),
                               ('MORTGAGE30US_lag2', 'MORTGAGE30US'),
                               ('MORTGAGE30US_lag1', 'MORTGAGE30US'),
                               ('FEDFUNDS_lag2', 'WPU081'),
                               ('FEDFUNDS_lag1', 'WPU081'),
                               ('WPU081_lag2', 'WPU081'),
                               ('WPU081_lag1', 'WPU081'),
                               ('WPU081_lag2', 'MSPUS'),
                               ('WPU081_lag1', 'MSPUS'),
                               ('MSPUS_lag2', 'MSPUS'),
                               ('MSPUS_lag1', 'MSPUS'),
                               ('MORTGAGE30US_lag2', 'MSPUS'),
                               ('MORTGAGE30US_lag1', 'MSPUS'),
                               ('MORTGAGE30US_lag2', 'RRVRUSQ156N'),
                               ('MORTGAGE30US_lag1', 'RRVRUSQ156N'),
                               ('MSPUS_lag2', 'RRVRUSQ156N'),
                               ('MSPUS_lag1', 'RRVRUSQ156N'),
                               ('RRVRUSQ156N_lag2', 'RRVRUSQ156N'),
                               ('RRVRUSQ156N_lag1', 'RRVRUSQ156N')
                               ])

    causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)

    causal_model.set_causal_mechanism('FEDFUNDS_lag2', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('FEDFUNDS_lag1', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('MORTGAGE30US_lag2', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('MORTGAGE30US_lag1', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('MSPUS_lag2', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('MSPUS_lag1', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('WPU081_lag2', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('WPU081_lag1', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('RRVRUSQ156N_lag2', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('RRVRUSQ156N_lag1', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('FEDFUNDS', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
    causal_model.set_causal_mechanism('MORTGAGE30US', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
    causal_model.set_causal_mechanism('MSPUS', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
    causal_model.set_causal_mechanism('WPU081', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
    causal_model.set_causal_mechanism('RRVRUSQ156N', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

    gcm.fit(causal_model, dfi)
    joblib.dump(causal_model, './models/model.joblib')

    return causal_model


def load_causal_model():
    """
    Load the causal model from a joblib file.
    :return:
    """
    causal_model: gcm.InvertibleStructuralCausalModel = joblib.load('./models/model.joblib')
    return causal_model


def counterfactual_forecast(dfi: pd.DataFrame,
                            causal_model: gcm.InvertibleStructuralCausalModel,
                            min_i: int = 150,
                            forecast_i: int = 20,
                            var: str = "FEDFUNDS",
                            rate: float = 5.0):

    cf = []

    fedfunds_lag2 = dfi['FEDFUNDS_lag2'].iloc[min_i]
    fedfunds_lag1 = dfi['FEDFUNDS_lag1'].iloc[min_i]
    mortgage_lag2 = dfi['MORTGAGE30US_lag2'].iloc[min_i]
    mortgage_lag1 = dfi['MORTGAGE30US_lag1'].iloc[min_i]
    sales_lag2 = dfi['MSPUS_lag2'].iloc[min_i]
    sales_lag1 = dfi['MSPUS_lag1'].iloc[min_i]
    rental_lag2 = dfi['RRVRUSQ156N_lag2'].iloc[min_i]
    rental_lag1 = dfi['RRVRUSQ156N_lag1'].iloc[min_i]
    lumber_lag2 = dfi['WPU081_lag2'].iloc[min_i]
    lumber_lag1 = dfi['WPU081_lag1'].iloc[min_i]
    noise = compute_noise_from_data(causal_model, dfi)
    # forecast noise - set to zero
    noise = pd.concat([noise, pd.DataFrame([{x: 0 for x in noise.columns}] * len(noise))])

    # counterfactual + counterfactual future forecast
    for i in range(0, len(dfi) - min_i + forecast_i):
        cf_ = gcm.counterfactual_samples(
            causal_model,
            {
                f'{var}_lag2': lambda x: rate,
                f'{var}_lag1': lambda x: rate,
                f'{var}': lambda x: rate,
            },
            noise_data=pd.DataFrame([{'FEDFUNDS_lag2': fedfunds_lag2,
                                      'FEDFUNDS_lag1': fedfunds_lag1,
                                      'MORTGAGE30US_lag2': mortgage_lag2,
                                      'MORTGAGE30US_lag1': mortgage_lag1,
                                      'MSPUS_lag2': sales_lag2,
                                      'MSPUS_lag1': sales_lag1,
                                      'RRVRUSQ156N_lag2': rental_lag2,
                                      'RRVRUSQ156N_lag1': rental_lag1,
                                      'WPU081_lag2': lumber_lag2,
                                      'WPU081_lag1': lumber_lag1,
                                      'FEDFUNDS': noise['FEDFUNDS'].iloc[min_i + i],
                                      'MORTGAGE30US': noise['MORTGAGE30US'].iloc[min_i + i],
                                      'MSPUS': noise['MSPUS'].iloc[min_i + i],
                                      'RRVRUSQ156N': noise['RRVRUSQ156N'].iloc[min_i + i],
                                      'WPU081': noise['WPU081'].iloc[min_i + i]
                                      }])
        )
        fedfunds_lag1 = cf_['FEDFUNDS'][0]
        fedfunds_lag2 = cf_['FEDFUNDS_lag1'][0]
        mortgage_lag1 = cf_['MORTGAGE30US'][0]
        mortgage_lag2 = cf_['MORTGAGE30US_lag1'][0]
        sales_lag1 = cf_['MSPUS'][0]
        sales_lag2 = cf_['MSPUS_lag1'][0]
        rental_lag1 = cf_['RRVRUSQ156N'][0]
        rental_lag2 = cf_['RRVRUSQ156N_lag1'][0]
        lumber_lag1 = cf_['WPU081'][0]
        lumber_lag2 = cf_['WPU081_lag1'][0]
        cf.extend(cf_.to_dict(orient="records"))
    cf = pd.DataFrame(cf)

    fed_cf = pd.concat([dfi.reset_index().iloc[:min_i, 3], cf['FEDFUNDS']]).reset_index(drop=True)
    mortgage_cf = pd.concat([dfi.reset_index().iloc[:min_i, 1], cf['MORTGAGE30US']]).reset_index(drop=True)
    sales_cf = pd.concat([dfi.reset_index().iloc[:min_i, 2], cf['MSPUS']]).reset_index(drop=True)
    sales_cf = np.exp(sales_cf)
    rental_cf = pd.concat([dfi.reset_index().iloc[:min_i, 4], cf['RRVRUSQ156N']]).reset_index(drop=True)
    lumber_cf = pd.concat([dfi.reset_index().iloc[:min_i, 5], cf['WPU081']]).reset_index(drop=True)
    lumber_cf = np.exp(lumber_cf)

    df_cf = pd.DataFrame({'FEDFUNDS': fed_cf, 'MORTGAGE30US': mortgage_cf, 'SALES': sales_cf, 'RENTAL': rental_cf, 'LUMBER': lumber_cf})

    return df_cf


def empirical_forecast(dfi: pd.DataFrame,
                       causal_model: gcm.InvertibleStructuralCausalModel,
                       forecast_i: int = 20):

    min_i_forecast = len(dfi) - 1

    fedfunds_lag2 = dfi['FEDFUNDS_lag2'].iloc[min_i_forecast]
    fedfunds_lag1 = dfi['FEDFUNDS_lag1'].iloc[min_i_forecast]
    mortgage_lag2 = dfi['MORTGAGE30US_lag2'].iloc[min_i_forecast]
    mortgage_lag1 = dfi['MORTGAGE30US_lag1'].iloc[min_i_forecast]
    sales_lag2 = dfi['MSPUS_lag2'].iloc[min_i_forecast]
    sales_lag1 = dfi['MSPUS_lag1'].iloc[min_i_forecast]
    rental_lag2 = dfi['RRVRUSQ156N_lag2'].iloc[min_i_forecast]
    rental_lag1 = dfi['RRVRUSQ156N_lag1'].iloc[min_i_forecast]
    lumber_lag2 = dfi['WPU081_lag2'].iloc[min_i_forecast]
    lumber_lag1 = dfi['WPU081_lag1'].iloc[min_i_forecast]
    noise = compute_noise_from_data(causal_model, dfi)
    # forecast noise - set to zero
    noise = pd.concat([noise, pd.DataFrame([{x: 0 for x in noise.columns}] * len(noise))])

    forecast = []

    for i in range(0, len(dfi) - min_i_forecast + forecast_i):
        cf_ = gcm.counterfactual_samples(
            causal_model,
            {},
            noise_data=pd.DataFrame([{'FEDFUNDS_lag2': fedfunds_lag2,
                                      'FEDFUNDS_lag1': fedfunds_lag1,
                                      'MORTGAGE30US_lag2': mortgage_lag2,
                                      'MORTGAGE30US_lag1': mortgage_lag1,
                                      'MSPUS_lag2': sales_lag2,
                                      'MSPUS_lag1': sales_lag1,
                                      'RRVRUSQ156N_lag2': rental_lag2,
                                      'RRVRUSQ156N_lag1': rental_lag1,
                                      'WPU081_lag2': lumber_lag2,
                                      'WPU081_lag1': lumber_lag1,
                                      'FEDFUNDS': noise['FEDFUNDS'].iloc[min_i_forecast + i],
                                      'MORTGAGE30US': noise['MORTGAGE30US'].iloc[min_i_forecast + i],
                                      'MSPUS': noise['MSPUS'].iloc[min_i_forecast + i],
                                      'RRVRUSQ156N': noise['RRVRUSQ156N'].iloc[min_i_forecast + i],
                                      'WPU081': noise['WPU081'].iloc[min_i_forecast + i]
                                      }])
        )
        fedfunds_lag1 = cf_['FEDFUNDS'][0]
        fedfunds_lag2 = cf_['FEDFUNDS_lag1'][0]
        mortgage_lag1 = cf_['MORTGAGE30US'][0]
        mortgage_lag2 = cf_['MORTGAGE30US_lag1'][0]
        sales_lag1 = cf_['MSPUS'][0]
        sales_lag2 = cf_['MSPUS_lag1'][0]
        rental_lag1 = cf_['RRVRUSQ156N'][0]
        rental_lag2 = cf_['RRVRUSQ156N_lag1'][0]
        lumber_lag1 = cf_['WPU081'][0]
        lumber_lag2 = cf_['WPU081_lag1'][0]
        forecast.extend(cf_.to_dict(orient="records"))
    forecast = pd.DataFrame(forecast)

    fed_fc = pd.concat([dfi.reset_index().iloc[:min_i_forecast, 3], forecast['FEDFUNDS']]).reset_index(
        drop=True)
    mortgage_fc = pd.concat([dfi.reset_index().iloc[:min_i_forecast, 1], forecast['MORTGAGE30US']]).reset_index(
        drop=True)
    sales_fc = pd.concat([dfi.reset_index().iloc[:min_i_forecast, 2], forecast['MSPUS']]).reset_index(drop=True)
    sales_fc = np.exp(sales_fc)
    rental_fc = pd.concat([dfi.reset_index().iloc[:min_i_forecast, 4], forecast['RRVRUSQ156N']]).reset_index(drop=True)
    lumber_fc = pd.concat([dfi.reset_index().iloc[:min_i_forecast, 5], forecast['WPU081']]).reset_index(drop=True)
    lumber_fc = np.exp(lumber_fc)

    df_fc = pd.DataFrame({'FEDFUNDS_EMP': fed_fc, 'MORTGAGE30US_EMP': mortgage_fc, 'SALES_EMP': sales_fc, 'RENTAL_EMP': rental_fc, 'LUMBER_EMP': lumber_fc})
    return df_fc

