import numpy as np
import streamlit as st
from st_cytoscape import cytoscape
from data import import_data, merge_counterfactual_empirical_data
from causal import create_causal_model, load_causal_model, counterfactual_forecast, empirical_forecast
from PIL import Image

st.set_page_config(page_title="Beacon Demo", layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

# initialize state
if 'rate' not in st.session_state:
    st.session_state["rate"] = 5.0
if 'point_of_div' not in st.session_state:
    st.session_state["point_of_div"] = 150
if 'forecast_quarters' not in st.session_state:
    st.session_state["forecast_quarters"] = 10
if 'graph' not in st.session_state:
    st.session_state["graph"] = {'nodes': ['Federal Rate'], 'edges': []}
if 'reset_pressed' not in st.session_state:
    st.session_state.reset_pressed = False

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def check_reset():
    if 'reset_pressed' in st.session_state and st.session_state.reset_pressed:
        st.session_state["rate"] = 5.0
        st.session_state["point_of_div"] = 150
        st.session_state["forecast_quarters"] = 10
        st.session_state["graph"] = {'nodes': ['Federal Rate'], 'edges': []}
        st.session_state.reset_pressed = False


# Call the check_reset function to modify the state if needed
check_reset()

with st.sidebar:
    st.title("Beacon")
    st.write("This is an example of a temporal causal graph. The federal interest rate impacts the mortgage rate which in turn impacts the median housing sales price in the United States. The Mortgage Rate and Sales Price in turn impact the rental vacancy proportion.")
    st.write("You can set a counterfactual by freezing one of the variables at a particular point in time. You can also set a forecast for the future. The underlying structural causal model will create the counterfactual model from the point of divergence, to the present, to the future with forecasts diverging from reality.")

# load data
dfi = import_data()
#causal_model = create_causal_model(dfi)
causal_model = load_causal_model()

reference_dict = {
    "Federal Rate": "FEDFUNDS",
    "Mortgage Rate": "MORTGAGE30US",
    "Sales Price": "MSPUS",
    "Rental Vacancies": "RRVRUSQ156N",
    "Lumber Index": "WPU081"
}

reference_unit_dict = {
    "Federal Rate": "%",
    "Mortgage Rate": "%",
    "Sales Price": "$USD",
    "Rental Vacancies": "%",
    "Lumber Index": "pts"
}

# cytoscape elements
elements = [
    {"data": {"id": "Federal Rate"}, "selected": True},
    {"data": {"id": "Mortgage Rate"}},
    {"data": {"id": "Sales Price"}},
    {"data": {"id": "Rental Vacancies"}},
    {"data": {"id": "Lumber Index"}},
    {"data": {"source": "Federal Rate", "target": "Mortgage Rate", "id": "FED➞MORTGAGE"}, "selectable": False},
    {"data": {"source": "Federal Rate", "target": "Lumber Index", "id": "FED➞LUMB"}, "selectable": False},
    {"data": {"source": "Mortgage Rate", "target": "Sales Price", "id": "MORTGAGE➞MSP"}, "selectable": False},
    {"data": {"source": "Lumber Index", "target": "Sales Price", "id": "LUMB➞MSP"}, "selectable": False},
    {"data": {"source": "Sales Price", "target": "Federal Rate", "id": "MSP➞FED"}, "selectable": False},
    {"data": {"source": "Mortgage Rate", "target": "Rental Vacancies", "id": "MORTGAGE➞RENT"}, "selectable": False},
    {"data": {"source": "Sales Price", "target": "Rental Vacancies", "id": "MSP➞RENT"}, "selectable": False},
    {"data": {"source": "Federal Rate", "target": "Federal Rate", "id": "FED➞FED"}, "selectable": False},
    {"data": {"source": "Mortgage Rate", "target": "Mortgage Rate", "id": "MORTGAGE➞MORTGAGE"}, "selectable": False},
    {"data": {"source": "Sales Price", "target": "Sales Price", "id": "MSP➞MSP"}, "selectable": False},
    {"data": {"source": "Rental Vacancies", "target": "Rental Vacancies", "id": "RENT➞RENT"}, "selectable": False},
    {"data": {"source": "Lumber Index", "target": "Lumber Index", "id": "LUMB➞LUMB"}, "selectable": False},
]

# cytoscape stylesheet
stylesheet = [
    {"selector": "node", "style": {"label": "data(id)", "width": 20, "height": 20}},
    {
        "selector": "edge",
        "style": {
            "width": 2,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
        },
    },
]

# cytoscape layout
layout = {"name": "fcose", "animationDuration": 0}
layout["alignmentConstraint"] = {"horizontal": [["Federal Rate", "Mortgage Rate"]]}
layout["relativePlacementConstraint"] = [{"left": "Federal Rate", "right": "Mortgage Rate"}]
layout["nodeRepulsion"] = 100000


col1, col2 = st.columns(2, gap="medium")

with col1:
    selected = cytoscape(
        elements,
        stylesheet,
        height="450px",
        layout=layout,
        selection_type="single",
        user_panning_enabled=False,
        user_zooming_enabled=False,
        key="graph")

    selected_nodes = ", ".join(selected["nodes"])
    if len(selected_nodes) > 0:
        selected_node = selected["nodes"][0]
    else:
        selected_node = ""
    reference_unit = reference_unit_dict.get(selected_node, "")
    if selected_node == "Sales Price":
        default_rate = 150000.0
        rate = st.number_input(f"Freeze {selected_node} at: ({reference_unit})", min_value=0.0, max_value=500000.0, value=default_rate, step=1000.0,
                               key="rate")
        rate = np.log(rate)
    elif selected_node == "Lumber Index":
        default_rate = 100.0
        rate = st.number_input(f"Freeze {selected_node} at: ({reference_unit})", min_value=0.0, max_value=1000.0, value=default_rate, step=1.0,
                               key="rate")
        rate = np.log(rate)
    else:
        default_rate = 5.0
        rate = st.number_input(f"Freeze {selected_node} at: ({reference_unit})", min_value=0.0, max_value=20.0, value=default_rate, step=0.1,
                               key="rate")

    point_of_div = st.number_input(f"Point of Divergence\n(in Quarters)", min_value=0, max_value=len(dfi), value=st.session_state["point_of_div"], step=1,
                                   key="point_of_div")

    forecast_quarters = st.number_input(f"Number of Quarters to Forecast", min_value=0, max_value=30, value=st.session_state["forecast_quarters"], step=1,
                                        key="forecast_quarters")

    if selected_node != "":
        # counterfactual and empirical calculations
        df_cf = counterfactual_forecast(dfi, causal_model,
                                        min_i=point_of_div,
                                        forecast_i=forecast_quarters,
                                        var=reference_dict[selected_node],
                                        rate=rate)

        df_emp = empirical_forecast(dfi, causal_model, forecast_i=forecast_quarters)

        dff = merge_counterfactual_empirical_data(df_cf, df_emp)

        csv = convert_df(dff)

        download_button = st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="data.csv",
            mime='text/csv'
        )

        reset_button = st.button("Reset")

with col2:
    if selected_node != "":
        for var in ["Sales Price", "Rental Vacancies", "Lumber Index", "Mortgage Rate", "Federal Rate"]:
            st.line_chart(data=dff, x=None, y=[f"{var} (Alternate)", f"{var} (Reality)"],
                          use_container_width=True)


if reset_button:
    st.session_state.reset_pressed = True
    st.rerun()
