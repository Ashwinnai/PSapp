import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
import re
import concurrent.futures

# PyWorkforce libraries
from pyworkforce.scheduling import MinAbsDifference, MinRequiredResources
from pyworkforce.queuing import MultiErlangC

# ------------------------------------------------------------------------------
#                           HELPER / UTILITY FUNCTIONS
# ------------------------------------------------------------------------------
@st.cache_data
def sanitize_sheet_name(sheet_name):
    return re.sub(r'[\\/*?:"<>|]', "_", sheet_name)

@st.cache_data
def adjust_interval_distribution(percentage_df):
    """Ensure each column sums to 100% by normalizing + small correction to largest cell."""
    for day in percentage_df.columns:
        total_percentage = percentage_df[day].sum()
        if total_percentage == 0:
            continue
        percentage_df[day] = percentage_df[day] / total_percentage * 100
        adjusted_total = percentage_df[day].sum()
        difference = 100 - adjusted_total
        if abs(difference) > 0.001:
            largest_index = percentage_df[day].idxmax()
            percentage_df.at[largest_index, day] += difference
    return percentage_df

@st.cache_data
def adjust_weekly_distribution(distribution_data):
    """Normalize day-level distribution to sum to 100%."""
    total_percentage = distribution_data["Percentage"].sum()
    if total_percentage == 0:
        return distribution_data
    distribution_data["Adjusted Percentage"] = distribution_data["Percentage"] / total_percentage * 100
    adjusted_total = distribution_data["Adjusted Percentage"].sum()
    difference = 100 - adjusted_total
    if abs(difference) > 0.001:
        largest_index = distribution_data["Adjusted Percentage"].idxmax()
        distribution_data.at[largest_index, "Adjusted Percentage"] += difference
    return distribution_data

@st.cache_data
def validate_and_convert_to_float(value, input_name):
    """Convert a string to float and handle errors."""
    try:
        result = float(value)
        if np.isnan(result):
            raise ValueError(f"Invalid input '{value}' is NaN for {input_name}.")
        return result
    except ValueError:
        raise ValueError(f"Invalid input '{value}' is not a valid number for {input_name}.")

@st.cache_data
def calculate_staffing(awt, shrinkage, max_occupancy, avg_aht, target, calls):
    """
    Calculate required positions using MultiErlangC for one scenario-line.
    """
    param_grid = {
        "transactions": [calls],
        "aht": [avg_aht / 60],  # Convert seconds to minutes
        "interval": [30],       # 30-min intervals
        "asa": [awt / 60],      # Convert seconds to minutes
        "shrinkage": [shrinkage / 100],
    }
    multi_erlang = MultiErlangC(param_grid=param_grid, n_jobs=-1)
    required_positions_scenarios = {
        "service_level": [target / 100],
        "max_occupancy": [max_occupancy / 100],
    }
    return multi_erlang.required_positions(required_positions_scenarios)

@st.cache_data
def generate_excel(all_scenarios, summary_df):
    """Generate an Excel file with scenario-level staffing details + summary sheet."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for scenario_name, (staffing_df, total_staffing, _, _) in all_scenarios.items():
            short_name = scenario_name.split(':')[0] 
            short_name_staffing = sanitize_sheet_name(f'{short_name}_Staff')
            short_name_summary = sanitize_sheet_name(f'{short_name}_Sum')
            short_name_staffing = short_name_staffing[:31]
            short_name_summary = short_name_summary[:31]

            # Pivot for staff distribution
            staffing_pivot = staffing_df.pivot(index='Interval', columns='Day', values='positions')
            staffing_pivot.to_excel(writer, sheet_name=short_name_staffing, index=True)
            total_staffing.to_excel(writer, sheet_name=short_name_summary)

        # Add scenario summary table
        summary_df.to_excel(writer, sheet_name='Scenario Summary Table', index=True)
    return output

def calculate_weighted_kpis(staffing_df):
    """Calculate Weighted ASA and Weighted Abandon Rate if columns exist."""
    if "waiting_probability" not in staffing_df.columns or "Calls" not in staffing_df.columns:
        return np.nan, np.nan

    staffing_df["ASA"] = staffing_df["AWT"] * staffing_df["waiting_probability"]
    staffing_df["Abandon_Rate(%)"] = staffing_df["waiting_probability"] * 5 * 100

    total_calls = staffing_df["Calls"].sum()
    if total_calls == 0:
        return np.nan, np.nan

    weighted_asa = (staffing_df["ASA"] * staffing_df["Calls"]).sum() / total_calls
    weighted_abandon = (staffing_df["Abandon_Rate(%)"] * staffing_df["Calls"]).sum() / total_calls

    return weighted_asa, weighted_abandon

def create_shifts_coverage(shift_names, periods):
    """Initialize shift coverage dictionary with zeros."""
    return {shift: [0] * periods for shift in shift_names}

def set_shift_coverage(shifts_coverage, shift, start, end, periods):
    """Mark coverage from 'start' to 'end' in the coverage array for each shift."""
    if end > start:
        shifts_coverage[shift][start:end] = [1]*(end - start)
    else:
        # wrap around scenario
        shifts_coverage[shift][start:periods] = [1]*(periods - start)
        shifts_coverage[shift][0:end] = [1]*end

def solve_scheduling_problem(method,
                             required_resources,
                             num_days,
                             num_periods,
                             shifts_coverage,
                             max_period_concurrency,
                             max_shift_concurrency,
                             costs,
                             objective_function="Minimize Cost",
                             cost_weight=0.7):
    """
    Solve scheduling with either MinAbsDifference or MinRequiredResources.
    """
    # Convert shape from (periods, days) => (days, periods)
    required_resources = list(map(list, zip(*required_resources)))

    scheduler_class = MinAbsDifference if method == "MinAbsDifference" else MinRequiredResources
    scheduler = scheduler_class(
        num_days=num_days,
        periods=num_periods,
        shifts_coverage=shifts_coverage,
        required_resources=required_resources,
        max_period_concurrency=max_period_concurrency,
        max_shift_concurrency=max_shift_concurrency,
        cost_dict=costs
    )

    if objective_function == "Minimize Cost":
        solution = scheduler.solve()
        solution['objective'] = "Minimize Cost"
    elif objective_function == "Maximize SL":
        # Placeholder: In real scenarios, adapt the solver for SL-based objectives
        solution = scheduler.solve()
        solution['objective'] = "Maximize SL"
    else:
        # Weighted combination (placeholder logic)
        solution = scheduler.solve()
        solution['objective'] = f"Weighted Combo (Cost={cost_weight*100:.0f}%, SL={(1-cost_weight)*100:.0f}%)"

    return solution

def calculate_costs(solution, costs):
    """Compute daily/shift costs and total from a scheduling solution dictionary."""
    daily_costs = {}
    shift_costs = {}
    total_cost = 0
    for item in solution['resources_shifts']:
        day = item['day']
        shift = item['shift']
        resources = item['resources']
        cost = resources * costs[shift]
        daily_costs[day] = daily_costs.get(day, 0) + cost
        shift_costs[shift] = shift_costs.get(shift, 0) + cost
        total_cost += cost
    return daily_costs, shift_costs, total_cost

def display_results(solution,
                    required_resources,
                    num_days,
                    periods,
                    shift_names,
                    shifts_coverage,
                    costs,
                    key_prefix="default"):
    """
    Show scheduling results: objective, total cost, cost tables, schedule pivot, 
    difference from required, etc.
    """
    daily_costs, shift_costs, total_cost = calculate_costs(solution, costs)
    objective = solution.get('objective', 'N/A')

    st.metric("Objective", objective)
    st.metric("Total Cost", f"${total_cost:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cost per Day")
        cost_per_day_df = pd.DataFrame.from_dict(daily_costs, orient='index', columns=['Cost'])
        cost_per_day_df.index.name = 'Day'
        st.dataframe(cost_per_day_df.style.highlight_max(axis=0))

    with col2:
        st.subheader("Cost per Shift")
        cost_per_shift_df = pd.DataFrame.from_dict(shift_costs, orient='index', columns=['Cost'])
        cost_per_shift_df.index.name = 'Shift'
        st.dataframe(cost_per_shift_df.style.highlight_max(axis=0))

    st.subheader("Generated Schedule")
    schedule_df = pd.DataFrame(solution['resources_shifts'])
    schedule_df = schedule_df.pivot(index='day', columns='shift', values='resources')
    schedule_df.index.name = "Day"
    st.dataframe(schedule_df.style.highlight_max(axis=1))

    st.subheader("Resource Requirements")
    req_df = pd.DataFrame(
        required_resources, 
        index=[f"Period {i+1}" for i in range(periods)],
        columns=[f"Day {i+1}" for i in range(num_days)]
    )
    st.dataframe(req_df.style.highlight_max(axis=1))

    # Requirements vs. Scheduled
    st.subheader("Requirements vs. Generated Schedule")
    fig = make_subplots(
        rows=num_days, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=[f"Day {i+1}" for i in range(num_days)]
    )

    differences = []
    full_day_dfs = []
    schedule_matrix = schedule_df.fillna(0)  # ensure no NaNs
    for day in range(num_days):
        required = [row[day] for row in required_resources]
        scheduled = [
            sum(schedule_matrix.loc[day, shift] * shifts_coverage[shift][period]
                for shift in shift_names)
            for period in range(periods)
        ]
        diff = [sch - req for sch, req in zip(scheduled, required)]
        differences.append(diff)

        fig.add_trace(
            go.Scatter(x=list(range(periods)), y=required,
                       mode='lines+markers', name=f'Required (Day {day+1})'),
            row=day+1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(periods)), y=scheduled,
                       mode='lines+markers', name=f'Scheduled (Day {day+1})'),
            row=day+1, col=1
        )

        day_df = pd.DataFrame({
            "Day": f"Day {day+1}",
            "Period": [f"Period {i+1}" for i in range(periods)],
            "Required": required,
            "Scheduled": scheduled,
            "Difference (Scheduled-Required)": diff
        })
        full_day_dfs.append(day_df)

    fig.update_layout(height=200*num_days, showlegend=False,
                      title_text="Requirements vs. Schedule per Day")
    fig.update_xaxes(title_text="Period")
    fig.update_yaxes(title_text="Resources")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_req_vs_sched")

    combined_df = pd.concat(full_day_dfs, ignore_index=True)
    st.subheader("Requirements vs. Schedule Data")
    st.dataframe(combined_df)

    # Over/Under staffing heatmap
    st.subheader("Over/Understaffing Heatmap")
    diff_df = pd.DataFrame(differences, 
                           index=[f"Day {i+1}" for i in range(num_days)],
                           columns=[f"Period {i+1}" for i in range(periods)])
    fig_diff = go.Figure(data=go.Heatmap(
        z=diff_df.values,
        x=diff_df.columns,
        y=diff_df.index,
        colorscale='RdBu',
        zmid=0
    ))
    fig_diff.update_layout(title='Difference (Scheduled - Required)')
    st.plotly_chart(fig_diff, use_container_width=True, key=f"{key_prefix}_diff_heatmap")

    st.subheader("Over/Understaffing Data")
    st.dataframe(diff_df)

    return total_cost

def display_scenario_details(scenario_name, staffing_df, total_staffing, days, intervals, working_hours, working_days, use_expander=True, context="Staffing"):
    """
    Display all details of a single scenario in an expander or directly.
    Includes ASA and Abandon Rate per interval.
    """
    if use_expander:
        with st.expander(scenario_name, expanded=False):
            # Display details inside the expander
            _display_scenario_details_content(scenario_name, staffing_df, total_staffing, days, intervals, working_hours, working_days, context)
    else:
        # Display details without using an expander
        st.markdown(f"### {scenario_name}")
        _display_scenario_details_content(scenario_name, staffing_df, total_staffing, days, intervals, working_hours, working_days, context)

def _display_scenario_details_content(scenario_name, staffing_df, total_staffing, days, intervals, working_hours, working_days, context):
    # Show interval-level ASA and Abandon Rate
    # ASA = AWT * waiting_probability
    # Abandon_Rate(%) = waiting_probability * 5 *100 (to keep consistent with scenario summary)
    if "waiting_probability" in staffing_df.columns and "Calls" in staffing_df.columns:
        staffing_df["ASA"] = staffing_df["AWT"] * staffing_df["waiting_probability"]
        staffing_df["Abandon_Rate(%)"] = staffing_df["waiting_probability"] * 5 * 100
    else:
        staffing_df["ASA"] = 0
        staffing_df["Abandon_Rate(%)"] = 0

    st.dataframe(staffing_df)

    st.write("Required Staffing")
    st.dataframe(total_staffing)

    # Staffing Levels Heatmap
    heatmap_data = staffing_df.pivot_table(index="Day", columns="Interval", values="positions", aggfunc="mean")
    heatmap_data = heatmap_data.reindex(days)
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlGnBu',
        colorbar=dict(title="Positions")
    ))
    fig.update_layout(
        title='Staffing Levels Heatmap',
        xaxis_nticks=48
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{sanitize_sheet_name(scenario_name)}_{context}_heatmap")

    st.write("Bar Plot per Day")
    for day_in_bar in days:
        bar_plot = go.Figure()
        day_data = staffing_df[staffing_df["Day"] == day_in_bar]
        bar_plot.add_trace(go.Bar(
            x=day_data["Interval"],
            y=day_data["positions"],
            name=day_in_bar
        ))
        bar_plot.update_layout(
            title=f'Staffing Levels for {day_in_bar}',
            xaxis_title="Interval",
            yaxis_title="Staffing Level",
            xaxis_nticks=48
        )
        st.plotly_chart(bar_plot, use_container_width=True, key=f"{sanitize_sheet_name(scenario_name)}_{context}_bar_{day_in_bar}")

# ------------------------------------------------------------------------------
#                            MAIN APP (Tabs)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Combined Staffing & Scheduling App", layout="wide")
st.title("Combined Staffing Calculator and Shift Schedule Generator")

# We define 3 tabs only: "Staffing Calculator", "Scheduling", "Scenario Comparison"
tab1, tab2, tab3 = st.tabs(["Staffing Calculator", "Scheduling", "Scenario Comparison"])

days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

if "interval_freq" not in st.session_state:
    st.session_state["interval_freq"] = "30T"
if "intervals" not in st.session_state:
    # Default to 30-min intervals (48 intervals in a day)
    st.session_state["intervals"] = pd.date_range("00:00", "23:30", freq="30T").time

# ------------------------------------------------------------------------------
#  TAB 1: Staffing Calculator
# ------------------------------------------------------------------------------
with tab1:
    intervals = st.session_state["intervals"]
    days = days_of_week

    st.subheader("Staffing Calculator")

    with st.sidebar.expander("Staffing User Inputs", expanded=True):
        acceptable_waiting_times_input = st.text_input("Acceptable Waiting Time (secs, comma-separated)", "30")
        try:
            acceptable_waiting_times = [
                validate_and_convert_to_float(awt, "Acceptable Waiting Times") 
                for awt in acceptable_waiting_times_input.split(',')
            ]
        except ValueError as e:
            st.error(str(e))
            st.stop()

        shrinkages_input = st.text_input("Shrinkage (% , comma-separated)", "20")
        try:
            shrinkages = [
                validate_and_convert_to_float(shrink, "Shrinkage") 
                for shrink in shrinkages_input.split(',')
            ]
        except ValueError as e:
            st.error(str(e))
            st.stop()

        max_occupancies_input = st.text_input("Max Occupancy (% , comma-separated)", "80")
        try:
            max_occupancies = [
                validate_and_convert_to_float(occ, "Max Occupancy") 
                for occ in max_occupancies_input.split(',')
            ]
        except ValueError as e:
            st.error(str(e))
            st.stop()

        service_level_targets_input = st.text_input("Service Level Targets (% , comma-separated)", "80")
        try:
            service_level_targets = [
                validate_and_convert_to_float(target, "Service Level Targets") 
                for target in service_level_targets_input.split(',')
            ]
        except ValueError as e:
            st.error(str(e))
            st.stop()

        working_hours = st.number_input("Working Hours per Day", min_value=1.0, max_value=24.0, value=8.0, step=0.5)
        working_days = st.number_input("Working Days per Week", min_value=1.0, max_value=7.0, value=5.0, step=0.5)

        aht_input_option = st.radio(
            "Choose AHT Input Method:",
            ("Multiple AHT values for all intervals and days", "AHT table at interval level for each day")
        )

        if aht_input_option == "Multiple AHT values for all intervals and days":
            average_handling_times_input = st.text_input("Average Handling Times (secs, comma-separated)", "300")
            try:
                average_handling_times = [
                    validate_and_convert_to_float(aht, "Average Handling Times") 
                    for aht in average_handling_times_input.split(',')
                ]
            except ValueError as e:
                st.error(str(e))
                st.stop()
        else:
            average_handling_times = []

        manual_input = st.checkbox("Auto Input Calls Offered per Interval", value=False)

    intervals_list = list(intervals)
    if "calls_df" not in st.session_state:
        st.session_state["calls_df"] = pd.DataFrame(index=intervals_list, columns=days)

    if manual_input:
        st.subheader("Using Weekly Volume + Distributions")
        with st.expander("Weekly Volume", expanded=True):
            weekly_volume = st.number_input("Enter Weekly Volume", min_value=0, value=800, step=100)

        with st.expander("Weekly Distribution (%)", expanded=True):
            distribution_data = pd.DataFrame({"Day": days, "Percentage": [15.0,20.0,20.0,15.0,15.0,10.0,5.0]})
            distribution_data["Percentage"] = distribution_data["Percentage"].astype(float)
            distribution_data = st.data_editor(distribution_data, key="distribution_data_editor")
            adjusted_distribution_data = adjust_weekly_distribution(distribution_data)
            st.dataframe(adjusted_distribution_data)

        with st.expander("Interval Distribution (Sun-Sat)", expanded=True):
            initial_data = [[2.08 for _ in days] for _ in intervals_list]
            percentage_df = pd.DataFrame(initial_data, index=intervals_list, columns=days)
            percentage_df = st.data_editor(percentage_df, key="percentage_df_editor")
            adjusted_percentage_df = adjust_interval_distribution(percentage_df)
            st.dataframe(adjusted_percentage_df)

            calls_df = pd.DataFrame(index=intervals_list)
            daily_volumes = [
                (weekly_volume * (dist / 100)) for dist in adjusted_distribution_data["Adjusted Percentage"].tolist()
            ]
            for day, daily_volume in zip(days, daily_volumes):
                calls_per_interval = [(daily_volume * (p / 100)) for p in adjusted_percentage_df[day]]
                calls_df[day] = calls_per_interval

            st.session_state["calls_df"] = calls_df
            st.data_editor(st.session_state["calls_df"], key="manual_calls_df_editor")
    else:
        st.subheader("Manual DataFrame for Calls Offered")
        st.session_state["calls_df"] = st.data_editor(st.session_state["calls_df"], key="calls_df_editor")

    if aht_input_option == "AHT table at interval level for each day":
        st.markdown("### AHT per Interval")
        if "aht_df" not in st.session_state:
            data_aht = {day: [0.0]*len(intervals_list) for day in days}
            aht_df = pd.DataFrame(data_aht, index=intervals_list)
            st.session_state["aht_df"] = aht_df
        st.session_state["aht_df"] = st.data_editor(st.session_state["aht_df"], key="aht_df_editor")

    if 'all_scenarios' not in st.session_state:
        st.session_state['all_scenarios'] = {}
    if 'scenario_summary' not in st.session_state:
        st.session_state['scenario_summary'] = pd.DataFrame()

    if "stop_run" not in st.session_state:
        st.session_state["stop_run"] = False

    def stop_calculations():
        st.session_state["stop_run"] = True

    stop_button = st.button("Stop Calculation", on_click=stop_calculations)
    calc_button = st.button("Calculate Staffing Requirements")

    if calc_button:
        st.session_state["stop_run"] = False
        start_time = time.time()

        progress_placeholder = st.empty()
        partial_results_placeholder = st.empty()

        intervals_list = list(intervals)
        num_intervals = len(intervals_list)
        num_days = len(days)

        total_combinations = (
            len(acceptable_waiting_times) *
            len(shrinkages) *
            len(max_occupancies) *
            len(service_level_targets) *
            num_days *
            num_intervals
        )

        progress = 0
        scenario_number = 1
        current_run_scenarios = {}

        partial_summary = pd.DataFrame(columns=[
            "Scenario", 
            "Required staff without Peak staffing", 
            "Peak Staffing requirement",
            "Weighted ASA (seconds, approx)",
            "Weighted Abandonment Rate (approx%)"
        ])

        for awt in acceptable_waiting_times:
            for shrinkage in shrinkages:
                for max_occupancy in max_occupancies:
                    if st.session_state["stop_run"]:
                        st.warning("Calculation stopped by user.")
                        break

                    # AHT logic
                    if aht_input_option == "Multiple AHT values for all intervals and days":
                        for avg_aht in average_handling_times:
                            for target in service_level_targets:
                                if st.session_state["stop_run"]:
                                    break

                                scenario_name = f"Scn{scenario_number}"
                                scenario_label = (f"{scenario_name}: AWT={awt}s, Shr={shrinkage}%, "
                                                  f"Occ={max_occupancy}%, AHT={avg_aht}s, SLT={target}%")

                                staffing_results = []
                                for day in days:
                                    for interval_time, calls in zip(intervals_list, st.session_state["calls_df"][day]):
                                        if st.session_state["stop_run"]:
                                            break

                                        calls_val = validate_and_convert_to_float(calls, f"Calls for {day} {interval_time}")
                                        if calls_val == 0:
                                            requirement = {
                                                "Day": day, "Interval": interval_time,
                                                "AWT": awt, "Shrinkage": shrinkage,
                                                "Max Occupancy": max_occupancy,
                                                "Average AHT": avg_aht, "Service Level Target": target,
                                                "raw_positions": 0, "positions": 0, "service_level": 0,
                                                "occupancy": 0, "waiting_probability": 0, "Calls": calls_val
                                            }
                                            staffing_results.append(requirement)
                                            progress += 1
                                            continue
                                        result = calculate_staffing(awt, shrinkage, max_occupancy, avg_aht, target, calls_val)
                                        for req in result:
                                            req.update({
                                                "Day": day,
                                                "Interval": interval_time,
                                                "AWT": awt,
                                                "Shrinkage": shrinkage,
                                                "Max Occupancy": max_occupancy,
                                                "Average AHT": avg_aht,
                                                "Service Level Target": target,
                                                "Calls": calls_val
                                            })
                                            staffing_results.append(req)
                                        progress += 1

                                    if st.session_state["stop_run"]:
                                        break

                                progress_text = f"Processed {progress}/{total_combinations} scenario-sub-steps..."
                                progress_placeholder.progress(min(progress / total_combinations, 1.0), text=progress_text)

                                if st.session_state["stop_run"]:
                                    break

                                staffing_df = pd.DataFrame(staffing_results)
                                required_cols = [
                                    "Day", "Interval", "AWT", "Shrinkage", "Max Occupancy",
                                    "Average AHT", "Service Level Target", "raw_positions", "positions",
                                    "service_level", "occupancy", "waiting_probability", "Calls"
                                ]
                                for col in required_cols:
                                    if col not in staffing_df.columns:
                                        staffing_df[col] = 0
                                staffing_df = staffing_df[required_cols]

                                total_staffing = staffing_df.groupby("Day")[["raw_positions","positions"]].sum()
                                total_staffing["Sum of Raw Positions"] = total_staffing["raw_positions"]
                                total_staffing["Sum of Positions"] = total_staffing["positions"]
                                total_staffing["Required staff without Peak staffing"] = \
                                    total_staffing["Sum of Positions"] / 2 / working_hours
                                total_staffing["Peak Staffing requirement"] = \
                                    total_staffing["Required staff without Peak staffing"].max()
                                total_staffing["Sum of the Week"] = \
                                    total_staffing["Required staff without Peak staffing"].sum()
                                total_staffing["Required staff without Peak staffing"] = \
                                    total_staffing["Sum of the Week"] / working_days
                                total_staffing = total_staffing.reindex(days)

                                weighted_asa, weighted_abandon = calculate_weighted_kpis(staffing_df)
                                current_run_scenarios[scenario_label] = (
                                    staffing_df, total_staffing, weighted_asa, weighted_abandon
                                )

                                req_staff_no_peak = total_staffing["Required staff without Peak staffing"].max()
                                peak_req = total_staffing["Peak Staffing requirement"].max()
                                row = {
                                    "Scenario": scenario_label,
                                    "Required staff without Peak staffing": req_staff_no_peak,
                                    "Peak Staffing requirement": peak_req,
                                    "Weighted ASA (seconds, approx)": weighted_asa,
                                    "Weighted Abandonment Rate (approx%)": weighted_abandon
                                }
                                partial_summary = pd.concat([partial_summary, pd.DataFrame([row])], ignore_index=True)
                                partial_results_placeholder.dataframe(partial_summary)

                                scenario_number += 1
                    else:
                        # AHT from aht_df
                        for target in service_level_targets:
                            if st.session_state["stop_run"]:
                                break

                            scenario_name = f"Scn{scenario_number}"
                            scenario_label = (f"{scenario_name}: AWT={awt}s, Shr={shrinkage}%, "
                                              f"Occ={max_occupancy}%, SLT={target}%")

                            staffing_results = []
                            for day in days:
                                for interval_time, calls, aht_val in zip(
                                    intervals_list, st.session_state["calls_df"][day], st.session_state["aht_df"][day]
                                ):
                                    if st.session_state["stop_run"]:
                                        break
                                    calls_val = validate_and_convert_to_float(calls, f"Calls for {day} {interval_time}")
                                    aht_val = validate_and_convert_to_float(aht_val, f"AHT for {day} {interval_time}")

                                    if calls_val == 0 or aht_val == 0:
                                        requirement = {
                                            "Day": day, "Interval": interval_time,
                                            "AWT": awt, "Shrinkage": shrinkage, "Max Occupancy": max_occupancy,
                                            "Average AHT": aht_val, "Service Level Target": target,
                                            "raw_positions": 0, "positions": 0, "service_level": 0,
                                            "occupancy": 0, "waiting_probability": 0, "Calls": calls_val
                                        }
                                        staffing_results.append(requirement)
                                        progress += 1
                                        continue

                                    result = calculate_staffing(awt, shrinkage, max_occupancy, aht_val, target, calls_val)
                                    for req in result:
                                        req.update({
                                            "Day": day,
                                            "Interval": interval_time,
                                            "AWT": awt,
                                            "Shrinkage": shrinkage,
                                            "Max Occupancy": max_occupancy,
                                            "Average AHT": aht_val,
                                            "Service Level Target": target,
                                            "Calls": calls_val
                                        })
                                        staffing_results.append(req)
                                    progress += 1

                            progress_text = f"Processed {progress}/{total_combinations} scenario-sub-steps..."
                            progress_placeholder.progress(min(progress / total_combinations, 1.0), text=progress_text)

                            if st.session_state["stop_run"]:
                                break

                            staffing_df = pd.DataFrame(staffing_results)
                            required_cols = [
                                "Day", "Interval", "AWT", "Shrinkage", "Max Occupancy",
                                "Average AHT", "Service Level Target", "raw_positions", "positions",
                                "service_level", "occupancy", "waiting_probability", "Calls"
                            ]
                            for col in required_cols:
                                if col not in staffing_df.columns:
                                    staffing_df[col] = 0
                            staffing_df = staffing_df[required_cols]

                            total_staffing = staffing_df.groupby("Day")[["raw_positions","positions"]].sum()
                            total_staffing["Sum of Raw Positions"] = total_staffing["raw_positions"]
                            total_staffing["Sum of Positions"] = total_staffing["positions"]
                            total_staffing["Required staff without Peak staffing"] = \
                                total_staffing["Sum of Positions"] / 2 / working_hours
                            total_staffing["Peak Staffing requirement"] = \
                                total_staffing["Required staff without Peak staffing"].max()
                            total_staffing["Sum of the Week"] = \
                                total_staffing["Required staff without Peak staffing"].sum()
                            total_staffing["Required staff without Peak staffing"] = \
                                total_staffing["Sum of the Week"] / working_days
                            total_staffing = total_staffing.reindex(days)

                            weighted_asa, weighted_abandon = calculate_weighted_kpis(staffing_df)
                            current_run_scenarios[scenario_label] = (
                                staffing_df, total_staffing, weighted_asa, weighted_abandon
                            )

                            req_staff_no_peak = total_staffing["Required staff without Peak staffing"].max()
                            peak_req = total_staffing["Peak Staffing requirement"].max()
                            row = {
                                "Scenario": scenario_label,
                                "Required staff without Peak staffing": req_staff_no_peak,
                                "Peak Staffing requirement": peak_req,
                                "Weighted ASA (seconds, approx)": weighted_asa,
                                "Weighted Abandonment Rate (approx%)": weighted_abandon
                            }
                            partial_summary = pd.concat([partial_summary, pd.DataFrame([row])], ignore_index=True)
                            partial_results_placeholder.dataframe(partial_summary)

                            scenario_number += 1

        # Update all scenarios in session state
        for scenario, data_tuple in current_run_scenarios.items():
            st.session_state["all_scenarios"][scenario] = data_tuple

        progress_placeholder.empty()
        end_time = time.time()
        st.write(f"**Time Taken**: {end_time - start_time:.2f} seconds")

        # Generate summary and store
        summary_df = pd.DataFrame()
        for scenario, (staffing_df, total_staffing, weighted_asa, weighted_abandon) in st.session_state["all_scenarios"].items():
            req_staff_no_peak = total_staffing["Required staff without Peak staffing"].max()
            peak_req = total_staffing["Peak Staffing requirement"].max()
            row = {
                "Scenario": scenario,
                "Required staff without Peak staffing": req_staff_no_peak,
                "Peak Staffing requirement": peak_req,
                "Weighted ASA (seconds, approx)": weighted_asa,
                "Weighted Abandonment Rate (approx%)": weighted_abandon
            }
            summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)
        st.session_state["scenario_summary"] = summary_df

    if "scenario_summary" in st.session_state and not st.session_state["scenario_summary"].empty:
        st.header("Scenario Summary Table")
        st.dataframe(st.session_state["scenario_summary"])

        excel_data = generate_excel(st.session_state["all_scenarios"], st.session_state["scenario_summary"])
        st.download_button(
            label="Download Excel",
            data=excel_data.getvalue(),
            file_name='staffing_requirements.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    # Always display scenario details if they exist
    if "all_scenarios" in st.session_state and st.session_state["all_scenarios"]:
        st.header("Scenario Details")
        for scenario_name, (staffing_df, total_staffing, weighted_asa, weighted_abandon) in st.session_state["all_scenarios"].items():
            display_scenario_details(scenario_name, staffing_df, total_staffing, days, intervals, working_hours, working_days, use_expander=True, context="Staffing")

# ------------------------------------------------------------------------------
#  TAB 2: Scheduling
# ------------------------------------------------------------------------------
with tab2:
    st.header("Shift Scheduling")

    if "all_scenarios" in st.session_state and st.session_state["all_scenarios"]:
        scenarios = list(st.session_state["all_scenarios"].keys())

        # Buttons to quickly select/deselect all
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("Select All Scenarios"):
                st.session_state["selected_scenarios"] = scenarios
        with col_s2:
            if st.button("Deselect All"):
                st.session_state["selected_scenarios"] = []

        selected = st.multiselect(
            "Choose Scenario(s) to Generate Schedules From",
            scenarios,
            default=st.session_state.get("selected_scenarios", [])
        )
        st.session_state["selected_scenarios"] = selected

        st.sidebar.header("Scheduling Objective Function")
        objective_function = st.sidebar.selectbox(
            "Select Objective",
            ["Minimize Cost", "Maximize SL", "Weighted Combination"]
        )
        cost_weight = 0.7
        if objective_function == "Weighted Combination":
            cost_weight = st.sidebar.slider(
                "Cost Weight (Cost vs. SL)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Cost Weight % (remaining is SL Weight)"
            )
            st.sidebar.write(f"**Cost:** {cost_weight*100:.0f}% | **SL:** {(1-cost_weight)*100:.0f}%")

        # SHIFT PARAMETERS
        st.sidebar.header("Shift Scheduling Parameters")
        with st.sidebar.expander("📅 Schedule Parameters", expanded=True):
            num_days = len(days_of_week)
            num_periods = len(st.session_state["intervals"])
            st.write(f"{num_days} days (Sun-Sat)")
            st.write(f"{num_periods} intervals/day")
            num_shifts = st.number_input("Number of Shifts", min_value=1, value=6)

        shift_names = [f"Shift_{i+1}" for i in range(num_shifts)]
        shifts_coverage = create_shifts_coverage(shift_names, num_periods)

        # Default coverage examples
        default_shift_configs = [
            (0,16),  (16,32),  (32,48),
            (10,26), (20,36),  (15,31)
        ]

        with st.sidebar.expander("⏰ Shift Coverage", expanded=True):
            for i, shift in enumerate(shift_names):
                st.subheader(shift)
                if i < len(default_shift_configs):
                    dstart, dend = default_shift_configs[i]
                    dend = min(dend, num_periods)
                else:
                    dstart, dend = (0, num_periods)

                col1, col2 = st.columns(2)
                with col1:
                    start = st.number_input(
                        f"Start {shift}",
                        min_value=0,
                        max_value=num_periods-1,
                        value=dstart,
                        key=f"start_{shift}"
                    )
                with col2:
                    end = st.number_input(
                        f"End {shift}",
                        min_value=0,
                        max_value=num_periods,
                        value=dend,
                        key=f"end_{shift}"
                    )
                set_shift_coverage(shifts_coverage, shift, start, end, num_periods)

        with st.sidebar.expander("🔢 Resource Limits", expanded=True):
            max_period_concurrency = st.number_input("Max resources per period", min_value=1, value=30)
            max_shift_concurrency = st.number_input("Max resources per shift", min_value=1, value=25)

        with st.sidebar.expander("💰 Shift Costs", expanded=True):
            costs = {
                shift: st.number_input(
                    f"Cost for {shift}", 
                    min_value=0.1, 
                    value=0.2, 
                    step=0.1
                )
                for shift in shift_names
            }

        st.session_state['scheduling_summary'] = pd.DataFrame()
        scenario_solutions = {}

        if st.button("Generate Schedules"):
            start_time_sched = time.time()
            scheduling_results = []

            with st.spinner("Generating Schedules..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_map = {}
                    for sc in selected:
                        scenario_solutions[sc] = {
                            "MinAbsDifference": None,
                            "MinRequiredResources": None
                        }
                        staff_df, _, _, _ = st.session_state["all_scenarios"][sc]
                        pivot_req = staff_df.pivot(index="Interval", columns="Day", values="positions")
                        pivot_req = pivot_req.reindex(columns=days_of_week)
                        required_resources = pivot_req.values.tolist()

                        f_abs = executor.submit(
                            solve_scheduling_problem,
                            "MinAbsDifference",
                            required_resources,
                            num_days,
                            num_periods,
                            shifts_coverage,
                            max_period_concurrency,
                            max_shift_concurrency,
                            costs,
                            objective_function,
                            cost_weight
                        )
                        f_req = executor.submit(
                            solve_scheduling_problem,
                            "MinRequiredResources",
                            required_resources,
                            num_days,
                            num_periods,
                            shifts_coverage,
                            max_period_concurrency,
                            max_shift_concurrency,
                            costs,
                            objective_function,
                            cost_weight
                        )

                        future_map[f_abs] = (sc, "MinAbsDifference", required_resources)
                        future_map[f_req] = (sc, "MinRequiredResources", required_resources)

                    for fut in concurrent.futures.as_completed(future_map):
                        sc, method, req_res = future_map[fut]
                        sol = fut.result()

                        scenario_solutions[sc][method] = sol

                        if sol['status'] == 'OPTIMAL':
                            _, _, total_cost_val = calculate_costs(sol, costs)
                            scheduling_results.append({
                                "Scenario": sc,
                                "Method": method,
                                "Objective": sol.get('objective', 'N/A'),
                                "Cost Weight(%)": f"{cost_weight*100:.0f}%",
                                "Total Cost": total_cost_val
                            })
                        else:
                            scheduling_results.append({
                                "Scenario": sc,
                                "Method": method,
                                "Objective": sol.get('objective', 'N/A'),
                                "Cost Weight(%)": f"{cost_weight*100:.0f}%",
                                "Total Cost": float('nan')
                            })

            end_time_sched = time.time()
            st.write(f"Scheduling completed in {end_time_sched - start_time_sched:.2f} seconds.")

            st.session_state['scheduling_summary'] = pd.DataFrame(scheduling_results)

            for sc in selected:
                with st.expander(f"Scheduling Results - {sc}", expanded=False):
                    col_left, col_right = st.columns(2)

                    sol_abs = scenario_solutions[sc]["MinAbsDifference"]
                    with col_left:
                        st.markdown("### MinAbsDifference")
                        if sol_abs and sol_abs['status'] == 'OPTIMAL':
                            staff_df, _, _, _ = st.session_state["all_scenarios"][sc]
                            pivot_req = staff_df.pivot(index="Interval", columns="Day", values="positions")
                            pivot_req = pivot_req.reindex(columns=days_of_week)
                            required_resources = pivot_req.values.tolist()

                            display_results(
                                sol_abs,
                                required_resources,
                                num_days,
                                num_periods,
                                shift_names,
                                shifts_coverage,
                                costs,
                                key_prefix=f"MinAbs_{sanitize_sheet_name(sc)}"
                            )
                        else:
                            st.error("No feasible schedule or solution not found.")

                    sol_req = scenario_solutions[sc]["MinRequiredResources"]
                    with col_right:
                        st.markdown("### MinRequiredResources")
                        if sol_req and sol_req['status'] == 'OPTIMAL':
                            staff_df, _, _, _ = st.session_state["all_scenarios"][sc]
                            pivot_req = staff_df.pivot(index="Interval", columns="Day", values="positions")
                            pivot_req = pivot_req.reindex(columns=days_of_week)
                            required_resources = pivot_req.values.tolist()

                            display_results(
                                sol_req,
                                required_resources,
                                num_days,
                                num_periods,
                                shift_names,
                                shifts_coverage,
                                costs,
                                key_prefix=f"MinReq_{sanitize_sheet_name(sc)}"
                            )
                        else:
                            st.error("No feasible schedule or solution not found.")

                    # Display Scenario Details within Scheduling Tab without using an expander
                    staffing_df, total_staffing, weighted_asa, weighted_abandon = st.session_state["all_scenarios"][sc]
                    display_scenario_details(sc, staffing_df, total_staffing, days, intervals, working_hours, working_days, use_expander=False, context="Scheduling")

    else:
        st.info("No scenarios available. Please run the Staffing Calculator first.")

    if 'scheduling_summary' in st.session_state and not st.session_state['scheduling_summary'].empty:
        st.header("Scheduling Summary")
        st.dataframe(st.session_state['scheduling_summary'])

        sched_output = io.BytesIO()
        with pd.ExcelWriter(sched_output, engine='xlsxwriter') as writer:
            st.session_state['scheduling_summary'].to_excel(writer, sheet_name='Scheduling_Summary', index=False)
        st.download_button(
            label="Download Scheduling Summary",
            data=sched_output.getvalue(),
            file_name='scheduling_summary.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

# ------------------------------------------------------------------------------
#  TAB 3: Scenario Comparison (Merged + Single-Scale Plot)
# ------------------------------------------------------------------------------
with tab3:
    st.header("Scenario Comparison & Merged DataFrame")

    # Staffing Comparison
    if "scenario_summary" in st.session_state and not st.session_state["scenario_summary"].empty:
        st.subheader("Staffing Comparison")
        scenario_df = st.session_state["scenario_summary"].copy()
        st.dataframe(scenario_df)
    else:
        scenario_df = pd.DataFrame()
        st.info("No Staffing Scenarios found.")

    # Scheduling Comparison
    if "scheduling_summary" in st.session_state and not st.session_state['scheduling_summary'].empty:
        st.subheader("Scheduling Comparison")
        scheduling_df = st.session_state["scheduling_summary"].copy()
        st.dataframe(scheduling_df)
    else:
        scheduling_df = pd.DataFrame()
        st.info("No Scheduling Results found.")

    # Merge both data frames
    if not scenario_df.empty and not scheduling_df.empty:
        st.subheader("Merged Staffing & Scheduling DataFrame")

        # Merge on "Scenario", do a left join or outer join if needed
        merged_df = scenario_df.merge(
            scheduling_df,
            on="Scenario",
            how="outer",
            suffixes=("_Staffing", "_Scheduling")
        )

        st.dataframe(merged_df)

        # Plot "Required staff without peak staffing", "Peak Staffing requirement", and "Total Cost"
        # on the same scale in a grouped bar chart
        st.subheader("Comparison Plot: Staff & Cost per Scenario (Unified Scale)")
        fig_merged = go.Figure()

        # 1) Required staff without peak staffing (bar)
        fig_merged.add_trace(
            go.Bar(
                x=merged_df["Scenario"],
                y=merged_df["Required staff without Peak staffing"],
                name="Staff w/o Peak"
            )
        )

        # 2) Peak Staffing requirement (bar)
        fig_merged.add_trace(
            go.Bar(
                x=merged_df["Scenario"],
                y=merged_df["Peak Staffing requirement"],
                name="Peak Staffing"
            )
        )

        # 3) Total Cost (bar)
        cost_vals = merged_df["Total Cost"].fillna(0)
        fig_merged.add_trace(
            go.Bar(
                x=merged_df["Scenario"],
                y=cost_vals,
                name="Total Cost"
            )
        )

        # Single y-axis: barmode='group' so they share the same scale
        fig_merged.update_layout(
            barmode='group',
            title="Required Staffing vs. Total Cost (Same Scale)",
            xaxis_title="Scenario",
            yaxis_title="Staffing / Cost (Unified Scale)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_merged, use_container_width=True)

    else:
        st.write("No merged data to plot. Please ensure you have run both Staffing and Scheduling steps.")

# Footer
st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 6px; background-color: #0e1117;">
        <p style='color:white;'>| Developed with ❤️ by Ashwin Nair |</p>
    </div>
""", unsafe_allow_html=True)
