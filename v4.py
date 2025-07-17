from io import BytesIO
import pandas as pd
import math
from math import sin, cos, atan2, radians, sqrt
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def haversine_distance(clat, clon, blat, blon):
	if math.isnan(clat) or math.isnan(clon) or math.isnan(blat) or math.isnan(blon):
		return 0
		
	delta_lat = radians(clat - blat)
	delta_lon = radians(clon - blon)
	
	a = sin(delta_lat/2)**2 + cos(radians(clat))*cos(radians(blat))*sin(delta_lon/2)**2
	c = 2*atan2(sqrt(a), sqrt(1-a))
	distance = 6371*c
	return distance

def merge_dfs(customer_data, banker_data, branch_data):
	customer_data = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
	final_table = customer_data.merge(banker_data, on = "PORT_CODE", how = "left")
	final_table.fillna(0, inplace = True)
	return final_table

def create_distance_circle(center_lat, center_lon, radius_km, num_points=100):
	"""Create points for a circle around a center point"""
	angles = np.linspace(0, 2*np.pi, num_points)
	circle_lats = []
	circle_lons = []
	
	for angle in angles:
		# Convert km to degrees (rough approximation)
		lat_offset = radius_km / 111.0  # 1 degree lat ≈ 111 km
		lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))
		
		lat = center_lat + lat_offset * math.cos(angle)
		lon = center_lon + lon_offset * math.sin(angle)
		
		circle_lats.append(lat)
		circle_lons.append(lon)
	
	# Close the circle
	circle_lats.append(circle_lats[0])
	circle_lons.append(circle_lons[0])
	
	return circle_lats, circle_lons

def create_interactive_map(filtered_data, au_data, max_distance=60, form_id=None):
	"""Create an interactive map using Plotly"""
	
	fig = go.Figure()
	
	# Add AU markers
	if not au_data.empty:
		for _, au in au_data.iterrows():
			# Add AU marker
			fig.add_trace(go.Scattermapbox(
				lat=[au['BRANCH_LAT_NUM']],
				lon=[au['BRANCH_LON_NUM']],
				mode='markers',
				marker=dict(
					size=15,
					color='red',
					symbol='building'
				),
				text=f"AU {au['AU']}",
				hovertemplate=f"""
				<b>AU {au['AU']}</b><br>
				City: {au.get('CITY', 'N/A')}<br>
				State: {au.get('STATECODE', 'N/A')}<br>
				Coordinates: {au['BRANCH_LAT_NUM']:.4f}, {au['BRANCH_LON_NUM']:.4f}
				<extra></extra>
				""",
				name=f"AU {au['AU']}",
				showlegend=True
			))
			
			# Add distance circle
			circle_lats, circle_lons = create_distance_circle(
				au['BRANCH_LAT_NUM'], au['BRANCH_LON_NUM'], max_distance
			)
			
			fig.add_trace(go.Scattermapbox(
				lat=circle_lats,
				lon=circle_lons,
				mode='lines',
				line=dict(width=2, color='red'),
				opacity=0.5,
				name=f"Distance Circle ({max_distance}km)",
				showlegend=True,
				hoverinfo='skip'
			))
	
	# Add customer markers
	if not filtered_data.empty:
		# Determine colors and labels
		if form_id:
			color = 'green'
			name = f"Form {form_id} Customers"
		else:
			color = 'blue'
			name = "Selected Customers"
		
		# Create hover text
		hover_text = []
		for _, customer in filtered_data.iterrows():
			hover_text.append(f"""
			<b>{customer.get('CG_ECN', 'N/A')}</b><br>
			Portfolio: {customer.get('PORT_CODE', 'N/A')}<br>
			Distance: {customer.get('Distance', 0):.1f} km<br>
			Revenue: ${customer.get('BANK_REVENUE', 0):,.0f}<br>
			Deposit: ${customer.get('DEPOSIT_BAL', 0):,.0f}<br>
			State: {customer.get('BILLINGSTATE', 'N/A')}<br>
			Type: {customer.get('TYPE', 'N/A')}
			""")
		
		fig.add_trace(go.Scattermapbox(
			lat=filtered_data['LAT_NUM'],
			lon=filtered_data['LON_NUM'],
			mode='markers',
			marker=dict(
				size=8,
				color=color,
				symbol='circle'
			),
			hovertemplate='%{text}<extra></extra>',
			text=hover_text,
			name=name,
			showlegend=True
		))
	
	# Update layout
	if not au_data.empty:
		center_lat = au_data['BRANCH_LAT_NUM'].iloc[0]
		center_lon = au_data['BRANCH_LON_NUM'].iloc[0]
		zoom = 8
	else:
		center_lat = 39.8283
		center_lon = -98.5795
		zoom = 4
	
	fig.update_layout(
		mapbox=dict(
			style="open-street-map",
			center=dict(lat=center_lat, lon=center_lon),
			zoom=zoom
		),
		height=500,
		margin=dict(l=0, r=0, t=0, b=0),
		showlegend=True,
		legend=dict(
			yanchor="top",
			y=0.99,
			xanchor="left",
			x=0.01,
			bgcolor="rgba(255,255,255,0.8)"
		)
	)
	
	return fig

def create_combined_map(form_results, branch_data):
	"""Create a combined map showing all forms and their customers"""
	
	if not form_results:
		return None
	
	fig = go.Figure()
	
	# Color scheme for different forms
	form_colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'lightblue', 'pink', 'darkgreen', 'brown', 'gray']
	
	# Get all unique AU locations
	au_locations = set()
	for form_id, df in form_results.items():
		if not df.empty:
			au_locations.add((df['AU'].iloc[0], df['BRANCH_LAT_NUM'].iloc[0], df['BRANCH_LON_NUM'].iloc[0]))
	
	# Add AU markers
	for au_id, au_lat, au_lon in au_locations:
		au_details = branch_data[branch_data['AU'] == au_id]
		au_name = au_details['CITY'].iloc[0] if not au_details.empty else f"AU {au_id}"
		
		fig.add_trace(go.Scattermapbox(
			lat=[au_lat],
			lon=[au_lon],
			mode='markers',
			marker=dict(
				size=15,
				color='red',
				symbol='building'
			),
			text=f"AU {au_id}",
			hovertemplate=f"""
			<b>AU {au_id}</b><br>
			Location: {au_name}<br>
			Coordinates: {au_lat:.4f}, {au_lon:.4f}
			<extra></extra>
			""",
			name=f"AU {au_id}",
			showlegend=True
		))
	
	# Add customers from each form
	for form_id, df in form_results.items():
		if df.empty:
			continue
		
		color = form_colors[(form_id - 1) % len(form_colors)]
		
		# Create hover text for this form
		hover_text = []
		for _, customer in df.iterrows():
			hover_text.append(f"""
			<b>{customer.get('CG_ECN', 'N/A')}</b><br>
			Form: {form_id}<br>
			Portfolio: {customer.get('PORT_CODE', 'N/A')}<br>
			Distance: {customer.get('Distance', 0):.1f} km<br>
			Revenue: ${customer.get('BANK_REVENUE', 0):,.0f}<br>
			Deposit: ${customer.get('DEPOSIT_BAL', 0):,.0f}<br>
			State: {customer.get('BILLINGSTATE', 'N/A')}<br>
			Type: {customer.get('TYPE', 'N/A')}
			""")
		
		fig.add_trace(go.Scattermapbox(
			lat=df['LAT_NUM'],
			lon=df['LON_NUM'],
			mode='markers',
			marker=dict(
				size=8,
				color=color,
				symbol='circle'
			),
			hovertemplate='%{text}<extra></extra>',
			text=hover_text,
			name=f"Form {form_id} ({len(df)} customers)",
			showlegend=True
		))
	
	# Calculate center point
	all_lats = []
	all_lons = []
	for form_id, df in form_results.items():
		if not df.empty:
			all_lats.extend(df['LAT_NUM'].tolist())
			all_lons.extend(df['LON_NUM'].tolist())
	
	if all_lats:
		center_lat = sum(all_lats) / len(all_lats)
		center_lon = sum(all_lons) / len(all_lons)
		zoom = 6
	else:
		center_lat = 39.8283
		center_lon = -98.5795
		zoom = 4
	
	fig.update_layout(
		mapbox=dict(
			style="open-street-map",
			center=dict(lat=center_lat, lon=center_lon),
			zoom=zoom
		),
		height=500,
		margin=dict(l=0, r=0, t=0, b=0),
		showlegend=True,
		legend=dict(
			yanchor="top",
			y=0.99,
			xanchor="left",
			x=0.01,
			bgcolor="rgba(255,255,255,0.8)"
		)
	)
	
	return fig

def to_excel(form_results):
	output = BytesIO()
	with pd.ExcelWriter(output , engine='openpyxl') as writer:
		for form_id, df in form_results.items():
			df.to_excel(writer , sheet_name = f"Form_{form_id}", index = False)
	output.seek(0)
	return output

def data_filteration(customer_data, branch_data, banker_data, form_id):
	st.subheader(f"Form {form_id}")
	
	# Select AU Section
	with st.expander("Select AU", expanded=True):
		col1, col2, col3 = st.columns(3)
		
		with col1:
			state = st.selectbox(f"State (Form {form_id})", branch_data['STATECODE'].dropna().unique(), key=f"State_{form_id}")
			
		filter_data = branch_data[branch_data['STATECODE'] == state]
		
		with col2:
			city = st.selectbox(f"City (Form {form_id})", filter_data['CITY'].dropna().unique(), key=f"City_{form_id}")
			
		au_options = filter_data[filter_data['CITY'] == city]['AU'].dropna().unique()
		with col3:
			selected_au = st.selectbox(f"AU (Form {form_id})", au_options, key = f"AU_{form_id}")
	
	# Select Customers Section
	with st.expander("Select Customers", expanded=True):
		col1, col2, col3 = st.columns(3)
		
		with col1:
			role_options = list(customer_data['TYPE'].dropna().unique())
			role = st.multiselect(f"Role (Form {form_id})", role_options, key=f"Role_{form_id}")
			if not role:
				role = None
		
		with col2:
			cust_state_options = list(customer_data['BILLINGSTATE'].dropna().unique())
			cust_state = st.multiselect(f"Customer State (Form {form_id})", cust_state_options, key=f"state_{form_id}")
			if not cust_state:
				cust_state = None
		
		with col3:
			cust_portcd = st.multiselect(f"Portfolio Code (Form {form_id})", customer_data['PORT_CODE'].dropna().unique(), key=f"port_cd_{form_id}")
			if not cust_portcd:
				cust_portcd = None
		
		col4, col5, col6 = st.columns(3)
		with col4:
			max_dist = st.slider(f"Max Distance (Form {form_id})", 1, 100, 20, key=f"Distance_{form_id}")
		with col5:
			min_rev = st.slider(f"Minimum Revenue (Form {form_id})", 0, 20000, 5000, step=1000, key=f"revenue_{form_id}")
		with col6:
			min_deposit = st.slider(f"Minimum Deposit (Form {form_id})", 0, 200000, 100000, step=5000, key=f"deposit_{form_id}")
	
	# Get AU data
	AU_row = branch_data[branch_data['AU'] == int(selected_au)].iloc[0]
	AU_lat = AU_row['BRANCH_LAT_NUM']
	AU_lon = AU_row['BRANCH_LON_NUM']
	
	# Filter customers by distance box
	box_lat = max_dist/111
	box_lon = max_dist/ (111 * np.cos(np.radians(AU_lat)))
	
	customer_data_boxed = customer_data[(customer_data['LAT_NUM'] >= AU_lat - box_lat) &
										(customer_data['LAT_NUM'] <= AU_lat + box_lat) &
										(customer_data['LON_NUM'] <= AU_lon + box_lon) &
										(customer_data['LON_NUM'] >= AU_lon - box_lon)]
	
	# Calculate distances
	customer_data_boxed['Distance'] = customer_data_boxed.apply(
		lambda row: haversine_distance(row['LAT_NUM'], row['LON_NUM'], AU_lat, AU_lon), axis=1
	)
	
	customer_data_boxed = customer_data_boxed.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
	filtered_data = customer_data_boxed.merge(banker_data, on="PORT_CODE", how='left')
	
	# Apply distance filter for all roles except CENTRALIZED
	if role is None or (role is not None and not any(r.lower().strip() == 'centralized' for r in role)):
		filtered_data = filtered_data[filtered_data['Distance'] <= int(max_dist)]
	
	# Apply role-specific filters
	if role is not None:
		filtered_data['TYPE_CLEAN'] = filtered_data['TYPE'].fillna('').str.strip().str.lower()
		role_clean = [r.strip().lower() for r in role]
		filtered_data = filtered_data[filtered_data['TYPE_CLEAN'].isin(role_clean)]
		filtered_data = filtered_data.drop('TYPE_CLEAN', axis=1)
	
	# Apply other filters
	filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
	filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
	
	if cust_state is not None:
		filtered_data = filtered_data[filtered_data['BILLINGSTATE'].isin(cust_state)]
	
	if cust_portcd is not None:
		filtered_data = filtered_data[filtered_data['PORT_CODE'].isin(cust_portcd)]
	
	# Create and display map side by side with table
	if not filtered_data.empty:
		# Create two columns for side-by-side layout
		col1, col2 = st.columns([1, 1])
		
		with col1:
			st.subheader("Portfolio Summary & Customer Selection")
			
			# Create portfolio summary table with selection
			portfolio_summary = []
			
			# Initialize form controls for this form if not exists
			if form_id not in st.session_state.form_controls:
				st.session_state.form_controls[form_id] = {}
			
			# Group by portfolio for assigned customers
			grouped = filtered_data[filtered_data['PORT_CODE'].notna()].groupby("PORT_CODE")
			
			for pid, group in grouped:
				total_customer = len(customer_data[customer_data["PORT_CODE"] == pid])
				
				# Determine portfolio type from the banker data or customer type
				portfolio_type = "Unknown"
				if not group.empty:
					# Get the most common type for this portfolio (excluding Unmanaged)
					types = group[group['TYPE'] != 'Unmanaged']['TYPE'].value_counts()
					if not types.empty:
						portfolio_type = types.index[0]
				
				# Initialize form controls
				st.session_state.form_controls[form_id][pid] = {"n": len(group), "exclude": []}
				
				portfolio_summary.append({
					'Portfolio ID': pid,
					'Portfolio Type': portfolio_type,
					'Total Customers': total_customer,
					'Available': len(group),
					'Select': len(group)  # Default to all available
				})
			
			# Add row for Unmanaged customers (not part of any portfolio)
			unmanaged_customers = filtered_data[
				(filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
				(filtered_data['PORT_CODE'].isna())
			]
			
			if not unmanaged_customers.empty:
				# Initialize form controls for unmanaged
				st.session_state.form_controls[form_id]['UNMANAGED'] = {"n": len(unmanaged_customers), "exclude": []}
				
				portfolio_summary.append({
					'Portfolio ID': 'UNMANAGED',
					'Portfolio Type': 'Unmanaged',
					'Total Customers': len(customer_data[
						(customer_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
						(customer_data['PORT_CODE'].isna())
					]),
					'Available': len(unmanaged_customers),
					'Select': len(unmanaged_customers)  # Default to all available
				})
			
			# Display portfolio summary table with selection
			portfolio_df = pd.DataFrame(portfolio_summary)
			
			# Sort by Available column in descending order
			portfolio_df = portfolio_df.sort_values('Available', ascending=False).reset_index(drop=True)
			
			# Create editable dataframe using st.data_editor
			edited_df = st.data_editor(
				portfolio_df,
				column_config={
					"Portfolio ID": st.column_config.TextColumn("Portfolio ID", disabled=True),
					"Portfolio Type": st.column_config.TextColumn("Portfolio Type", disabled=True),
					"Total Customers": st.column_config.NumberColumn("Total Customers", disabled=True),
					"Available": st.column_config.NumberColumn("Available", disabled=True),
					"Select": st.column_config.NumberColumn(
						"Select",
						help="Number of customers to select from this portfolio",
						min_value=0,
						step=1
					)
				},
				hide_index=True,
				use_container_width=True,
				height=500,
				key=f"portfolio_editor_{form_id}"
			)
			
			# Update form controls based on edited values
			for idx, row in edited_df.iterrows():
				pid = row['Portfolio ID']
				if pid in st.session_state.form_controls[form_id]:
					max_available = row['Available']
					selected = min(int(row['Select']), max_available)  # Ensure not exceeding available
					st.session_state.form_controls[form_id][pid]["n"] = selected
					st.session_state.form_controls[form_id][pid]["exclude"] = []
		
		with col2:
			st.subheader("Geographic Distribution")
			
			au_df = pd.DataFrame([AU_row])
			map_fig = create_interactive_map(filtered_data, au_df, max_dist, form_id)
			st.plotly_chart(map_fig, use_container_width=True)
		
		# Display statistics below the two columns
		st.subheader("Summary Statistics")
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			st.metric("Total Customers", len(filtered_data))
		with col2:
			st.metric("Avg Distance", f"{filtered_data['Distance'].mean():.1f} km")
		with col3:
			st.metric("Average Revenue", f"${filtered_data['BANK_REVENUE'].mean():,.0f}")
		with col4:
			st.metric("Average Deposits", f"${filtered_data['DEPOSIT_BAL'].mean():,.0f}")
			
	else:
		st.info("No customers available for selection with current filters.")
	
	return [filtered_data, role, city, state, max_dist, selected_au]

def recommend_reassignment(form_res: dict) -> pd.DataFrame:
	combine_df = pd.concat([df.assign(original_form = form_id) for form_id , df in form_res.items()], ignore_index = True)
	
	au_map = { form_id: ( df["BRANCH_LAT_NUM"].iloc[0], df["BRANCH_LON_NUM"].iloc[0])
				for form_id , df in form_res.items()
				if not df.empty}
				
	records = []
	for _, row in combine_df.iterrows():
		best_form = None
		min_dist = float("inf")
		for form_id , (au_lat , au_lon) in au_map.items():
			dist = haversine_distance(row['LAT_NUM'], row['LON_NUM'] , au_lat , au_lon)
			if dist < min_dist:
				best_form = form_id
				min_dist = dist
				
		row_data = row.to_dict()
		row_data['recommended_form'] = best_form
		row_data['recommended_dist'] = min_dist
		records.append(row_data)
		
	return pd.DataFrame(records)

#------------------------Streamlit App---------------------------------------------------------------
st.set_page_config("Portfolio Creation tool", layout="wide")

# Custom CSS for header styling
st.markdown("""
<style>
.header-container {
    background-color: rgb(215, 30, 40);
    padding: 1rem 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.header-title {
    color: white;
    font-size: 2.5rem;
    font-weight: bold;
    margin: 0;
}
.header-input {
    background-color: white;
    border-radius: 5px;
    padding: 0.5rem;
    min-width: 200px;
}
</style>
""", unsafe_allow_html=True)

# Header with colored background
st.markdown('<div class="header-container"><h1 class="header-title">Portfolio Creation Tool</h1></div>', unsafe_allow_html=True)

# Number of portfolios input below header
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    num_forms = st.number_input("Number of Portfolios", min_value=1, max_value=10, value=1, key="num_portfolios")

page = st.selectbox("Select Page", ["Portfolio Assignment", "Portfolio Mapping"])

# Initialize session state
if 'form_results' not in st.session_state:
	st.session_state.form_results = {}
	
if 'form_controls' not in st.session_state:
	st.session_state.form_controls = {}
	
if 'recommend_reassignment' not in st.session_state:
	st.session_state.recommend_reassignment = {}

# Load data from local CSV files
@st.cache_data
def load_data():
	customer_data = pd.read_csv("customer_data.csv")
	banker_data = pd.read_csv("banker_data.csv")
	branch_data = pd.read_csv("branch_data.csv")
	return customer_data, banker_data, branch_data

# Load data on app startup
customer_data, banker_data, branch_data = load_data()
data = merge_dfs(customer_data, banker_data, branch_data)

if page == "Portfolio Assignment":
	
	tab_titles = [f"Form {i}" for i in range(1, num_forms+1)]
	tabs = st.tabs(tab_titles)
	
	form_results = {}
	
	for form_id, tab in enumerate(tabs, start=1):
		with tab:
			filtered_data, role, city, state, max_dist, selected_au = data_filteration(
				customer_data, branch_data, banker_data, form_id
			)
			
			filtered_data['FormID'] = form_id
			
			# Initialize form controls for this form if not exists
			if form_id not in st.session_state.form_controls:
				st.session_state.form_controls[form_id] = {}
			
			# Clean up form controls to only include valid portfolio IDs
			if not filtered_data.empty:
				valid_pids = set(filtered_data['PORT_CODE'].unique())
				st.session_state.form_controls[form_id] = {
					pid: val for pid, val in st.session_state.form_controls[form_id].items()
					if pid in valid_pids
				}
			
			# Conflict detection
			assigned = {cid for fid, df in st.session_state.form_results.items() 
					   if fid != form_id for cid in df['CG_ECN']}
			conflicts = filtered_data[filtered_data['CG_ECN'].isin(assigned)]
			if not conflicts.empty:
				st.warning(f"{len(conflicts)} customers already assigned and removed")
				filtered_data = filtered_data[~filtered_data['CG_ECN'].isin(assigned)]
			
			if st.button(f"Save Form {form_id}", key=f"save_{form_id}"):
				if not filtered_data.empty:
					result = []
					au_row = branch_data[branch_data['AU'] == selected_au].iloc[0]
					b_au, b_lat, b_lon = au_row['AU'], au_row['BRANCH_LAT_NUM'], au_row['BRANCH_LON_NUM']
					
					# Handle regular portfolios
					grouped = filtered_data[filtered_data['PORT_CODE'].notna()].groupby("PORT_CODE")
					for pid, group in grouped:
						if pid in st.session_state.form_controls[form_id]:
							ctrl = st.session_state.form_controls[form_id][pid]
							selected_customers = group[~group["CG_ECN"].isin(ctrl["exclude"])]
							top_n = selected_customers.sort_values(by='Distance').head(ctrl["n"])
							top_n['AU'] = b_au
							top_n['BRANCH_LAT_NUM'] = b_lat
							top_n['BRANCH_LON_NUM'] = b_lon
							result.append(top_n)
					
					# Handle unmanaged customers
					unmanaged_customers = filtered_data[
						(filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged') |
						(filtered_data['PORT_CODE'].isna())
					]
					if not unmanaged_customers.empty and 'UNMANAGED' in st.session_state.form_controls[form_id]:
						ctrl = st.session_state.form_controls[form_id]['UNMANAGED']
						selected_unmanaged = unmanaged_customers[~unmanaged_customers["CG_ECN"].isin(ctrl["exclude"])]
						top_n_unmanaged = selected_unmanaged.sort_values(by='Distance').head(ctrl["n"])
						top_n_unmanaged['AU'] = b_au
						top_n_unmanaged['BRANCH_LAT_NUM'] = b_lat
						top_n_unmanaged['BRANCH_LON_NUM'] = b_lon
						result.append(top_n_unmanaged)
					
					form_df = pd.concat(result) if result else pd.DataFrame()
					
					# Handle conflicts
					conflicted_ids = []
					reassigned_rows = []
					for cid in form_df["CG_ECN"]:
						for other_fid, other_df in st.session_state.form_results.items():
							if other_fid == form_id:
								continue
							if cid in other_df["CG_ECN"].values:
								old_row = other_df[other_df["CG_ECN"] == cid].iloc[0]
								new_row = form_df[form_df["CG_ECN"] == cid].iloc[0]
								
								if new_row['Distance'] < old_row['Distance']:
									st.session_state.form_results[other_fid] = other_df[other_df["CG_ECN"] != cid]
									reassigned_rows.append(new_row)
									conflicted_ids.append((cid, other_fid, old_row['Distance'], form_id, new_row['Distance']))
								else:
									form_df = form_df[form_df["CG_ECN"] != cid]
					
					if reassigned_rows:
						form_df = pd.concat([form_df, pd.DataFrame(reassigned_rows)])
					
					st.session_state.form_results[form_id] = form_df
					st.success(f"Form {form_id} saved with {len(form_df)} customers")
					
					if conflicted_ids:
						with st.expander("Conflict resolutions (Auto Handled)"):
							conflict_df = pd.DataFrame(conflicted_ids, columns=[
								"CG_ECN", "Previous Form", "Previous Distance", "Assigned Form", "New Distance"
							])
							st.warning("Some customers were reassigned based on distance:")
							st.dataframe(conflict_df, use_container_width=True)
				else:
					st.error("No customers to save. Please adjust your filters.")
	
	st.markdown("----")
	
	if st.session_state.form_results:
		st.subheader("Combined Geographic View")
		combined_map = create_combined_map(st.session_state.form_results, branch_data)
		if combined_map:
			st.plotly_chart(combined_map, use_container_width=True)
	
	if st.button("Recommended form"):
		if st.session_state.form_results:
			rec_df = recommend_reassignment(st.session_state.form_results)
			st.session_state.recommend_reassignment = rec_df
			st.subheader("Form reassignment")
			st.dataframe(st.session_state.recommend_reassignment)
		else:
			st.warning("No forms saved yet. Please save at least one form first.")
	
	if st.button("Save all forms"):
		if st.session_state.form_results:
			combined_result = pd.concat(st.session_state.form_results.values())
			st.session_state.final_result = combined_result
			st.success("All Forms are saved successfully")
			st.write(st.session_state.final_result)
			
			# Download button
			excel_buffer = to_excel(st.session_state.form_results)
			st.download_button(
				label="Download Excel Report",
				data=excel_buffer,
				file_name="portfolio_assignments.xlsx",
				mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
			)
		else:
			st.warning("No forms to save. Please create and save at least one form first.")

elif page == "Portfolio Mapping":
	st.subheader("Portfolio Mapping")
	
	# Portfolio mapping functionality can be added here
	st.info("Portfolio Mapping functionality coming soon...")
	
	col1, col2 = st.columns(2)
	
	with col1:
		st.subheader("Customer Distribution by Type")
		if not data.empty and 'TYPE' in data.columns:
			type_counts = data['TYPE'].value_counts()
			st.bar_chart(type_counts)
	
	with col2:
		st.subheader("Customer Distribution by State")
		if not data.empty and 'BILLINGSTATE' in data.columns:
			state_counts = data['BILLINGSTATE'].value_counts().head(10)
			st.bar_chart(state_counts)
	
	if not data.empty:
		st.subheader("Summary Statistics")
		col1, col2, col3, col4 = st.columns(4)
		
		with col1:
			st.metric("Total Customers", len(data))
		with col2:
			if 'BANK_REVENUE' in data.columns:
				st.metric("Total Revenue", f"${data['BANK_REVENUE'].sum():,.0f}")
		with col3:
			if 'DEPOSIT_BAL' in data.columns:
				st.metric("Total Deposits", f"${data['DEPOSIT_BAL'].sum():,.0f}")
		with col4:
			if 'PORT_CODE' in data.columns:
				st.metric("Unique Portfolios", data['PORT_CODE'].nunique())
