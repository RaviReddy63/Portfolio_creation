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
			role_options = ['All Roles'] + list(customer_data['TYPE'].dropna().unique())
			role = st.selectbox(f"Role (Form {form_id})", role_options, key=f"Role_{form_id}")
			if role == 'All Roles':
				role = None
		
		with col2:
			cust_state_options = ['All States'] + list(customer_data['BILLINGSTATE'].dropna().unique())
			cust_state = st.selectbox(f"Customer State (Form {form_id})", cust_state_options, key=f"state_{form_id}")
			if cust_state == 'All States':
				cust_state = None
		
		with col3:
			cust_portcd = st.multiselect(f"Portfolio Code (Form {form_id})", customer_data['PORT_CODE'].dropna().unique(), key=f"port_cd_{form_id}")
			if not cust_portcd:
				cust_portcd = None
		
		col4, col5 = st.columns(2)
		with col4:
			max_dist = st.slider(f"Max Distance (Form {form_id})", 20, 100, 60, key=f"Distance_{form_id}")
		with col5:
			min_rev = st.slider(f"Minimum Revenue (Form {form_id})", 0, 20000, 5000, key=f"revenue_{form_id}")
		
		min_deposit = st.slider(f"Minimum Deposit (Form {form_id})", 0, 200000, 100000, key=f"deposit_{form_id}")
	
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
	if role is None or (role is not None and role.lower().strip() != 'centralized'):
		filtered_data = filtered_data[filtered_data['Distance'] <= int(max_dist)]
	
	# Apply role-specific filters
	if role is not None:
		filtered_data['TYPE_CLEAN'] = filtered_data['TYPE'].fillna('').str.strip().str.lower()
		role_clean = role.strip().lower()
		filtered_data = filtered_data[filtered_data['TYPE_CLEAN'] == role_clean]
		filtered_data = filtered_data.drop('TYPE_CLEAN', axis=1)
	
	# Apply other filters
	filtered_data = filtered_data[filtered_data['BANK_REVENUE'] >= min_rev]
	filtered_data = filtered_data[filtered_data['DEPOSIT_BAL'] >= min_deposit]
	
	if cust_state is not None:
		filtered_data = filtered_data[filtered_data['BILLINGSTATE'] == cust_state]
	
	if cust_portcd is not None:
		filtered_data = filtered_data[filtered_data['PORT_CODE'].isin(cust_portcd)]
	
	# Create and display map
	if not filtered_data.empty:
		st.subheader("Geographic Distribution")
		
		au_df = pd.DataFrame([AU_row])
		map_fig = create_interactive_map(filtered_data, au_df, max_dist, form_id)
		st.plotly_chart(map_fig, use_container_width=True)
		
		# Display statistics
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			st.metric("Total Customers", len(filtered_data))
		with col2:
			st.metric("Avg Distance", f"{filtered_data['Distance'].mean():.1f} km")
		with col3:
			st.metric("Total Revenue", f"${filtered_data['BANK_REVENUE'].sum():,.0f}")
		with col4:
			st.metric("Total Deposits", f"${filtered_data['DEPOSIT_BAL'].sum():,.0f}")
		
		# Additional metrics for Unassigned and Unmanaged
		if len(filtered_data) > 0:
			unassigned_total = len(filtered_data[filtered_data['TYPE'].str.lower().str.strip() == 'unassigned'])
			unmanaged_total = len(filtered_data[filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged'])
			
			if unassigned_total > 0 or unmanaged_total > 0:
				st.markdown("#### Special Customer Categories")
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.metric("🔴 Unassigned", unassigned_total)
				with col2:
					st.metric("🟡 Unmanaged", unmanaged_total)
				with col3:
					if unassigned_total > 0:
						unassigned_revenue = filtered_data[filtered_data['TYPE'].str.lower().str.strip() == 'unassigned']['BANK_REVENUE'].sum()
						st.metric("Unassigned Revenue", f"${unassigned_revenue:,.0f}")
				with col4:
					if unmanaged_total > 0:
						unmanaged_revenue = filtered_data[filtered_data['TYPE'].str.lower().str.strip() == 'unmanaged']['BANK_REVENUE'].sum()
						st.metric("Unmanaged Revenue", f"${unmanaged_revenue:,.0f}")
		
		with st.expander("Sample Filtered Data"):
			st.dataframe(filtered_data[['CG_ECN', 'TYPE', 'Distance', 'BANK_REVENUE', 'DEPOSIT_BAL', 'BILLINGSTATE']].head(10))
	else:
		st.warning("No customers found matching the current filters. Try adjusting your criteria.")
	
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
st.title("Portfolio creation tool")
page = st.selectbox("Select Page", ["Portfolio Assignment", "Portfolio Mapping"])

#------------------------File Upload----------------------------------------------------------------------------------
st.sidebar.header("Upload CSV files")
cust_file = st.sidebar.file_uploader("Customer Data", type=["csv"])
banker_file = st.sidebar.file_uploader("Banker Data", type=["csv"])
branch_file = st.sidebar.file_uploader("Branch Data", type=["csv"])

if 'form_results' not in st.session_state:
	st.session_state.form_results = {}
	
if 'form_controls' not in st.session_state:
	st.session_state.form_controls = {}
	
if 'recommend_reassignment' not in st.session_state:
	st.session_state.recommend_reassignment = {}

if page == "Portfolio Assignment":
	if cust_file and banker_file and branch_file:
		customer_data = pd.read_csv(cust_file)
		banker_data = pd.read_csv(banker_file)
		branch_data = pd.read_csv(branch_file)
		data = merge_dfs(customer_data, banker_data, branch_data)
		
		st.sidebar.header("Form Configuration")
		num_forms = st.sidebar.number_input("Number of Portfolios", min_value=1, max_value=10, value=1)
		
		tab_titles = [f"Form {i}" for i in range(1, num_forms+1)]
		tabs = st.tabs(tab_titles)
		
		assigned_customers = set()
		form_results = {}
		
		for form_id, tab in enumerate(tabs, start=1):
			with tab:
				filtered_data, role, city, state, max_dist, selected_au = data_filteration(
					customer_data, branch_data, banker_data, form_id
				)
				
				filtered_data['FormID'] = form_id
				st.session_state.form_controls.setdefault(form_id, {})
				
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
				
				if not filtered_data.empty:
					grouped = filtered_data.groupby("PORT_CODE")
					
					for pid, group in grouped:
						total_customer = len(data[data["PORT_CODE"] == pid])
						
						# Count unassigned and unmanaged customers in this portfolio
						unassigned_count = len(group[group['TYPE'].str.lower().str.strip() == 'unassigned'])
						unmanaged_count = len(group[group['TYPE'].str.lower().str.strip() == 'unmanaged'])
						
						# Create a more detailed title
						title_parts = [f"Portfolio {pid}"]
						title_parts.append(f"{total_customer} total customers")
						title_parts.append(f"{len(group)} available")
						
						if unassigned_count > 0:
							title_parts.append(f"{unassigned_count} Unassigned")
						if unmanaged_count > 0:
							title_parts.append(f"{unmanaged_count} Unmanaged")
						
						portfolio_title = " - ".join(title_parts)
						
						with st.expander(portfolio_title):
							st.session_state.form_controls[form_id][pid] = {"n": len(group), "exclude": []}
							ctrl = st.session_state.form_controls[form_id][pid]
							ctrl["n"] = st.number_input(
								f"Top N customers to select from Portfolio {pid}",
								min_value=0,
								max_value=len(group),
								value=min(ctrl["n"], len(group)),
								key=f"slider_{form_id}_{pid}"
							)
							ctrl["exclude"] = []
							
							# Show breakdown by customer type
							if len(group) > 0:
								st.write("**Customer Type Breakdown:**")
								type_breakdown = group['TYPE'].value_counts()
								col1, col2 = st.columns(2)
								
								with col1:
									for customer_type, count in type_breakdown.items():
										st.write(f"• {customer_type}: {count}")
								
								with col2:
									if unassigned_count > 0 or unmanaged_count > 0:
										st.write("**Special Categories:**")
										if unassigned_count > 0:
											st.write(f"🔴 Unassigned: {unassigned_count}")
										if unmanaged_count > 0:
											st.write(f"🟡 Unmanaged: {unmanaged_count}")
							
							st.write("**Sample Customers:**")
							st.dataframe(group[['CG_ECN', 'TYPE', 'Distance', 'BANK_REVENUE', 'DEPOSIT_BAL']].head())
				else:
					st.info("No customers available for selection with current filters.")
				
				if st.button(f"Save Form {form_id}", key=f"save_{form_id}"):
					if not filtered_data.empty:
						result = []
						au_row = branch_data[branch_data['AU'] == selected_au].iloc[0]
						b_au, b_lat, b_lon = au_row['AU'], au_row['BRANCH_LAT_NUM'], au_row['BRANCH_LON_NUM']
						
						grouped = filtered_data.groupby("PORT_CODE")
						for pid, group in grouped:
							if pid in st.session_state.form_controls[form_id]:
								ctrl = st.session_state.form_controls[form_id][pid]
								selected_customers = group[~group["CG_ECN"].isin(ctrl["exclude"])]
								top_n = selected_customers.sort_values(by='Distance').head(ctrl["n"])
								top_n['AU'] = b_au
								top_n['BRANCH_LAT_NUM'] = b_lat
								top_n['BRANCH_LON_NUM'] = b_lon
								result.append(top_n)
						
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
		
		# Live tracking
		from collections import defaultdict
		tracker = defaultdict(int)
		data_for_pivot = []
		pid_track = {}
		already_assigned = {cid for df in st.session_state.form_results.values() for cid in df['CG_ECN']}
		
		for fid, controls in st.session_state.form_controls.items():
			pid_track[fid] = 0
			for pid, ctrl in controls.items():
				df_pid = data[data['PORT_CODE'] == pid]
				valid = df_pid[~df_pid['CG_ECN'].isin(ctrl['exclude']) &
							  ~df_pid['CG_ECN'].isin(already_assigned)]
				
				sel_count = min(len(valid), ctrl['n'])
				if sel_count > 0:
					tracker[pid] += sel_count
					data_for_pivot.append([pid, "Form"+str(fid), sel_count])
				
				pid_track[fid] += sel_count
		
		tracker_df = pd.DataFrame(data_for_pivot, columns=["PortfolioID", "FormID", "Customers"])
		if not tracker_df.empty:
			tracker_df = pd.pivot_table(tracker_df, values="Customers", index="PortfolioID", columns="FormID", aggfunc='sum')
		
		st.markdown("-----")
		st.sidebar.subheader("Live Portfolio Tracker")
		if tracker:
			tdf = pd.DataFrame([{"PortfolioID": pid, "Customers selected": n} for pid, n in tracker.items()])
			st.sidebar.dataframe(tdf)
		
		# Show Unassigned/Unmanaged summary in sidebar
		st.sidebar.markdown("**Special Customer Categories**")
		total_unassigned = 0
		total_unmanaged = 0
		
		for fid, controls in st.session_state.form_controls.items():
			form_unassigned = 0
			form_unmanaged = 0
			
			for pid, ctrl in controls.items():
				df_pid = data[data['PORT_CODE'] == pid]
				valid = df_pid[~df_pid['CG_ECN'].isin(ctrl['exclude']) &
							  ~df_pid['CG_ECN'].isin(already_assigned)]
				
				# Count unassigned and unmanaged in this portfolio for this form
				pid_unassigned = len(valid[valid['TYPE'].str.lower().str.strip() == 'unassigned'])
				pid_unmanaged = len(valid[valid['TYPE'].str.lower().str.strip() == 'unmanaged'])
				
				form_unassigned += min(pid_unassigned, ctrl['n'])
				form_unmanaged += min(pid_unmanaged, ctrl['n'])
			
			total_unassigned += form_unassigned
			total_unmanaged += form_unmanaged
			
			if form_unassigned > 0 or form_unmanaged > 0:
				st.sidebar.write(f"Form {fid}:")
				if form_unassigned > 0:
					st.sidebar.write(f"  🔴 Unassigned: {form_unassigned}")
				if form_unmanaged > 0:
					st.sidebar.write(f"  🟡 Unmanaged: {form_unmanaged}")
		
		if total_unassigned > 0 or total_unmanaged > 0:
			st.sidebar.markdown("**Total Special Categories:**")
			if total_unassigned > 0:
				st.sidebar.write(f"🔴 Total Unassigned: {total_unassigned}")
			if total_unmanaged > 0:
				st.sidebar.write(f"🟡 Total Unmanaged: {total_unmanaged}")
		
		st.sidebar.markdown("**Customer count per Form**")
		if not tracker_df.empty:
			st.sidebar.dataframe(tracker_df)
		for fid, counts in pid_track.items():
			st.sidebar.write(f"Form {fid} → {counts} Customer(s)")
		
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
	
	else:
		st.info("Please upload all 3 files to begin")

elif page == "Portfolio Mapping":
	st.subheader("Portfolio Mapping")
	
	if cust_file and banker_file and branch_file:
		customer_data = pd.read_csv(cust_file)
		banker_data = pd.read_csv(banker_file)
		branch_data = pd.read_csv(branch_file)
		data = merge_dfs(customer_data, banker_data, branch_data)
		
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
	else:
		st.info("Please upload all 3 files to begin portfolio mapping")
